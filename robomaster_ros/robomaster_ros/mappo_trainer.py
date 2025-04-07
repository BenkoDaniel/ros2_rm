import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
from robomaster_soccer_env import RobomasterSoccerEnv

# Environment setup
def make_env():
    env = RobomasterSoccerEnv()
    return env

env = make_env()
obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
action_space_type = 'continuous'
action_dim = 3
action_high = env.action_space(env.possible_agents[0]).high[0]
action_low = env.action_space(env.possible_agents[0]).low[0]

n_agents = len(env.possible_agents)

# Hyperparameters
LR_ACTOR = 1e-5
LR_CRITIC = 1e-5
GAMMA = 0.99
LAMDA = 0.95
EPS_CLIP = 0.3
K_EPOCHS = 10
BUFFER_SIZE = 10000
BATCH_SIZE = 64
UPDATE_INTERVAL = 100
ENTROPY_COEF = 0.1
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_space_type):
        super(ActorCritic, self).__init__()
        self.action_space_type = action_space_type
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Continuous action head
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Linear(64, action_dim)
        
        # Critic head
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.shared_layers(x)
        return x
        
    def act(self, x):
        hidden = self.forward(x)
        mean = self.actor_mean(hidden)
        log_std = self.actor_logstd(hidden)
        
        # Enforce minimum exploration (NEW)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) + 1e-3  # Add small epsilon
        
        dist = Normal(mean, std)
        action = dist.sample()
        
        # Clip actions to valid range but keep gradients
        action = torch.tanh(action) * (action_high - action_low)/2 + (action_high + action_low)/2
        
        log_prob = dist.log_prob(action).sum(-1)
        action_np = np.squeeze(action).flatten()
        return action_np.detach().cpu().numpy(), log_prob.detach()
        

    
    def evaluate(self, x, action):
        hidden = self.forward(x)

        mean = self.actor_mean(hidden)
        log_std = self.actor_logstd(hidden)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) + 1e-6
            
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)  # Sum over action dimensions
        entropy = dist.entropy().sum(-1)  # Sum over action dimensions
        value = self.critic(hidden).squeeze(-1)  # Remove extra dimension
        
        return log_prob, value, entropy

# MAPPO Agent
class MAPPO:
    def __init__(self, n_agents, obs_dim, action_dim, action_space_type):
        self.n_agents = n_agents
        self.action_space_type = action_space_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policies = [ActorCritic(obs_dim, action_dim, action_space_type) for _ in range(n_agents)]
        self.old_policies = [ActorCritic(obs_dim, action_dim, action_space_type) for _ in range(n_agents)]

        for policy in self.policies:
            for layer in policy.shared_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    layer.bias.data.zero_()
            nn.init.orthogonal_(policy.actor_mean.weight, gain=0.01)
            policy.actor_mean.bias.data.zero_()
            nn.init.orthogonal_(policy.actor_logstd.weight, gain=0.01)
            policy.actor_logstd.bias.data.zero_()
        
        # Initialize old policies with same parameters
        for new_policy, old_policy in zip(self.policies, self.old_policies):
            old_policy.load_state_dict(new_policy.state_dict())
        
        self.optimizers = [optim.Adam([
            {'params': policy.shared_layers.parameters(), 'lr': LR_CRITIC},
            {'params': policy.actor.parameters() if action_space_type == 'discrete' else [
                param for name, param in policy.named_parameters() 
                if 'actor_mean' in name or 'actor_logstd' in name
            ], 'lr': LR_ACTOR},
            {'params': policy.critic.parameters(), 'lr': LR_CRITIC}
        ]) for policy in self.policies]
        
        self.buffer = deque(maxlen=BUFFER_SIZE)
        
    def update_old_policies(self):
        for i in range(self.n_agents):
            self.old_policies[i].load_state_dict(self.policies[i].state_dict())
    
    def act(self, obs, agent_idx):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.old_policies[agent_idx].act(obs_tensor)
        return action, log_prob
    
    def store_transition(self, transition):
        self.buffer.append(transition)
    
    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        values = np.asarray(values)
        
        # Initialize arrays
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae = 0.0
        
        # Extend values with next_value for bootstrapping
        extended_values = np.append(values, next_value)
        
        # Calculate advantages in reverse order
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * (1 - dones[t]) * extended_values[t+1] - extended_values[t]
            last_gae = delta + GAMMA * LAMDA * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + extended_values[t]
        
        # Convert to tensors and normalize advantages
        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        returns = torch.as_tensor(returns, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        
        batch = random.sample(self.buffer, BATCH_SIZE)
        
        # Unpack the batch
        obs = torch.FloatTensor(np.array([t[0] for t in batch]))
        actions = torch.FloatTensor(np.array([t[1] for t in batch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in batch]))
        next_obs = torch.FloatTensor(np.array([t[3] for t in batch]))
        dones = torch.FloatTensor(np.array([t[4] for t in batch]))
        old_log_probs = torch.FloatTensor(np.array([t[5] for t in batch]))
        agent_indices = np.array([t[6] for t in batch])
        
        for agent_idx in range(self.n_agents):
            mask = (agent_indices == agent_idx)
            if not mask.any():
                continue
                
            agent_obs = obs[mask]
            agent_actions = actions[mask]
            agent_rewards = rewards[mask].cpu().numpy()
            agent_next_obs = next_obs[mask]
            agent_dones = dones[mask].cpu().numpy()
            agent_old_log_probs = old_log_probs[mask]
            
            with torch.no_grad():
                # Get value estimates
                _, old_values, _ = self.old_policies[agent_idx].evaluate(agent_obs, agent_actions)
                old_values = old_values.cpu().numpy().flatten()
                
                # Get bootstrap value for last state
                _, next_values, _ = self.old_policies[agent_idx].evaluate(agent_next_obs, agent_actions)
                next_value = next_values[-1].item() if len(next_values) > 0 else 0
                
                # Compute returns and advantages
                returns, advantages = self.compute_returns_and_advantages(
                    agent_rewards,
                    agent_dones,
                    old_values,
                    next_value
                )
            
            # Convert to tensors if needed
            returns = returns.to(self.device)
            advantages = advantages.to(self.device)
            
            # Optimize policy for K epochs
            for _ in range(K_EPOCHS):
                log_probs, values, entropy = self.policies[agent_idx].evaluate(agent_obs, agent_actions)
                
                # Calculate policy loss - ensure we're using means for scalar outputs
                ratios = torch.exp(log_probs - agent_old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss - ensure scalar output
                critic_loss = 0.5 * nn.MSELoss()(values.squeeze(), returns).mean()
                
                # Calculate entropy - ensure scalar
                entropy_loss = entropy.mean()
                
                # Total loss - scalar
                loss = actor_loss + critic_loss - ENTROPY_COEF * entropy_loss
                
                # Update policy
                self.optimizers[agent_idx].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policies[agent_idx].parameters(), max_norm=1.0)
                self.optimizers[agent_idx].step()

# Training loop
def train():
    mappo = MAPPO(n_agents, obs_dim, action_dim, action_space_type)
    episode_rewards = []
    
    try:
        for episode in range(1000):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                actions = {}
                log_probs = {}
                
                for i, agent in enumerate(env.agents):
                    action, log_prob = mappo.act(obs[agent], i)
                    if episode < 100:  # Add noise for first 100 episodes
                        action += np.random.normal(0, 0.2, size=action.shape)  # Adjust 0.2 as needed
                        action = np.clip(action, action_low, action_high)
                    actions[agent] = [float(x) for x in action]
                    log_probs[agent] = log_prob
                
                
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
                dones = {
                    "robot1": terminations["robot1"] or terminations["robot1"],
                    "robot2": terminations["robot2"] or terminations["robot2"]
                }
                done = terminations["robot1"] or terminations["robot2"] or truncations["robot1"] or truncations["robot2"]

                
                for i, agent in enumerate(env.agents):
                    transition = (
                        obs[agent], 
                        np.array(actions[agent]), 
                        rewards[agent], 
                        next_obs[agent], 
                        dones[agent], 
                        log_probs[agent], 
                        i
                    )
                    mappo.store_transition(transition)
                
                obs = next_obs
                episode_reward += sum(rewards.values())
                
                if len(mappo.buffer) >= BATCH_SIZE and len(mappo.buffer) % UPDATE_INTERVAL == 0:
                    mappo.update_old_policies()
                    mappo.update()
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode}, Reward: {episode_reward}")
            
            if (episode + 1) % 10 == 0:
                if not os.path.exists('models'):
                    os.makedirs('models')
                for i in range(n_agents):
                    torch.save(mappo.policies[i].state_dict(), f'models/agent_{i}_ep_{episode+1}.pth')
    
    except KeyboardInterrupt:
        print("Training interrupted. Saving models...")
        if not os.path.exists('models'):
            os.makedirs('models')
        for i in range(n_agents):
            torch.save(mappo.policies[i].state_dict(), f'models/agent_{i}_final.pth')
    
    finally:
        env.close()

if __name__ == "__main__":
    train()