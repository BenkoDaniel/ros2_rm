import os
import gym
import numpy as np
from pettingzoo.utils import parallel_to_aec
from torch import nn
import torch
import torch.optim as optim
from tqdm import trange
from datetime import datetime
import glob
from torch.utils.tensorboard import SummaryWriter

from robomaster_soccer_env import RobomasterSoccerEnv 

# === Actor-Critic networks ===
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, total_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# === Helper functions ===
def select_action(actor, obs):
    logits = actor(torch.tensor(obs, dtype=torch.float32))
    dist = torch.distributions.Normal(logits, 1.0)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)
    return action.detach().numpy(), log_prob

def compute_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        advantages[t] = last_adv = delta + gamma * gae_lambda * (1 - dones[t]) * last_adv
    return advantages

# ===  Load latest model checkpoint ===
def load_latest_models(model_dir, actors, critic):
    if not os.path.exists(model_dir):
        return 0  # No previous models

    runs = sorted(glob.glob(os.path.join(model_dir, "run_*")))
    if not runs:
        return 0

    latest_run = runs[-1]
    checkpoints = sorted(glob.glob(os.path.join(latest_run, "epoch_*")))
    if not checkpoints:
        return 0

    latest_ckpt = checkpoints[-1]
    epoch_num = int(latest_ckpt.split('_')[-1])

    for agent in actors:
        actor_path = os.path.join(latest_ckpt, f"actor_{agent}.pt")
        if os.path.exists(actor_path):
            actors[agent].load_state_dict(torch.load(actor_path))
    critic_path = os.path.join(latest_ckpt, "critic.pt")
    if os.path.exists(critic_path):
        critic.load_state_dict(torch.load(critic_path))

    print(f"Loaded latest models")
    return epoch_num

# === Save model checkpoint ===
def save_models(save_dir, epoch, actors, critic):
    ckpt_dir = os.path.join(save_dir, f"epoch_{epoch:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    for agent in actors:
        torch.save(actors[agent].state_dict(), os.path.join(ckpt_dir, f"actor_{agent}.pt"))
    torch.save(critic.state_dict(), os.path.join(ckpt_dir, "critic.pt"))

# === Training script ===
def train_mappo():
    # Set up environment
    env = RobomasterSoccerEnv()
    agents = env.possible_agents
    env.reset()

    obs_space = env.observation_space(agents[0])
    act_space = env.action_space(agents[0])
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    total_obs_dim = obs_dim * len(agents)

    # Create models and optimizers
    actors = {agent: Actor(obs_dim, act_dim) for agent in agents}
    critic = Critic(total_obs_dim)
    optimizers = {agent: optim.Adam(actors[agent].parameters(), lr=3e-4) for agent in agents}
    critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

    # Setup logging
    log_dir = "logs/mappo_soccer"
    writer = SummaryWriter(log_dir=log_dir)

    # Setup model saving
    model_root = "models"
    os.makedirs(model_root, exist_ok=True)
    current_run_dir = os.path.join(model_root, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(current_run_dir, exist_ok=True)

    # Try loading latest
    start_epoch = load_latest_models(model_root, actors, critic)

    # Training params
    n_epochs = 1000
    horizon = 200
    save_interval = 10

    for epoch in trange(start_epoch, n_epochs):
        # Initialize buffers
        obs_buffer = {agent: [] for agent in agents}
        action_buffer = {agent: [] for agent in agents}
        logprob_buffer = {agent: [] for agent in agents}
        reward_buffer = {agent: [] for agent in agents}
        done_buffer = {agent: [] for agent in agents}
        value_buffer = []

        obs, info = env.reset()
        done = {agent: False for agent in agents}

        for step in range(horizon):
            actions, log_probs, all_obs = {}, {}, []

            for agent in agents:
                if not done[agent]:
                    action, logp = select_action(actors[agent], obs[agent])
                    actions[agent] = action
                    log_probs[agent] = logp
                    all_obs.append(obs[agent])
                else:
                    actions[agent] = np.zeros(act_dim)
                    log_probs[agent] = 0.0
                    all_obs.append(np.zeros(obs_dim))

            value_input = np.concatenate(all_obs)
            value = critic(torch.tensor(value_input, dtype=torch.float32)).item()
            value_buffer.append(value)

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = terminations | truncations

            for agent in agents:
                obs_buffer[agent].append(obs[agent])
                action_buffer[agent].append(actions[agent])
                logprob_buffer[agent].append(log_probs[agent])
                reward_buffer[agent].append(rewards[agent])
                done_buffer[agent].append(dones[agent])

            obs = next_obs
            done = dones
            if all(dones.values()):
                break

        all_obs = [obs[agent] if not done[agent] else np.zeros(obs_dim) for agent in agents]
        value_input = np.concatenate(all_obs)
        final_value = critic(torch.tensor(value_input, dtype=torch.float32)).item()
        value_buffer.append(final_value)

        rewards = np.mean([reward_buffer[agent] for agent in agents], axis=0)
        dones = np.mean([done_buffer[agent] for agent in agents], axis=0)
        values = np.array(value_buffer)
        advantages = compute_advantages(rewards, values, dones)
        returns = advantages + values[:-1]

        # Critic update
        critic_loss_total = 0
        
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values[:-1], dtype=torch.float32)
        advantages = (returns - values).detach()
        torch.autograd.set_detect_anomaly(True)
        for _ in range(4):  # PPO epochs
            optimizers[agent].zero_grad()
            total_loss = 0.0

            for t in range(len(reward_buffer[agent])):
                obs_t = torch.tensor(obs_buffer[agent][t], dtype=torch.float32)
                action_t = torch.tensor(action_buffer[agent][t], dtype=torch.float32)
                old_logprob = torch.tensor(logprob_buffer[agent][t], dtype=torch.float32).clone().detach()

                logits = actors[agent](obs_t)
                dist = torch.distributions.Normal(logits, 1.0)
                logprob = dist.log_prob(action_t).sum()
                ratio = torch.exp(logprob - old_logprob)

                advantage = torch.tensor(advantages[t], dtype=torch.float32).clone().detach()
                clip_adv = torch.clamp(ratio, 0.8, 1.2) * advantage
                actor_loss = -torch.min(ratio * advantage, clip_adv)

                total_loss += actor_loss

            total_loss.backward(retain_graph=True)
            optimizers[agent].step()



        # Actor update
        for agent in agents:
            for _ in range(4):
                optimizers[agent].zero_grad()
                for t in range(len(reward_buffer[agent])):
                    obs_t = torch.tensor(obs_buffer[agent][t], dtype=torch.float32)
                    action_t = torch.tensor(action_buffer[agent][t], dtype=torch.float32)
                    old_logprob = logprob_buffer[agent][t]

                    logits = actors[agent](obs_t)
                    dist = torch.distributions.Normal(logits, 1.0)
                    logprob = dist.log_prob(action_t).sum()
                    ratio = torch.exp(logprob - old_logprob)

                    advantage = advantages[t]
                    clip_adv = torch.clamp(ratio, 0.8, 1.2) * advantage
                    actor_loss = -torch.min(ratio * advantage, clip_adv)
                    actor_loss.backward(retain_graph=True)
                optimizers[agent].step()

        # Save models
        if epoch % save_interval == 0:
            save_models(current_run_dir, epoch, actors, critic)

        # Log to TensorBoard
        writer.add_scalar("Return/Mean", np.sum(rewards), epoch)
        writer.add_scalar("Loss/Critic", critic_loss_total, epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Return={np.sum(rewards):.2f}")

    writer.close()

if __name__ == "__main__":
    train_mappo()
