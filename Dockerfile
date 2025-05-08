FROM osrf/ros:humble-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/humble/lib

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-controller-manager \
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    tmux \
    git \
    libopencv-dev \
    python3-opencv \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    && apt-get clean

# Create workspace directory
WORKDIR /root/ros2_ws/src
RUN git clone https://github.com/BenkoDaniel/ros2_rm.git robomaster_ros

# Install Python dependencies with version pinning
WORKDIR /root/ros2_ws
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80 \
    stable-baselines3 \
    pettingzoo \
    gym \
    torch \
    matplotlib \
    tensorboard

# Build the workspace
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# Fix permissions and create necessary directories
RUN mkdir -p /root/.gazebo && \
    chmod -R 777 /root/.gazebo && \
    mkdir -p /root/.ros && \
    chmod -R 777 /root/.ros

# Setup environment
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> ~/.bashrc

# Corrected CMD with proper paths
CMD bash -c "\
    source /opt/ros/humble/setup.bash && \
    source /root/ros2_ws/install/setup.bash && \
    export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/humble/lib && \
    export GAZEBO_MODEL_PATH=/root/ros2_ws/src/robomaster_ros/robomaster_description/models && \
    ros2 launch robomaster_ros simulation.launch use_gui:=false & \
    sleep 30 && \
    python3 /root/ros2_ws/src/robomaster_ros/robomaster_ros/robomaster_ros/ippo_trainer.py"