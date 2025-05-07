FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/ros2_rm

RUN apt-get update && apt-get install -y \
    gazebo11 \
    ros-humble-gazebo-ros-pkgs \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p $ROS_WS/src

WORKDIR $ROS_WS/src
RUN git clone https://github.com/BenkoDaniel/ros2_rm.git


COPY requirements.txt $ROS_WS/src/robomaster_ros/
RUN pip3 install -r requirements.txt

WORKDIR $ROS_WS
RUN . /opt/ros/humble/setup.sh && \
    colcon build

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source $ROS_WS/install/setup.bash" >> ~/.bashrc

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source $ROS_WS/install/setup.bash && ros2 launch robomaster_ros simulation.launch.py"]
CMD ["bash", "-c", "python3 $ROS_WS/src/robomaster_ros/robomaster_ros/robomaster_ros/ippo_train.py"]

