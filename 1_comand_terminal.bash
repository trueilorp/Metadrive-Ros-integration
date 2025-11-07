#!/bin/bash

PROJECT_DIR=~/metadrive-ros-integration
ROS_BRIDGE_DIR="$PROJECT_DIR/metadrive/bridges/ros_bridge"

cd "$ROS_BRIDGE_DIR" || { echo "ROS bridge directory not found"; exit 1; }

colcon build
source install/setup.bash

ros2 launch metadrive_example_bridge metadrive_example_bridge.launch.py
