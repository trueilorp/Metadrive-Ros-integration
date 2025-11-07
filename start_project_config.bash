#!/bin/bash
set -e

# Setup vision_msgs workspace
echo "Starting vision_msgs workspace setup..."
./config/setup_vision_msgs_ws.bash
source ~/metadrive-ros-integration/ros2_vision_ws/install/setup.bash

# Setup ROS bridge MetaDrive
echo "Starting ROS bridge MetaDrive setup..."
./config/setup_ros_metadrive_bridge.bash
source ~/metadrive-ros-integration/metadrive/bridges/ros_bridge/install/setup.bash

echo "All setup completed!"
