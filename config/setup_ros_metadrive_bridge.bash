#!/bin/bash

# setup_ros_bridge.sh

# Simple script to set up ROS bridge for MetaDrive



# Directory of the ROS bridge

WORKDIR="$HOME/metadrive-ros-integration/metadrive/bridges/ros_bridge"

# Change directory

cd "$WORKDIR" || { echo "Directory not found!"; exit 1; }



# Activate ROS environment

echo "Sourcing ROS setup..."

source /opt/ros/${ROS_DISTRO}/setup.bash



# Initialize rosdep (only needed the first time)

echo "Initializing rosdep..."

sudo rosdep init 2>/dev/null || echo "rosdep already initialized"



# Install pyzmq using system Python

echo "Installing pyzmq..."

pip install --user pyzmq



# Update rosdep and install dependencies

echo "Updating rosdep..."

rosdep update

echo "Installing ROS dependencies..."

rosdep install --from-paths src --ignore-src -y



# Build the workspace

echo "Building workspace with colcon..."

colcon build



# Source the workspace

echo "Sourcing workspace..."

source install/setup.bash



echo "ROS bridge setup complete!"
