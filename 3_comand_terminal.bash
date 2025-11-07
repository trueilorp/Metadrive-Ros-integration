#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

ROS_BRIDGE_SETUP="$PROJECT_DIR/metadrive/bridges/ros_bridge/install/setup.bash"
ROBOTIC_PROJECT_DIR="$PROJECT_DIR/robotic_project"

if [ ! -f "$ROS_BRIDGE_SETUP" ]; then
    echo "Error: setup.bash for MetaDrive ROS bridge not found at:"
    echo "  $ROS_BRIDGE_SETUP"
    echo "Run 1_command_terminal.bash first to build the ROS bridge."
    exit 1
fi

if [ ! -d "$ROBOTIC_PROJECT_DIR" ]; then
    echo "Error: robotic_project directory not found at:"
    echo "  $ROBOTIC_PROJECT_DIR"
    exit 1
fi

echo "Sourcing MetaDrive ROS bridge environment..."
source "$ROS_BRIDGE_SETUP"

echo "Building robotic_project workspace..."
cd "$ROBOTIC_PROJECT_DIR"
colcon build --symlink-install

echo "Sourcing robotic_project environment..."
source install/setup.bash

echo "Launching metadrive_controller node..."
ros2 run metadrive_controller metadrive_controller