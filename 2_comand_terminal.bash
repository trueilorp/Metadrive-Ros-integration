#!/bin/bash
set -e  

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
ROS_BRIDGE_DIR=""

if [ -d "$PROJECT_DIR/metadrive/bridges/ros_bridge" ]; then
    ROS_BRIDGE_DIR="$PROJECT_DIR/metadrive/bridges/ros_bridge"
elif [ -d "$PROJECT_DIR/bridges/ros_bridge" ]; then
    ROS_BRIDGE_DIR="$PROJECT_DIR/bridges/ros_bridge"
else
    echo "Error: 'ros_bridge' directory not found."
    echo "Expected at: metadrive/bridges/ros_bridge or bridges/ros_bridge"
    exit 1
fi

echo "ROS bridge directory found at: $ROS_BRIDGE_DIR"
cd "$ROS_BRIDGE_DIR"

if [ ! -f "ros_socket_server.py" ]; then
    echo "Error: ros_socket_server.py not found in $ROS_BRIDGE_DIR"
    exit 1
fi

echo "Launching ros_socket_server.py ..."
python3 ros_socket_server.py
