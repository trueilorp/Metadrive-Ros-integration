#!/bin/bash

cd metadrive/bridges/ros_bridge
colcon build

source metadrive/bridges/ros_bridge/install/setup.bash

cd metadrive/bridges/ros_bridge

ros2 launch metadrive_example_bridge metadrive_example_bridge.launch.py
