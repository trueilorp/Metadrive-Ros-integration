#!/bin/bash

cd /home/trueilorp/metadrive/bridges/ros_bridge
colcon build

source /home/trueilorp/metadrive/bridges/ros_bridge/install/setup.bash

cd /home/trueilorp/metadrive/bridges/ros_bridge

ros2 launch metadrive_example_bridge metadrive_example_bridge.launch.py
