#!/bin/bash

# setup_all.bash

# Run setup_ros_metadrive_bridge.bash and setup_vision_msgs_ws.bash



set -e 



echo "Starting vision_msgs workspace setup..."

./config/setup_vision_msgs_ws.bash

source ~/ros2_vision_ws/install/setup.bash





echo "Starting ROS bridge MetaDrive setup..."

./config/setup_ros_metadrive_bridge.bash

source ~/metadrive-ros-integration/metadrive/bridges/ros_bridge/install/setup.bash



cd ~/metadrive-ros-integration/metadrive/bridges/ros_bridge



# Launch ROS bridge

echo "Launch ROS bridge MetaDrive..."


echo "All setup completed!"
