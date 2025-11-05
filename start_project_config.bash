#!/bin/bash

# setup_all.bash

# Esegue setup_ros_metadrive_bridge.bash e setup_vision_msgs_ws.bash uno dopo l'altro



set -e  # esce se uno dei comandi fallisce



echo "Eseguo setup di vision_msgs workspace..."

./setup_vision_msgs_ws.bash

source ~/ros2_vision_ws/install/setup.bash





echo "Eseguo setup del ROS bridge MetaDrive..."

./setup_ros_metadrive_bridge.bash

source ~/metadrive/bridges/ros_bridge/install/setup.bash



cd ~/metadrive/bridges/ros_bridge



# Lancio il ROS bridge

echo "Lancio il ROS bridge MetaDrive..."



echo "Tutti i setup completati correttamente!"
