#!/bin/bash

# setup_vision_msgs_ws.bash

# Script to install vision_msgs and RViz plugins and build ROS2 workspace



set -e  



# Workspace

VISION_WS="$HOME/metadrive-ros-integration/ros2_vision_ws"

SRC_DIR="$VISION_WS/src"



echo "Workspace creation $VISION_WS..."

mkdir -p "$SRC_DIR"

cd "$SRC_DIR"



if [ ! -d vision_msgs ]; then

    echo "Cloning vision_msgs..."

    git clone -b humble https://github.com/ros-perception/vision_msgs.git

else

    echo "vision_msgs already present, pull..."

    cd vision_msgs

    git pull

    cd ..

fi


echo "Installation of numpy for /usr/bin/python3..."

sudo /usr/bin/python3 -m pip install numpy



echo "Clean workspace..."

cd "$VISION_WS"

rm -rf build/ install/ log/



echo "Build workspace..."

source /opt/ros/humble/setup.bash

colcon build --symlink-install --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3


echo "Source env..."

source install/setup.bash



echo "Workspace vision_msgs ready!"