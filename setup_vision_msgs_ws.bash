#!/bin/bash

# setup_vision_msgs_ws.bash

# Script per installare vision_msgs e RViz plugins da sorgenti e buildare il workspace ROS2



set -e  # esce se c'è un errore



# Workspace

VISION_WS="$HOME/ros2_vision_ws"

SRC_DIR="$VISION_WS/src"



echo "Creazione workspace $VISION_WS..."

mkdir -p "$SRC_DIR"

cd "$SRC_DIR"



# Clona vision_msgs

if [ ! -d vision_msgs ]; then

    echo "Clonazione vision_msgs..."

    git clone -b humble https://github.com/ros-perception/vision_msgs.git

else

    echo "vision_msgs già presente, faccio pull..."

    cd vision_msgs

    git pull

    cd ..

fi







# Installa numpy per il Python di sistema

echo "Installazione di numpy per /usr/bin/python3..."

sudo /usr/bin/python3 -m pip install numpy



# Pulisce eventuali build precedenti

echo "Pulizia workspace..."

cd "$VISION_WS"

rm -rf build/ install/ log/



# Build con colcon, forzando Python di sistema

echo "Build del workspace..."

source /opt/ros/humble/setup.bash

colcon build --symlink-install --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3



# Sorgi l'ambiente

echo "Sorgo l'ambiente..."

source install/setup.bash



echo "Workspace vision_msgs pronto!"

echo "Puoi ora lanciare i nodi ROS2 normalmente."
