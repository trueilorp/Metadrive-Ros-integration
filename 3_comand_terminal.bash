source metadrive/bridges/ros_bridge/install/setup.bash

cd ~/robotic_project

colcon build

. install/setup.bash

ros2 run metadrive_controller metadrive_controller
