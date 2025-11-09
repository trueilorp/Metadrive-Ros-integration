# AI & Robotics Project: MetaDrive and ROS Integration

This project integrates the vehicular simulation **MetaDrive** with the **ROS 2** (Robot Operating System 2) framework for managing and training a robotic controller.

## Main Directory Structure

The `~/metadrive-ros-integration` directory contains the following main files and folders:

| File/Folder Name | Description |
| :--- | :--- |
| `1_comand_terminal.bash` | Script for **Terminal 1** (Launching ROS Bridges). |
| `2_comand_terminal.bash` | Script for **Terminal 2** (Starting MetaDrive and Socket Server). |
| `3_comand_terminal.bash` | Script for **Terminal 3** (Starting the ROS Controller Node). |
| `eval_model.py` | Python file to start **model evaluation**. |
| `train_model.py` | Python file to start **model training**. |
| `config.json` | Config file to define model and env configuration. |
| `start_project_config.bash` | Script to initialize project dependencies. |

| `bridges` | Contains the ROS bridges. |
| `callback` | Contains the callback of training. |
| `config` | Contains the two script used by *start_project_config.bash*. |
| `env` | Contains the customize metadrive environment. |
| `models_trained` | Contains some models already trained. |
| `nets` | Contains the customize CNN for multi input policy. |
| `obs` | Contains the customize observations. |
| `robotic_project` | Contains the code for the ROS controller node. |

---
## Prerequisite: Install and Integrate MetaDrive

Before running this project, you need to clone and install **MetaDrive** from its official repository *https://github.com/metadriverse/metadrive*
and follow the MetaDrive’s official installation instructions.

Once installed, place the MetaDrive directory inside `metadrive-ros-integration`, then replace its existing `bridges` folder with the one provided in `bridges` from this repository.

## Before starting
First, run the **start_project_config.bash** file to initialize the working space.

## Model Training and Evaluation

The following scripts are used for training and evaluating the driving model:

* **Training:** Run `train_model.py` in the root directory.
* **Evaluation:** Run `eval_model.py` in the root directory.

### Models and Configuration

* **Pre-trained Models:** Some pre-trained models are available in the folder:
    * `/models_trained`
* **Configuration:** Parameters for training and configuring the MetaDrive environment can be modified in the file:
    * `config.json`
    
    Please make sure that the model loaded parameters match the parameters in the *config.json* file.
    

## Startup Instructions: The Three Terminals

The project is designed to be started on **three separate and distinct terminals** by executing the commands in order:

### 1. Terminal 1: ROS Bridges (Socket Client)
Starts the ROS nodes responsible for communication (via sockets) with the MetaDrive environment.

* **File Executed:** `1_comand_terminal.bash`
* **Main Action:** Launches the bridges from the launch file:
    * `/metadrive/bridges/ros_bridge/src/metadrive_example_bridge/launch/metadrive_example_bridge.launch.py`
* **Bridge Files:** The nodes include `camera_bridge.py`, `cmd_vel_bridge.py`, `lidar_bridge.py`, `obj_bridge.py`, and `state_and_lidar_bridge.py`.

### 2. Terminal 2: MetaDrive Environment (Socket Server)
Starts the MetaDrive simulation environment and the socket connection.

* **File Executed:** `2_comand_terminal.bash`
* **Main Action:** Generates the MetaDrive environment and opens the **socket server** connection with ROS.
* **Reference File:** `/metadrive/bridges/ros_bridge/ros_socket_server.py`

### 3. Terminal 3: ROS Controller Node
Starts the ROS control node which processes received data and sends commands to the vehicle.

* **File Executed:** `3_comand_terminal.bash`
* **Main Action:** Starts the ROS controller node.
* **Reference File:** `/robotic_project/src/metadrive_controller/metadrive_controller/main.py`

---