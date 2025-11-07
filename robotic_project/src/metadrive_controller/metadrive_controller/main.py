import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge
from rclpy.wait_for_message import wait_for_message

import cv2
import json
import os
import numpy as np
import gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO, DDPG, SAC, TD3

np.random.seed(0)

# ------------- READ CONFIG PARAMETERS --------------
with open("../config.json", "r") as f:
	cfg = json.load(f)

# Extract training params
train_cfg = cfg["training"]
env_cfg = cfg["environment"]
veh_cfg = cfg["vehicle"]
model_name = cfg["model"]
model_zip_file = cfg["model_zip_file"]

# Assign training variables
total_timesteps = train_cfg["total_timesteps"]
n_steps = train_cfg["n_steps"]
batch_size = train_cfg["batch_size"]
img_shape = train_cfg["img_shape"]
buffer_size = train_cfg["buffer_size"]
learning_starts = train_cfg["learning_starts"]


class RLmetadrive(Node):
	def __init__(self):
		super().__init__('metadrive_controller')

		self.bridge = CvBridge()

		# QoS Profiles
		qos_profile_best_effort = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1
		)
		qos_profile_reliable = QoSProfile(
			reliability=QoSReliabilityPolicy.RELIABLE,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=1
		)

		# Load trained model
		if model_name.upper() == "PPO":
			print("Loading PPO model...")
			self.model = PPO.load(os.path.join("../models_trained", model_zip_file), device="cuda")
		elif model_name.upper() == "DDPG":
			print("Loading DDPG model...")
			self.model = DDPG.load(os.path.join("../models_trained", model_zip_file), device="cuda")
		elif model_name.upper() == "TQC":
			print("Loading TQC model...")
			self.model = TQC.load(os.path.join("../models_trained", model_zip_file), device="cuda")
		elif model_name.upper() == "SAC":
			print("Loading SAC model...")
			self.model = SAC.load(os.path.join("../models_trained", model_zip_file), device="cuda")
		else:
			raise ValueError(f"Unsupported model type: {model_name}")

		self.img_shape = img_shape

		# Subscriptions
		self.camera_sub = self.create_subscription(
			Image, 'metadrive/image', self.camera_callback, qos_profile_best_effort
		)
		self.state_and_lidar_sub = self.create_subscription(
			Float32MultiArray, 'metadrive/state_and_lidar', self.state_and_lidar_callback, qos_profile_best_effort
		)

		# Publisher
		self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', qos_profile_reliable)

		self.camera_msg = None
		self.state_and_lidar = None

		# ------------- WAIT FOR FIRST MESSAGES --------------
		wait_for_message(Image, self, 'metadrive/image')
		wait_for_message(Float32MultiArray, self, 'metadrive/state_and_lidar')

		# Control loop timer
		self.dt = 0.1
		self.timer = self.create_timer(self.dt, self.control_loop)

	def camera_callback(self, msg):
		"""Convert ROS2 Image to numpy array compatible with the model."""
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
			if cv_image.shape[:2] != (84, 84):
				cv_image = cv2.resize(cv_image, (84, 84), interpolation=cv2.INTER_AREA)
			cv_image = cv_image.astype(np.float32) / 255.0
			cv_image = np.expand_dims(cv_image, axis=-1)
			self.camera_msg = cv_image
		except Exception as e:
			self.get_logger().error(f"Errore nella conversione dell'immagine: {e}", exc_info=True)

	def state_and_lidar_callback(self, msg):
		try:
			self.state_and_lidar = np.array(msg.data, dtype=np.float32)
		except Exception as e:
			self.get_logger().error(f"Errore nella state_and_lidar_callback: {e}", exc_info=True)

	def control_loop(self):
		print("\n------------------- Control Loop Started ------------------")

		# ------- GET OBS -------
		img = self.camera_msg
		state_and_lidar = self.state_and_lidar

		# ------- BUILD OBS -------
		obs = {
			"image": np.expand_dims(img, axis=0),  # add batch dim
			"state_and_lidar": np.expand_dims(state_and_lidar, axis=0)
		}

		# ------- PREDICT ACTION -------
		action, _ = self.model.predict(obs, deterministic=True)
		action = action[0]

		print("Predicted Action:", action)

		steering = action[0]
		throttle = action[1]

		print("Steering:", steering, "Throttle:", throttle)

		# ------- PUBLISH CMD_VEL -------
		twist = Twist()
		twist.linear.x = float(throttle)
		twist.angular.z = float(steering)
		self.publisher_vel.publish(twist)

		print("------------------- Control Loop Executed ------------------ \n")

	def cleanup(self):
		"""Send stop command and cleanup resources."""
		twist = Twist()
		twist.linear.x = 0.0
		twist.angular.z = 0.0
		self.publisher_vel.publish(twist)
		self.get_logger().info("Sent stop Twist before shutdown.")
		rclpy.spin_once(self, timeout_sec=0.2)
		self.publisher_vel.destroy()
		self.get_logger().info("Destroyed publisher to flush QoS queue.")
		self.destroy_timer(self.timer)


def main(args=None):
	rclpy.init(args=args)
	rl_meta = RLmetadrive()
	try:
		rclpy.spin(rl_meta)
	except KeyboardInterrupt:
		rl_meta.get_logger().info("Keyboard interrupt — shutting down controller...")
	finally:
		rl_meta.cleanup()
		rl_meta.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
