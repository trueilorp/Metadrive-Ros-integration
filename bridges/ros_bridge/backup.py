# # -------------- WRAPPERS AND OBSERVATION PREPROCESSING --------------
# class ResizeImageAndProcessDict(gym.ObservationWrapper):
# 	# definisco il nuovo spazio delle osservazioni
# 	def __init__(self, env, img_shape, n_objects):
# 		# old observation state --> Dict('image': Box(-0.0, 1.0, (900, 1200, 3, 3), float32), 'state': Box(-0.0, 1.0, (80,), float32))
# 		super().__init__(env)
# 		self.img_shape = (img_shape, img_shape)
# 		self.n_objects = n_objects

# 		old_space = env.observation_space
# 		new_spaces = {}

# 		for key, space in old_space.spaces.items():
# 			if key == "image":
# 				# SB3 vuole channel come ultima dimensione
# 				new_spaces[key] = spaces.Box( # Box perchè observation è composta di Box spaces inizialmente
# 					low=0, high=255, shape=(self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8
# 				)
# 			elif key == "state":
# 				new_spaces[key] = spaces.Box(
# 					low=-np.inf, high=np.inf, shape=(self.n_objects,), dtype=np.float32
# 				)
# 			else:
# 				return NotImplementedError(f"Unknown observation key: {key}")

# 		self.observation_space = spaces.Dict(new_spaces) # creo la nuova observation
	
# 	# ogni volta trasformo l'obs per farla combaciare con lo spazio definito sopra da __init__
# 	def observation(self, obs): # questa funzione viene chiamata ad ogni step e reset per processare l'osservazione
# 		# ---------------- Camera ----------------
# 		img = obs.get("image")
# 		if img is not None:
# 			# # Sometimes MetaDrive returns (H,W,3,3)
# 			# if img.ndim == 4 and img.shape[-1] == 3:
# 			img = img[..., 0] # prendo solo la prima camera, perchè a volte img contiene 3 immagini da 3 diverse camere
# 			img = cv2.resize(img, self.img_shape, interpolation=cv2.INTER_AREA)
# 			obs["image"] = img.astype(np.uint8)  # Keep channel-last

# 		# ---------------- Objects ----------------
# 		objects = obs.get("state", None)
# 		if objects is None or len(objects) == 0:
# 			obs["state"] = np.zeros(self.n_objects, dtype=np.float32)
# 		else:
# 			# Flatten or pad/truncate to fixed size
# 			obj_array = np.array(objects, dtype=np.float32).flatten()
# 			if obj_array.size < self.n_objects:
# 				padded = np.zeros(self.n_objects, dtype=np.float32)
# 				padded[:obj_array.size] = obj_array
# 				obs["state"] = padded
# 			else:
# 				obs["state"] = obj_array[:self.n_objects]
# 		return obs


import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from cv_bridge import CvBridge
import cv2
import json

import numpy as np
from rclpy.wait_for_message import wait_for_message
import ros2_numpy
from std_msgs.msg import Float32MultiArray

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC

np.random.seed(0)

# ------------- READ CONFIG PARAMETERS --------------
# --- Load config file ---
with open("/home/trueilorp/metadrive/bridges/ros_bridge/config.json", "r") as f:
	cfg = json.load(f)

# Extract training params
train_cfg = cfg["training"]
env_cfg = cfg["environment"]
veh_cfg = cfg["vehicle"]
model_name = cfg["model"]

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
		
		self.model = PPO.load("/home/trueilorp/metadrive/bridges/ros_bridge/models/ppo_metadrive_multimodal.zip")
		# self.model = DDPG.load("/home/trueilorp/metadrive/bridges/ros_bridge/models/ddpg_metadrive_multimodal.zip")
		# self.model = SAC.load("/home/trueilorp/metadrive/bridges/ros_bridge/models/sac_metadrive_multimodal.zip")
		# print(self.model.observation_space)
		
		self.img_shape = img_shape
		self.object_shape = self.model.observation_space["state_and_lidar"].shape[0]
		
		self.camera_sub = self.create_subscription(Image, 'metadrive/image', self.camera_callback, qos_profile_best_effort)
		self.state_and_lidar_sub = self.create_subscription(Float32MultiArray, 'metadrive/state_and_lidar', self.state_and_lidar_callback, qos_profile_best_effort)
		# self.object_sub = self.create_subscription(BoundingBox3DArray, 'metadrive/object', self.object_callback, qos_profile_best_effort)
		# self.lidar_sub = self.create_subscription(PointCloud2, 'metadrive/lidar', self.lidar_callback, qos_profile_best_effort)
		
		self.camera_msg = None
		# self.object_msg = None
		# self.lidar_msg = None
		self.state_and_lidar = None
		
		self.publisher_vel = self.create_publisher(Twist, '/cmd_vel', qos_profile_reliable)

		wait_for_message(Image, self, 'metadrive/image')
		# wait_for_message(BoundingBox3DArray, self, 'metadrive/object')
		# wait_for_message(PointCloud2, self, 'metadrive/lidar')
		wait_for_message(Float32MultiArray, self, 'metadrive/state_and_lidar')

		self.dt = 0.1
		self.timer = self.create_timer(self.dt, self.control_loop)

	def camera_callback(self, msg):
		"""Callback per convertire il messaggio ROS Image in un frame numpy compatibile col modello."""
		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') # converto il messaggio di ROS2 in img opencv
			cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) # converto in RGB
			if cv_image.shape[:2] != (84, 84): # ridimensiono l'immagine a (84,84) se non lo è già
				cv_image = cv2.resize(cv_image, (84, 84), interpolation=cv2.INTER_AREA)
			cv_image = cv_image.astype(np.float32) / 255.0 # normalizzo tra 0 e 1
			cv_image = np.expand_dims(cv_image, axis=-1) # expando perchè nel training trainavo img con shape (84,84,3,1)
			# cv_image = np.transpose(cv_image, (3, 2, 0, 1)) # riordino in (84,84,3,1)
			self.camera_msg = cv_image
		except Exception as e:
			self.get_logger().error(f"Errore nella conversione dell'immagine: {e}", exc_info=True)

	def object_callback(self, msg):
		object_list = msg.boxes
		all_detected_objects = [] # lista vuota per contenere i dati di tutti gli oggetti
		for box in object_list: # per ogni oggetto nella lista (pedoni, ...)
			center_pos = box.center.position # estrapolo la pos x,y,z (con z=0.0)
			center_orient = box.center.orientation # estrapolo la rot (quaternion4D) con x=0.0 e y=0.0
			box_size = box.size # estrapolo la size (lunghezza, larghezza, altezza) dell'oggetto
			all_detected_objects.append({
				"position": center_pos,
				"orientation": center_orient,
				"size": box_size
			}) # aggiungo il dizionario alla lista
		self.object_msg = all_detected_objects
		#self.get_logger().info(f'OBJECT --> Ricevuti e processati {len(self.object_msg)} oggetti.')
	
	def lidar_callback(self, msg: PointCloud2):
		"""
		Callback finale e funzionante per il LiDAR di MetaDrive.
		Estrae (x, y, z) da 'xyz' e ricostruisce il vettore di distanze normalizzate.
		"""
		try:
			# Converti il messaggio PointCloud2 in un dizionario NumPy
			data_dict = ros2_numpy.numpify(msg)

			# Estrai i punti 3D (N, 3)
			points = data_dict['xyz']  # già array np.ndarray con (x, y, z)
			if points.size == 0:
				self.get_logger().warn("LIDAR: messaggio vuoto ricevuto.")
				return

			# Calcola le distanze in metri (solo piano XY)
			distances = np.linalg.norm(points[:, :2], axis=1)  # sqrt(x^2 + y^2)

			# Normalizza le distanze (0–1)
			lidar_max_distance = getattr(self, "lidar_max_distance", 50.0)  # default 50 m
			normalized = np.clip(distances / lidar_max_distance, 0.0, 1.0)

			# Troncamento o padding per rispettare num_lasers
			num_lasers = getattr(self, "num_lasers", 240)
			if len(normalized) > num_lasers:
				normalized = normalized[:num_lasers]
			elif len(normalized) < num_lasers:
				normalized = np.pad(normalized, (0, num_lasers - len(normalized)), 'constant')

			# Salva il vettore lidar (float32) per uso nel main RL
			self.lidar_state = normalized.astype(np.float32)

			# Log (opzionale)
			self.get_logger().info(f"LIDAR OK: vettore {self.lidar_state.shape}, "
								   f"esempio={self.lidar_state[:5]}")

		except Exception as e:
			self.get_logger().error(f"Errore nella lidar_callback: {e}", exc_info=True)

	def state_and_lidar_callback(self, msg):
		try:
			self.state_and_lidar = np.array(msg.data, dtype=np.float32)
		except Exception as e:
			self.get_logger().error(f"Errore nella state_and_lidar_callback: {e}", exc_info=True)
			
	def control_loop(self):
		
		print("\n ------------------- Control Loop Started ------------------")
		
		# Get messages from metadrive
		# print("Camera MSG", self.camera_msg)
		# print("Object MSG", self.object_msg)
		# print("Lidar MSG", self.lidar_msg)
		
		# preprocess object
		# if len(self.object_msg) == 0:
		# 		object_msg = np.zeros((self.object_shape,), dtype=np.float32)
		# else:
		# 	object_msg = self.map_objects_to_state(self.object_msg)
		
		# # preprocess image, resize to the training shape (84,84)
		# if len(self.camera_msg) != 0:
		# 	img = self.camera_msg  # assuming it's a numpy array already (900x1200x3)
		# 	# img_resized = cv2.resize(img, (self.img_shape, self.img_shape), interpolation=cv2.INTER_AREA) # resize to match training dim
			
		# 	# print("Length Object MSG:", len(self.object_msg))
		# 	# Detected objects: [{'position': geometry_msgs.msg.Point(x=52.41254425048828, y=-4.6554059982299805, z=0.0), 'orientation': geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.021263538997177787, w=0.9997739053952726), 'size': geometry_msgs.msg.Vector3(x=4.869999885559082, y=2.0460000038146973, z=1.850000023841858)}]
		img = self.camera_msg
		state_and_lidar = self.state_and_lidar
		
		# ------- BUILD OBS -------
		obs = {
			"image": img,
			"state_and_lidar": state_and_lidar
		}
		print("Observation:", obs)
		
		# ------- PREDICT ACTION -------
		action, _ = self.model.predict(obs, deterministic=True)
		# action = action[0] # sto passando un batch di osservazioni al modello
		print("Predicted Action:", action) # Predicted Action: [-0.05352883  0.16370402]
		steering = action[0]
		throttle = action[1]
		
		print("Steering:", steering, "Throttle:", throttle)
		
		# ------- PUBLISH CMD_VEL -------
		twist = Twist()
		twist.linear.x = float(throttle)
		twist.angular.z = float(steering)
		self.publisher_vel.publish(twist)
		# print("Published Twist:", twist)
		
		print("------------------- Control Loop Executed ------------------ \n")

	def map_objects_to_state(self, object_list):
		max_objects = 3
		state = []
		
		# print("Object_list:", object_list)
		for i in range(max_objects):
			if i < len(object_list):
				obj = object_list[i]
				pos = obj['position']
				size = obj['size']

				# Aggiungi x,y,z della posizione e x,y,z delle dimensioni
				state.extend([pos.x, pos.y, pos.z, size.x, size.y, size.z])
			else:
				# Se ci sono meno oggetti, riempi con zeri
				state.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

		# PPO era stato allenato con 19 elementi → aggiungi un valore finale (ad esempio 0)
		if len(state) < self.object_shape:
			state.append(0.0)

		# Converti in numpy array float32
		return np.array(state[:self.object_shape], dtype=np.float32)
	
	def cleanup(self):
		# Stop signal
		twist = Twist()
		twist.linear.x = 0.0
		twist.angular.z = 0.0
		self.publisher_vel.publish(twist)
		self.get_logger().info("Sent stop Twist before shutdown.")

		# Force ROS executor to send out messages
		rclpy.spin_once(self, timeout_sec=0.2)

		# Destroy publisher completely
		self.publisher_vel.destroy()
		self.get_logger().info("Destroyed publisher to flush QoS queue.")

		# Kill timers, cleanup
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
	
	
	# def run(self, test=False):
	# 	# ------------- VEHICLE CONFIG --------------
	# 	vehicle_config = {
	# 			"show_navi_mark": True,
	# 			"enable_reverse": True,
	# 			"show_lidar": True,
	# 			"image_source":"rgb_camera",

	# 			# "side_detector": {
	# 			# 	"num_lasers": 32,
	# 			# 	"distance": 50
	# 			# },
	# 			# "show_side_detector": True,

	# 			# "lane_line_detector": {
	# 			# 	"num_lasers": 32,
	# 			# 	"distance": 50
	# 			# },
	# 			# "show_lane_line_detector": True,

	# 			"lidar": {
	# 				"num_lasers": veh_cfg["num_lasers"],
	# 				"distance": veh_cfg["distance"],
	# 				"num_others": veh_cfg["num_others"],
	# 				"add_others_navi": veh_cfg["add_others_navi"],
	# 				"gaussian_noise": 0.0,
	# 				"dropout_prob": 0.0
	# 			}
	# 		}
			
	# 	# ------------- ENV CONFIG --------------
	# 	config = (dict(map="C", # map="SCSTRS",
	# 					use_render=True, 
	# 					image_observation=True,
	# 					agent_observation=CustomizeObs,
						
	# 					manual_control=False,
	# 					random_spawn_lane_index=False,
	# 					num_scenarios=env_cfg["num_scenarios"],
	# 					traffic_density=env_cfg["traffic_density"],
	# 					accident_prob=0,
	# 					log_level=50,
						
	# 					out_of_road_penalty=env_cfg["out_of_road_penalty"], # di default è 10
	# 					crash_sidewalk_penalty=env_cfg["crash_sidewalk_penalty"], # di default è 0
				
	# 					vehicle_config = vehicle_config,
	# 					sensors={"rgb_camera": (RGBCamera, img_shape, img_shape)},
	# 					stack_size=1 # tengo solo l'ultimo frame della camera, perchè memorizza più frame in stack
	# 				))

		# env = MetaDriveEnv(config)
		env = MetaDriveEnvGymnasium(config)
		
		# try:
		env.reset()
		print(HELP_MESSAGE)
		# env.agent.expert_takeover = False
		
		# crea socket per ricevere comandi ######### NUOVO
		cmd_socket = self.context.socket(zmq.PULL)
		cmd_socket.bind("ipc:///tmp/cmd_vel")  # ROS2 pubblicherà qui
		cmd_socket.set_hwm(5)
		
		reward = 0.0
	
		reward = 0.0
			while True:
				try:
					ricevi comandi come due float concatenati
					msg = cmd_socket.recv(flags=zmq.NOBLOCK)
					steering, throttle  = struct.unpack('ff', msg)
					action = [steering, throttle]
				except zmq.Again:
					nessun comando disponibile, mantieni ultima azione
					action = [0.0, 0.0]
					
				############################################
				print("ACTION RECEIVED:", action)
				o = env.step(action)
				
				reward += o[1]
				print("OBSERVATION:", o) # mi assicuro che env.step() restituisca qualcosa di compatibile con stable_baselines3, ovvero obs, rew, done, truncated, info
				
				env.render()
				if test:
					image_data = np.zeros((512, 512, 3))  # fake data for testing
					image_data[::16, :, :] = 255
				else:
				
				#################
				SEND IMAGE DATA
				#################
				print("Image shape to send",  o[0]['image'].shape) # Image shape to send (84, 84, 3, 1)
				image_data = o[0]['image'][..., -1] # prende solo l'ultimo frame, quindi shape = (84,84,3)
				print("Image shape send", image_data.shape) # Image shape send (84, 84, 3)
				send via socket
				image_data = image_data.astype(np.uint8) # COMMENTATO IO
				Scala da [0,1] a [0,255] e cast a uint8
				image_data = (image_data * 255).astype(np.uint8) # AGGIUNTO IO
				dim_data = struct.pack('ii', image_data.shape[1], image_data.shape[0])
				image_data = bytearray(image_data)
				concatenate the dimensions and image data into a single byte array
				image_data = dim_data + image_data
				try:
					self.img_socket.send(image_data, zmq.NOBLOCK)
				except zmq.error.Again:
					msg = "ros_socket_server: error sending image"
					if test:
						raise ValueError(msg)
					else:
						print(msg)
				del image_data  # explicit delete to free memory
				
				#################
				SEND OBJECT DATA
				#################
				lidar_data, objs = env.agent.lidar.perceive(
					env.agent,
					env.engine.physics_world.dynamic_world,
					env.agent.config["lidar"]["num_lasers"], # "num_lasers": 100
					env.agent.config["lidar"]["distance"],   # "distance": 200
					height=1.0,
				)

				ego_x = env.agent.position[0]
				ego_y = env.agent.position[1]
				ego_theta = np.arctan2(env.agent.heading[1], env.agent.heading[0])

				num_data = struct.pack('i', len(objs))
				obj_data = []
				for obj in objs: # scorro dentro tutti gli oggetti rilevati dal radar
					obj_x = obj.position[0]
					obj_y = obj.position[1]
					obj_theta = np.arctan2(obj.heading[1], obj.heading[0])

					obj_x = obj_x - ego_x
					obj_y = obj_y - ego_y
					obj_x_new = np.cos(-ego_theta) * obj_x - np.sin(-ego_theta) * obj_y
					obj_y_new = np.sin(-ego_theta) * obj_x + np.cos(-ego_theta) * obj_y

					obj_data.append(obj_x_new)
					obj_data.append(obj_y_new)
					obj_data.append(obj_theta - ego_theta)
					obj_data.append(obj.LENGTH)
					obj_data.append(obj.WIDTH)
					obj_data.append(obj.HEIGHT)
				obj_data = np.array(obj_data, dtype=np.float32)
				obj_data = bytearray(obj_data)
				obj_data = num_data + obj_data
				try:
					self.obj_socket.send(obj_data, zmq.NOBLOCK)
				except zmq.error.Again:
					msg = "ros_socket_server: error sending objs"
					if test:
						raise ValueError(msg)
					else:
						print(msg)
				del obj_data  # explicit delete to free memory

				##################
				SEND LIDAR DATA
				##################
				lidar_data = np.array(lidar_data) * env.agent.config["lidar"]["distance"]
				lidar_range = env.agent.lidar._get_lidar_range(
					env.agent.config["lidar"]["num_lasers"], env.agent.lidar.start_phase_offset
				)
				point_x = lidar_data * np.cos(lidar_range)
				point_y = lidar_data * np.sin(lidar_range)
				point_z = np.ones(lidar_data.shape)  # assume height = 1.0
				lidar_data = np.stack([point_x, point_y, point_z], axis=-1).astype(np.float32)
				dim_data = struct.pack('i', len(lidar_data))
				lidar_data = bytearray(lidar_data)
				concatenate the dimensions and lidar data into a single byte array
				lidar_data = dim_data + lidar_data
				try:
					self.lidar_socket.send(lidar_data, zmq.NOBLOCK)
				except zmq.error.Again:
					msg = "ros_socket_server: error sending lidar"
					if test:
						raise ValueError(msg)
					else:
						print(msg)
				del lidar_data  # explicit delete to free memory
				
				#################
				SEND STATE AND LIDAR 
				#################
				state_and_lidar = o[0]['state_and_lidar'] # perchè è un dizionario con un elemento
				try:
					flat_data = state_and_lidar.flatten().astype(np.float32)
					dim = len(flat_data)

					Prepara il pacchetto: [dim (4 bytes)] + [data (float32 * dim)]
					dim_bytes = struct.pack('i', dim)
					payload = dim_bytes + flat_data.tobytes()

					Invia in modalità non bloccante
					self.state_and_lidar_socket.send(payload, zmq.NOBLOCK)

				except zmq.error.Again:
					msg = "ros_socket_server: error sending state and lidar"
					if test:
						raise ValueError(msg)
					else:
						print(msg)
				del state_and_lidar
						
				#################
				CHECK DONE
				#################
				if o[2]:  # done
					break

		except Exception as e:
			stop_action = struct.pack('ff', 0.0, 0.0)
			self.context.socket(zmq.PUSH).connect("ipc:///tmp/cmd_vel").send(stop_action)
			cmd_socket.close()
			env.close()
			
			print("\n######################")
			print("##### FINAL REWARD:" + str(reward) + " #####")
			print("######################\n")
		finally:
			stop_action = struct.pack('ff', 0.0, 0.0)
			self.context.socket(zmq.PUSH).connect("ipc:///tmp/cmd_vel").send(stop_action)
			cmd_socket.close()
			env.close()
			
			print("\n######################")
			print("##### FINAL REWARD:" + str(reward) + " #####")
			print("######################\n")
			
			
# launch sockets to send sensor readings to ROS
import argparse
import struct
from functools import partial
import numpy as np
import zmq
import json
import gymnasium as gym
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

import time
import threading
import os

from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.lange_change_policy import LaneChangePolicy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif
from IPython.display import Image

# ------------- READ CONFIG PARAMETERS --------------
# --- Load config file ---
with open("config.json", "r") as f:
	cfg = json.load(f)

# Extract training params
train_cfg = cfg["training"]
env_cfg = cfg["environment"]
veh_cfg = cfg["vehicle"]
model_name = cfg["model"]

# Assign training variables
total_timesteps = train_cfg["total_timesteps"]
n_steps = train_cfg["n_steps"]
batch_size = train_cfg["batch_size"]
img_shape = train_cfg["img_shape"]
buffer_size = train_cfg["buffer_size"]
learning_starts = train_cfg["learning_starts"]

# ------------- CUSTOMIZE OBSERVATION SPACE --------------
class CustomizeObs(BaseObservation):
	"""
	Use camera, ego state info, navigation info and lidar as input
	"""
	IMAGE = "image"
	LIDAR = "state_and_lidar"

	def __init__(self, config):
		super(CustomizeObs, self).__init__(config)
		self.img_obs = ImageObservation(config, config["vehicle_config"]["image_source"], config["norm_pixel"])
		self.lidar_obs = LidarStateObservation(config)

	@property
	def observation_space(self):
		return gym.spaces.Dict({
			self.IMAGE: self.img_obs.observation_space,
			self.LIDAR: self.lidar_obs.observation_space
		})

	def observe(self, vehicle: BaseVehicle):
		return {
			self.IMAGE: self.img_obs.observe(),
			self.LIDAR: self.lidar_obs.observe(vehicle)
		}

	def destroy(self):
		super(CustomizeObs, self).destroy()
		self.img_obs.destroy()
		self.lidar_obs.destroy()


# ------------- VEHICLE CONFIG --------------
vehicle_config = {
	"show_navi_mark": True,
	"enable_reverse": True,
	"show_lidar": True,
	"image_source": "rgb_camera",
	"lidar": {
		"num_lasers": veh_cfg["num_lasers"],
		"distance": veh_cfg["distance"],
		"num_others": veh_cfg["num_others"],
		"add_others_navi": veh_cfg["add_others_navi"],
		"gaussian_noise": 0.0,
		"dropout_prob": 0.0
	}
}

# ------------- ENV CONFIG --------------
config = dict(
	map="C",
	use_render=True,
	image_observation=True,
	agent_observation=CustomizeObs,
	manual_control=False,
	random_spawn_lane_index=False,
	num_scenarios=env_cfg["num_scenarios"],
	traffic_density=0.2,
	accident_prob=0,
	log_level=50,
	out_of_road_penalty=env_cfg["out_of_road_penalty"],
	crash_sidewalk_penalty=env_cfg["crash_sidewalk_penalty"],
	vehicle_config=vehicle_config,
	sensors={"rgb_camera": (RGBCamera, img_shape, img_shape)},
	stack_size=1
)

# ----------- INIT ENV --------------
def create_env():
	env_base = MetaDriveEnvGymnasium(config)
	return env_base


# ------------- ENV CLASS FOR GYMNASIUM COMPATIBILITY --------------
class MetaDriveEnvGymnasium(MetaDriveEnv):
	def reset(self, *, seed=None, options=None):
		obs, info = super().reset(seed=seed)
		return obs, info

	def reward_function(self, vehicle_id: str):
		vehicle = self.agents[vehicle_id]
		step_info = dict()

		if vehicle.lane in vehicle.navigation.current_ref_lanes:
			current_lane = vehicle.lane
			positive_road = 1
		else:
			current_lane = vehicle.navigation.current_ref_lanes[0]
			current_road = vehicle.navigation.current_road
			positive_road = 1 if not current_road.is_negative_road() else -1

		long_last, _ = current_lane.local_coordinates(vehicle.last_position)
		long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

		if self.config["use_lateral_reward"]:
			lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
		else:
			lateral_factor = 1.0

		reward = 0.0
		reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
		reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

		step_info["step_reward"] = reward

		if self._is_arrive_destination(vehicle):
			reward = +self.config["success_reward"]
		elif self._is_out_of_road(vehicle):
			reward = -self.config["out_of_road_penalty"]
		elif vehicle.crash_vehicle:
			reward = -self.config["crash_vehicle_penalty"]
		elif vehicle.crash_object:
			reward = -self.config["crash_object_penalty"]
		elif vehicle.crash_sidewalk:
			reward = -self.config["crash_sidewalk_penalty"]

		step_info["route_completion"] = vehicle.navigation.route_completion
		return reward, step_info


# ------------- ROS SOCKET SERVER --------------
class RosSocketServer():
	def __init__(self):
		self.context = zmq.Context().instance()
		self.context.setsockopt(zmq.IO_THREADS, 2)

		self.img_socket = self.context.socket(zmq.PUSH)
		self.img_socket.setsockopt(zmq.SNDBUF, 4194304)
		self.img_socket.bind("ipc:///tmp/rgb_camera")
		self.img_socket.set_hwm(5)

		self.lidar_socket = self.context.socket(zmq.PUSH)
		self.lidar_socket.setsockopt(zmq.SNDBUF, 4194304)
		self.lidar_socket.bind("ipc:///tmp/lidar")
		self.lidar_socket.set_hwm(5)

		self.obj_socket = self.context.socket(zmq.PUSH)
		self.obj_socket.setsockopt(zmq.SNDBUF, 4194304)
		self.obj_socket.bind("ipc:///tmp/obj")
		self.obj_socket.set_hwm(5)

		self.state_and_lidar_socket = self.context.socket(zmq.PUSH)
		self.state_and_lidar_socket.setsockopt(zmq.SNDBUF, 4194304)
		self.state_and_lidar_socket.bind("ipc:///tmp/state_and_lidar")
		self.state_and_lidar_socket.set_hwm(5)

		self.current_action = [0.0, 0.0]
		self.running = True

	def run(self, test=False):
		env = DummyVecEnv([partial(create_env)])
		env.training = False
		env.reset()
		print(HELP_MESSAGE)

		ipc_path = "/tmp/cmd_vel"
		if os.path.exists(ipc_path):
			print(f"Removing state IPC file at {ipc_path}")
			os.remove(ipc_path)

		cmd_socket = self.context.socket(zmq.PULL)
		cmd_socket.bind("ipc:///tmp/cmd_vel")
		cmd_socket.set_hwm(5)

		reward = 0.0
		while True:
			try:
				msg = cmd_socket.recv(flags=zmq.NOBLOCK)
				steering, throttle = struct.unpack('ff', msg)
				action = [steering, throttle]
			except zmq.Again:
				action = [0.0, 0.0]

			print("ACTION RECEIVED:", action)
			o = env.step([action])
			reward += o[1]
			env.render()

			image_data = o[0]['image'][..., -1]
			image_data = (image_data * 255).astype(np.uint8)
			dim_data = struct.pack('ii', image_data.shape[1], image_data.shape[0])
			image_data = bytearray(image_data)
			image_data = dim_data + image_data
			try:
				self.img_socket.send(image_data, zmq.NOBLOCK)
			except zmq.error.Again:
				msg = "ros_socket_server: error sending image"
				if test:
					raise ValueError(msg)
				else:
					print(msg)
			del image_data

			# lidar_data, objs = env.agent.lidar.perceive(
			# 	env.agent,
			# 	env.engine.physics_world.dynamic_world,
			# 	env.agent.config["lidar"]["num_lasers"],
			# 	env.agent.config["lidar"]["distance"],
			# 	height=1.0,
			# )

			# ego_x = env.agent.position[0]
			# ego_y = env.agent.position[1]
			# ego_theta = np.arctan2(env.agent.heading[1], env.agent.heading[0])

			# num_data = struct.pack('i', len(objs))
			# obj_data = []
			# for obj in objs:
			# 	obj_x = obj.position[0]
			# 	obj_y = obj.position[1]
			# 	obj_theta = np.arctan2(obj.heading[1], obj.heading[0])

			# 	obj_x = obj_x - ego_x
			# 	obj_y = obj_y - ego_y
			# 	obj_x_new = np.cos(-ego_theta) * obj_x - np.sin(-ego_theta) * obj_y
			# 	obj_y_new = np.sin(-ego_theta) * obj_x + np.cos(-ego_theta) * obj_y

			# 	obj_data.extend([obj_x_new, obj_y_new, obj_theta - ego_theta, obj.LENGTH, obj.WIDTH, obj.HEIGHT])

			# obj_data = np.array(obj_data, dtype=np.float32)
			# obj_data = bytearray(obj_data)
			# obj_data = num_data + obj_data
			# try:
			# 	self.obj_socket.send(obj_data, zmq.NOBLOCK)
			# except zmq.error.Again:
			# 	msg = "ros_socket_server: error sending objs"
			# 	if test:
			# 		raise ValueError(msg)
			# 	else:
			# 		print(msg)
			# del obj_data

			# lidar_data = np.array(lidar_data) * env.agent.config["lidar"]["distance"]
			# lidar_range = env.agent.lidar._get_lidar_range(
			# 	env.agent.config["lidar"]["num_lasers"], env.agent.lidar.start_phase_offset
			# )
			# point_x = lidar_data * np.cos(lidar_range)
			# point_y = lidar_data * np.sin(lidar_range)
			# point_z = np.ones(lidar_data.shape)
			# lidar_data = np.stack([point_x, point_y, point_z], axis=-1).astype(np.float32)
			# dim_data = struct.pack('i', len(lidar_data))
			# lidar_data = bytearray(lidar_data)
			# lidar_data = dim_data + lidar_data
			# try:
			# 	self.lidar_socket.send(lidar_data, zmq.NOBLOCK)
			# except zmq.error.Again:
			# 	msg = "ros_socket_server: error sending lidar"
			# 	if test:
			# 		raise ValueError(msg)
			# 	else:
			# 		print(msg)
			# del lidar_data

			state_and_lidar = o[0]['state_and_lidar']
			try:
				flat_data = state_and_lidar.flatten().astype(np.float32)
				dim = len(flat_data)
				dim_bytes = struct.pack('i', dim)
				payload = dim_bytes + flat_data.tobytes()
				self.state_and_lidar_socket.send(payload, zmq.NOBLOCK)
			except zmq.error.Again:
				msg = "ros_socket_server: error sending state and lidar"
				if test:
					raise ValueError(msg)
				else:
					print(msg)
			del state_and_lidar

			if o[2]:
				break

		stop_action = struct.pack('ff', 0.0, 0.0)
		self.context.socket(zmq.PUSH).connect("ipc:///tmp/cmd_vel").send(stop_action)
		cmd_socket.close()
		env.close()

		print("\n######################")
		print(f"##### FINAL REWARD: {reward} #####")
		print("######################\n")


def main(test=False):
	server = RosSocketServer()
	server.run(test)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--test", action="store_true")
	args = parser.parse_args()
	main(args.test)
