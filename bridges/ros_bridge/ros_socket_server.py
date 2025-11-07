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
	# STATE = "state"
	LIDAR = "state_and_lidar"

	def __init__(self, config):
		super(CustomizeObs, self).__init__(config)
		self.img_obs = ImageObservation(config, config["vehicle_config"]["image_source"], config["norm_pixel"])
		# self.state_obs = StateObservation(config)
		self.lidar_obs = LidarStateObservation(config)

	@property
	def observation_space(self):
		return gym.spaces.Dict(
			{
				self.IMAGE: self.img_obs.observation_space,
				# self.STATE: self.state_obs.observation_space
				self.LIDAR: self.lidar_obs.observation_space
			}
		)

	def observe(self, vehicle: BaseVehicle):
		return {self.IMAGE: self.img_obs.observe(), 
				# self.STATE: self.state_obs.observe(vehicle), 
				self.LIDAR: self.lidar_obs.observe(vehicle)}

	def destroy(self):
		super(CustomizeObs, self).destroy()
		self.img_obs.destroy()
		# self.state_obs.destroy()
		self.lidar_obs.destroy()

# ------------- ENV CLASS FOR GYMNASIUM COMPATIBILITY --------------
class MetaDriveEnvGymnasium(MetaDriveEnv):
	def reset(self, *, seed=None, options=None):
		obs, info = super().reset(seed=seed)
		return obs, info
	
	def reward_function(self, vehicle_id: str):
		"""
		Override this func to get a new reward function
		:param vehicle_id: id of BaseVehicle
		:return: reward
		"""
		vehicle = self.agents[vehicle_id]
		step_info = dict()

		# Reward for moving forward in current lane
		if vehicle.lane in vehicle.navigation.current_ref_lanes:
			current_lane = vehicle.lane
			positive_road = 1
		else:
			current_lane = vehicle.navigation.current_ref_lanes[0]
			current_road = vehicle.navigation.current_road
			positive_road = 1 if not current_road.is_negative_road() else -1
		long_last, _ = current_lane.local_coordinates(vehicle.last_position)
		long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

		# reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
		if self.config["use_lateral_reward"]:
			lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
		else:
			lateral_factor = 1.0

		reward = 0.0
		reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
		reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

		step_info["step_reward"] = reward

		if self._is_arrive_destination(vehicle):
			reward = +self.config["success_reward"] # 10
		elif self._is_out_of_road(vehicle):
			reward = -self.config["out_of_road_penalty"] # -5
		elif vehicle.crash_vehicle:
			reward = -self.config["crash_vehicle_penalty"] # -5
		elif vehicle.crash_object:
			reward = -self.config["crash_object_penalty"] # -5
		elif vehicle.crash_sidewalk:
			reward = -self.config["crash_sidewalk_penalty"]
		step_info["route_completion"] = vehicle.navigation.route_completion

		return reward, step_info
	
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

	# 	# env = MetaDriveEnv(config)
	# 	env = MetaDriveEnvGymnasium(config)
		
	# 	# try:
	# 	env.reset()
	# 	print(HELP_MESSAGE)
	# 	# env.agent.expert_takeover = False
		
	# 	# crea socket per ricevere comandi ######### NUOVO
	# 	cmd_socket = self.context.socket(zmq.PULL)
	# 	cmd_socket.bind("ipc:///tmp/cmd_vel")  # ROS2 pubblicherà qui
	# 	cmd_socket.set_hwm(5)
		
	# 	reward = 0.0
	def run(self, test=False):
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
			map="SC",
			use_render=True,
			image_observation=True,
			agent_observation=CustomizeObs,
			manual_control=False,
			random_spawn_lane_index=False,
			num_scenarios=env_cfg["num_scenarios"],
			traffic_density=env_cfg["traffic_density"],
			accident_prob=0,
			log_level=50,
			out_of_road_penalty=env_cfg["out_of_road_penalty"],
			crash_sidewalk_penalty=env_cfg["crash_sidewalk_penalty"],
			vehicle_config=vehicle_config,
			sensors={"rgb_camera": (RGBCamera, img_shape, img_shape)},
			stack_size=1
		)

		env = MetaDriveEnvGymnasium(config)
		env.training = False
		env.reset()
		print(HELP_MESSAGE)

		# ---- ZMQ socket per comandi ROS2 ----
		ipc_path = "/tmp/cmd_vel"
		if os.path.exists(ipc_path):
			print(f"Removing state IPC file at {ipc_path}") # evito che rimangono messaggi nel nodo ros
			os.remove(ipc_path)
		cmd_socket = self.context.socket(zmq.PULL)
		cmd_socket.bind("ipc:///tmp/cmd_vel")
		cmd_socket.set_hwm(5)

		# Thread di ricezione comandi (PRIMA del loop)
		def command_listener():
			print("Thread comandi avviato")
			while self.running:
				try:
					msg = cmd_socket.recv(flags=0)
					steering, throttle = struct.unpack('ff', msg)
					self.current_action = [steering, throttle]
				except Exception as e:
					print("Errore ricezione comando:", e)
					break

		cmd_thread = threading.Thread(target=command_listener, daemon=True)
		cmd_thread.start()

		# ---- Loop principale di simulazione ----
		reward = 0.0
		try:
			while self.running:
				start = time.time()
				action = self.current_action
				print("ACTION RECEIVED:", action)
				obs, r, done, truncated, info = env.step(action)
				reward += r

				# --- Invio sensori ---
				try:
					image_data = obs['image'][..., -1]
					image_data = (image_data * 255).astype(np.uint8)
					dim_data = struct.pack('ii', image_data.shape[1], image_data.shape[0])
					payload = bytearray(dim_data + image_data.tobytes())
					self.img_socket.send(payload, zmq.NOBLOCK)
				except zmq.error.Again:
					print("Errore invio immagine")

				try:
					state_and_lidar = obs['state_and_lidar']
					flat_data = state_and_lidar.flatten().astype(np.float32)
					payload = struct.pack('i', len(flat_data)) + flat_data.tobytes()
					self.state_and_lidar_socket.send(payload, zmq.NOBLOCK)
				except zmq.error.Again:
					print("Errore invio lidar")

				env.render()

				elapsed = time.time() - start
				time.sleep(max(0, 0.02 - elapsed))  # ~50Hz

				if done:
					env.reset()

		except KeyboardInterrupt:
			print("\nInterruzione da tastiera, chiusura in corso...")
			self.running = False

		finally:
			cmd_thread.join(timeout=1)
			cmd_socket.close()
			env.close()
			print("Simulazione terminata, reward totale:", reward)


			
		# 	reward = 0.0
		# 	while True:
		# 		try:
		# 			ricevi comandi come due float concatenati
		# 			msg = cmd_socket.recv(flags=zmq.NOBLOCK)
		# 			steering, throttle  = struct.unpack('ff', msg)
		# 			action = [steering, throttle]
		# 		except zmq.Again:
		# 			nessun comando disponibile, mantieni ultima azione
		# 			action = [0.0, 0.0]
					
		# 		############################################
		# 		print("ACTION RECEIVED:", action)
		# 		o = env.step(action)
				
		# 		reward += o[1]
		# 		print("OBSERVATION:", o) # mi assicuro che env.step() restituisca qualcosa di compatibile con stable_baselines3, ovvero obs, rew, done, truncated, info
				
		# 		env.render()
		# 		if test:
		# 			image_data = np.zeros((512, 512, 3))  # fake data for testing
		# 			image_data[::16, :, :] = 255
		# 		else:
				
		# 		#################
		# 		SEND IMAGE DATA
		# 		#################
		# 		print("Image shape to send",  o[0]['image'].shape) # Image shape to send (84, 84, 3, 1)
		# 		image_data = o[0]['image'][..., -1] # prende solo l'ultimo frame, quindi shape = (84,84,3)
		# 		print("Image shape send", image_data.shape) # Image shape send (84, 84, 3)
		# 		send via socket
		# 		image_data = image_data.astype(np.uint8) # COMMENTATO IO
		# 		Scala da [0,1] a [0,255] e cast a uint8
		# 		image_data = (image_data * 255).astype(np.uint8) # AGGIUNTO IO
		# 		dim_data = struct.pack('ii', image_data.shape[1], image_data.shape[0])
		# 		image_data = bytearray(image_data)
		# 		concatenate the dimensions and image data into a single byte array
		# 		image_data = dim_data + image_data
		# 		try:
		# 			self.img_socket.send(image_data, zmq.NOBLOCK)
		# 		except zmq.error.Again:
		# 			msg = "ros_socket_server: error sending image"
		# 			if test:
		# 				raise ValueError(msg)
		# 			else:
		# 				print(msg)
		# 		del image_data  # explicit delete to free memory
				
		# 		#################
		# 		SEND OBJECT DATA
		# 		#################
		# 		lidar_data, objs = env.agent.lidar.perceive(
		# 			env.agent,
		# 			env.engine.physics_world.dynamic_world,
		# 			env.agent.config["lidar"]["num_lasers"], # "num_lasers": 100
		# 			env.agent.config["lidar"]["distance"],   # "distance": 200
		# 			height=1.0,
		# 		)

		# 		ego_x = env.agent.position[0]
		# 		ego_y = env.agent.position[1]
		# 		ego_theta = np.arctan2(env.agent.heading[1], env.agent.heading[0])

		# 		num_data = struct.pack('i', len(objs))
		# 		obj_data = []
		# 		for obj in objs: # scorro dentro tutti gli oggetti rilevati dal radar
		# 			obj_x = obj.position[0]
		# 			obj_y = obj.position[1]
		# 			obj_theta = np.arctan2(obj.heading[1], obj.heading[0])

		# 			obj_x = obj_x - ego_x
		# 			obj_y = obj_y - ego_y
		# 			obj_x_new = np.cos(-ego_theta) * obj_x - np.sin(-ego_theta) * obj_y
		# 			obj_y_new = np.sin(-ego_theta) * obj_x + np.cos(-ego_theta) * obj_y

		# 			obj_data.append(obj_x_new)
		# 			obj_data.append(obj_y_new)
		# 			obj_data.append(obj_theta - ego_theta)
		# 			obj_data.append(obj.LENGTH)
		# 			obj_data.append(obj.WIDTH)
		# 			obj_data.append(obj.HEIGHT)
		# 		obj_data = np.array(obj_data, dtype=np.float32)
		# 		obj_data = bytearray(obj_data)
		# 		obj_data = num_data + obj_data
		# 		try:
		# 			self.obj_socket.send(obj_data, zmq.NOBLOCK)
		# 		except zmq.error.Again:
		# 			msg = "ros_socket_server: error sending objs"
		# 			if test:
		# 				raise ValueError(msg)
		# 			else:
		# 				print(msg)
		# 		del obj_data  # explicit delete to free memory

		# 		##################
		# 		SEND LIDAR DATA
		# 		##################
		# 		lidar_data = np.array(lidar_data) * env.agent.config["lidar"]["distance"]
		# 		lidar_range = env.agent.lidar._get_lidar_range(
		# 			env.agent.config["lidar"]["num_lasers"], env.agent.lidar.start_phase_offset
		# 		)
		# 		point_x = lidar_data * np.cos(lidar_range)
		# 		point_y = lidar_data * np.sin(lidar_range)
		# 		point_z = np.ones(lidar_data.shape)  # assume height = 1.0
		# 		lidar_data = np.stack([point_x, point_y, point_z], axis=-1).astype(np.float32)
		# 		dim_data = struct.pack('i', len(lidar_data))
		# 		lidar_data = bytearray(lidar_data)
		# 		concatenate the dimensions and lidar data into a single byte array
		# 		lidar_data = dim_data + lidar_data
		# 		try:
		# 			self.lidar_socket.send(lidar_data, zmq.NOBLOCK)
		# 		except zmq.error.Again:
		# 			msg = "ros_socket_server: error sending lidar"
		# 			if test:
		# 				raise ValueError(msg)
		# 			else:
		# 				print(msg)
		# 		del lidar_data  # explicit delete to free memory
				
		# 		#################
		# 		SEND STATE AND LIDAR 
		# 		#################
		# 		state_and_lidar = o[0]['state_and_lidar'] # perchè è un dizionario con un elemento
		# 		try:
		# 			flat_data = state_and_lidar.flatten().astype(np.float32)
		# 			dim = len(flat_data)

		# 			Prepara il pacchetto: [dim (4 bytes)] + [data (float32 * dim)]
		# 			dim_bytes = struct.pack('i', dim)
		# 			payload = dim_bytes + flat_data.tobytes()

		# 			Invia in modalità non bloccante
		# 			self.state_and_lidar_socket.send(payload, zmq.NOBLOCK)

		# 		except zmq.error.Again:
		# 			msg = "ros_socket_server: error sending state and lidar"
		# 			if test:
		# 				raise ValueError(msg)
		# 			else:
		# 				print(msg)
		# 		del state_and_lidar
						
		# 		#################
		# 		CHECK DONE
		# 		#################
		# 		if o[2]:  # done
		# 			break

		# except Exception as e:
		# 	stop_action = struct.pack('ff', 0.0, 0.0)
		# 	self.context.socket(zmq.PUSH).connect("ipc:///tmp/cmd_vel").send(stop_action)
		# 	cmd_socket.close()
		# 	env.close()
			
		# 	print("\n######################")
		# 	print("##### FINAL REWARD:" + str(reward) + " #####")
		# 	print("######################\n")
		# finally:
		# 	stop_action = struct.pack('ff', 0.0, 0.0)
		# 	self.context.socket(zmq.PUSH).connect("ipc:///tmp/cmd_vel").send(stop_action)
		# 	cmd_socket.close()
		# 	env.close()
			
		# 	print("\n######################")
		# 	print("##### FINAL REWARD:" + str(reward) + " #####")
		# 	print("######################\n")

def main(test=False):
	server = RosSocketServer()
	server.run(test)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--test", action="store_true")
	args = parser.parse_args()
	main(args.test)