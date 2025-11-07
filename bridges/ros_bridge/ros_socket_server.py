#!/usr/bin/env python3
# ros_socket_server.py
"""
Launch MetaDrive, receive cmd_vel via IPC, and push sensor data via ZMQ IPC sockets:
- /tmp/rgb_camera         (image: [width:int, height:int] + raw bytes)
- /tmp/lidar              (lidar points: [n:int] + float32 * n * 3)
- /tmp/obj                (objects: [n:int] + float32 * (6 * n) each obj [x,y,theta,length,width,height])
- /tmp/state_and_lidar    (state+lidar flattened: [len:int] + float32 * len)
Commands are received from /tmp/cmd_vel as two floats (steering, throttle) packed with struct 'ff'.
"""

import argparse
import struct
import json
import os
import time
import threading
import numpy as np
import zmq

import gymnasium as gym
from metadrive.constants import HELP_MESSAGE
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.component.vehicle.base_vehicle import BaseVehicle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from obs.custom_obs import CustomizeObs
from env.custom_meta_env import MetaDriveEnvGymnasium

# ---------- Read config ----------
with open("../../../config.json", "r") as f:
	cfg = json.load(f)

train_cfg = cfg.get("training", {})
env_cfg = cfg.get("environment", {})
veh_cfg = cfg.get("vehicle", {})

# parameters used below (with safe defaults)
img_shape = train_cfg.get("img_shape", 84)

# -------------------------------------------------

class RosSocketServer:
	def __init__(self):
		self.context = zmq.Context().instance()
		self.context.setsockopt(zmq.IO_THREADS, 2)

		# PUSH sockets for sensors (server -> ROS bridge)
		self.img_socket = self.context.socket(zmq.PUSH)
		self.img_socket.setsockopt(zmq.SNDBUF, 4_194_304)
		self.img_socket.bind("ipc:///tmp/rgb_camera")
		self.img_socket.set_hwm(5)

		self.lidar_socket = self.context.socket(zmq.PUSH)
		self.lidar_socket.setsockopt(zmq.SNDBUF, 4_194_304)
		self.lidar_socket.bind("ipc:///tmp/lidar")
		self.lidar_socket.set_hwm(5)

		self.obj_socket = self.context.socket(zmq.PUSH)
		self.obj_socket.setsockopt(zmq.SNDBUF, 4_194_304)
		self.obj_socket.bind("ipc:///tmp/obj")
		self.obj_socket.set_hwm(5)

		self.state_and_lidar_socket = self.context.socket(zmq.PUSH)
		self.state_and_lidar_socket.setsockopt(zmq.SNDBUF, 4_194_304)
		self.state_and_lidar_socket.bind("ipc:///tmp/state_and_lidar")
		self.state_and_lidar_socket.set_hwm(5)

		# control
		self.current_action = [0.0, 0.0]
		self.running = True

	def _safe_send(self, sock, payload: bytes, desc: str):
		try:
			sock.send(payload, zmq.NOBLOCK)
		except zmq.error.Again:
			print(f"ZMQ {desc}: send would block (skip)")
		except Exception as e:
			print(f"ZMQ {desc}: unexpected send error: {e}")

	def run(self, test=False):
		# ---------- VEHICLE & ENV CONFIG ----------
		vehicle_config = {
			"show_navi_mark": True,
			"enable_reverse": True,
			"show_lidar": True,
			"image_source": "rgb_camera",
			"lidar": {
				"num_lasers": int(veh_cfg.get("num_lasers", 64)),
				"distance": float(veh_cfg.get("distance", 120.0)),
				"num_others": int(veh_cfg.get("num_others", 8)),
				"add_others_navi": bool(veh_cfg.get("add_others_navi", True)),
				"gaussian_noise": 0.0,
				"dropout_prob": 0.0
			}
		}

		config = dict(
			map="SC",
			use_render=True,
			image_observation=True,
			agent_observation=CustomizeObs,
			manual_control=False,
			random_spawn_lane_index=False,
			num_scenarios=int(env_cfg.get("num_scenarios", 1)),
			traffic_density=float(env_cfg.get("traffic_density", 0.0)),
			accident_prob=0,
			log_level=50,
			out_of_road_penalty=float(env_cfg.get("out_of_road_penalty", 10)),
			crash_sidewalk_penalty=float(env_cfg.get("crash_sidewalk_penalty", 0)),
			vehicle_config=vehicle_config,
			sensors={"rgb_camera": (RGBCamera, img_shape, img_shape)},
			stack_size=1
		)

		env = MetaDriveEnvGymnasium(config)
		env.training = False
		env.reset()
		print(HELP_MESSAGE)

		# Setup command socket (PULL) - ROS will PUSH commands to this IPC
		ipc_path = "/tmp/cmd_vel"
		# Remove stale IPC file if exists
		try:
			if os.path.exists(ipc_path):
				print(f"Removing existing IPC file {ipc_path}")
				os.remove(ipc_path)
		except Exception as e:
			print("Warning: could not remove ipc file:", e)

		cmd_socket = self.context.socket(zmq.PULL)
		cmd_socket.bind("ipc:///tmp/cmd_vel")
		cmd_socket.set_hwm(5)

		# Command listener thread
		def command_listener():
			print("Command listener thread started")
			while self.running:
				try:
					msg = cmd_socket.recv(flags=0)  # blocking read
					if not msg:
						continue
					try:
						steering, throttle = struct.unpack('ff', msg)
						self.current_action = [float(steering), float(throttle)]
					except struct.error:
						# if message size differs, try to interpret differently
						print("Received malformed command packet")
				except Exception as e:
					if self.running:
						print("Command listener error:", e)
					break
			print("Command listener thread exiting")

		cmd_thread = threading.Thread(target=command_listener, daemon=True)
		cmd_thread.start()

		reward = 0.0
		try:
			while self.running:
				loop_start = time.time()
				action = self.current_action
				# step env
				try:
					obs, r, done, truncated, info = env.step(action)
					reward += r
				except TypeError:
					# some MetaDrive versions return 4-tuple
					obs, r, done, info = env.step(action)
					truncated = False
					reward += r

				# --- SEND IMAGE ---
				try:
					image_stack = obs.get('image')  # expected shape (H, W, C, stack) or (H,W,C)
					if image_stack is None:
						raise ValueError("No 'image' in obs")
					# select last frame in stack if present
					if image_stack.ndim == 4:
						image_data = image_stack[..., -1]
					else:
						image_data = image_stack
					# convert [0,1] floats to uint8
					if image_data.dtype != np.uint8:
						image_data = (np.clip(image_data, 0.0, 1.0) * 255.0).astype(np.uint8)
					h, w = int(image_data.shape[0]), int(image_data.shape[1])
					dim_data = struct.pack('ii', w, h)
					payload = dim_data + image_data.tobytes()
					self._safe_send(self.img_socket, payload, "image")
				except Exception as e:
					print("Error preparing/sending image:", e)

				# --- SEND STATE AND LIDAR (from observation) ---
				try:
					state_and_lidar = obs.get('state_and_lidar')
					if state_and_lidar is not None:
						flat = np.array(state_and_lidar).flatten().astype(np.float32)
						payload = struct.pack('i', len(flat)) + flat.tobytes()
						self._safe_send(self.state_and_lidar_socket, payload, "state_and_lidar")
				except Exception as e:
					print("Error preparing/sending state_and_lidar:", e)

				# --- SEND OBJECTS & LIDAR using agent's lidar perception (best-effort) ---
				lidar_data, objs = env.agent.lidar.perceive(
					env.agent,
					env.engine.physics_world.dynamic_world,
					env.agent.config["lidar"]["num_lasers"],
					env.agent.config["lidar"]["distance"],
					height=1.0,
				)

				ego_x = env.agent.position[0]
				ego_y = env.agent.position[1]
				ego_theta = np.arctan2(env.agent.heading[1], env.agent.heading[0])

				num_data = struct.pack('i', len(objs))
				obj_data = []
				for obj in objs:
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

				# convert lidar data to xyz
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
				# concatenate the dimensions and lidar data into a single byte array
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
				# render (if any)
				try:
					env.render()
				except Exception:
					pass

				# timing to approx 50Hz
				elapsed = time.time() - loop_start
				time.sleep(max(0, 0.02 - elapsed))

				# reset on done
				if done or truncated:
					env.reset()

		except KeyboardInterrupt:
			print("\nKeyboardInterrupt received, shutting down...")
			self.running = False
		except Exception as e:
			print("Unexpected server error:", e)
			self.running = False
		finally:
			# join thread and cleanup sockets/env
			self.running = False
			try:
				cmd_thread.join(timeout=1.0)
			except Exception:
				pass
			try:
				cmd_socket.close()
			except Exception:
				pass
			try:
				self.img_socket.close()
				self.lidar_socket.close()
				self.obj_socket.close()
				self.state_and_lidar_socket.close()
			except Exception:
				pass
			try:
				env.close()
			except Exception:
				pass
			print("Server stopped. Total reward:", reward)


def main(test=False):
	server = RosSocketServer()
	server.run(test)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--test", action="store_true", help="Run in test mode (keeps behavior similar)")
	args = parser.parse_args()
	main(args.test)

