from functools import partial
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output

import gymnasium as gym
from gymnasium import spaces

from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif

from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from sb3_contrib import TQC

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

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


# ------------- --------------- -------------- #
# ------------- --------------- -------------- #
# ------------- --------------- -------------- #

# ------------- WANDB -------------- 
config_wandb = {
	"policy_type": "MultiInputPolicy",
	"total_timesteps": total_timesteps,
	"env_name": "metadrive",
}

run = wandb.init(
	project="metadrive-training",
	config=config_wandb,
	sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
	monitor_gym=True  # auto-upload the videos of agents playing the game
)

# crea una directory per i log
log_dir = f"logs/{wandb.run.name}"
os.makedirs(log_dir, exist_ok=True)

# collega SB3 al logger wandb
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

# ------------- CALLBACK FOR LOGGING TRAINING PROGRESS -------------- 
class TrainingLoggerCallback(BaseCallback):
	def __init__(self, verbose=0):
		super().__init__(verbose)
		self.n_rollout_steps = 0

	def _on_training_start(self):
		if self.verbose > 0:
			print("\n--- Training Started ---")
			print(f"Total timesteps requested: {self.model._total_timesteps}")
			print(f"Batch Size (n_steps * n_envs): {self.model.n_steps * self.model.n_envs}")

	def _on_rollout_start(self):
		self.n_rollout_steps = 0
		if self.verbose > 0:
			print(f"\n[TIMESTEP: {self.num_timesteps}] Starting new data collection rollout.")

	def _on_step(self) -> bool:
		if self.num_timesteps % 50 == 0:
			obs = self.locals.get("new_obs", None)
			print(f"Sample observation at timestep {self.num_timesteps}: {obs}")
		self.n_rollout_steps += 1
		if self.verbose > 1 and self.n_rollout_steps % 100 == 0:
			print(f"  > Rollout step {self.n_rollout_steps}/{self.model.n_steps} collected.")
		return True

	def _on_rollout_end(self):
		if self.verbose > 0:
			
			print(f"Data collection finished. Total steps collected: {self.n_rollout_steps}.")
			print("Starting policy optimization (update)...")

	def _on_training_end(self):
		if self.verbose > 0:
			print("\n--- Training Finished ---")

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

# ------------- VEHICLE CONFIG --------------
vehicle_config = {
		"show_navi_mark": True,
		"enable_reverse": True,
		"show_lidar": True,
		"image_source":"rgb_camera",

		# "side_detector": {
		# 	"num_lasers": 32,
		# 	"distance": 50
		# },
		# "show_side_detector": True,

		# "lane_line_detector": {
		# 	"num_lasers": 32,
		# 	"distance": 50
		# },
		# "show_lane_line_detector": True,

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
config = (dict(map="C", # map="SCSTRS",
				use_render=True,
				image_observation=True,
				agent_observation=CustomizeObs, # custom observation

				manual_control=False,
				random_spawn_lane_index=False,
				num_scenarios=env_cfg["num_scenarios"],
				traffic_density=env_cfg["traffic_density"],
				accident_prob=0,
				log_level=50,
				
				out_of_road_penalty=env_cfg["out_of_road_penalty"], # di default è 10
				crash_sidewalk_penalty=env_cfg["crash_sidewalk_penalty"], # di default è 0
				
				vehicle_config = vehicle_config,
				sensors={"rgb_camera": (RGBCamera, img_shape, img_shape)},
				stack_size=1 # tengo solo l'ultimo frame della camera, perchè memorizza più frame in stack
			))

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

# ----------- INIT ENV --------------
def create_env():
	env_base = MetaDriveEnvGymnasium(config) # creo l'ambiente --> Observation space:  Dict('image': Box(-0.0, 1.0, (84, 84, 3, 1), float32), 'state_and_lidar': Box(-0.0, 1.0, (275,), float32))
	env_monitored = Monitor(env_base)
	return env_monitored

# ------------- CUSTOM CNN FOR IMAGE PROCESSING WITH BATCH NORM --------------
class CustomCNN(BaseFeaturesExtractor):
	def __init__(self, observation_space, features_dim=1024): # Increased features_dim
		super().__init__(observation_space, features_dim)
		# Image shape is (H, W, C, S) where S is stack_size
		n_input_channels = observation_space["image"].shape[2] 

		self.cnn = nn.Sequential(
			nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Flatten(),
		)

		# Dynamic calculation of the size after the CNN
		with torch.no_grad():
			sample = observation_space["image"].sample()[None]
			if sample.ndim == 5:
				sample = sample.squeeze(-1)
			sample_torch = torch.as_tensor(sample).permute(0,3,1,2).float() # permute from [B,H,W,C] to [B,C,H,W]
			n_flatten = self.cnn(sample_torch).shape[1]

		# The final linear layer uses the new, larger features_dim
		self.linear = nn.Sequential(
			nn.Linear(n_flatten, features_dim),
			nn.ReLU()
		)

	def forward(self, observations):
		x = observations["image"]
		if x.ndim == 5:
			x = x.squeeze(-1) # squeeze extra dimension
		x = x.permute(0, 3, 1, 2) # (B,H,W,C) -> (B,C,H,W)
		x = self.cnn(x)
		return self.linear(x)

# -------------- DEFINE CALLBACKS --------------
callback_train = TrainingLoggerCallback(verbose=1)
wandb_callback = WandbCallback(gradient_save_freq=5, model_save_path=f"models/{run.id}", verbose=2)

# ------------ TRAINING ----------------
set_random_seed(0)
# 1 subprocess to rollout (con use_render=True non posso farne girare di piu di uno, perchè si blocca)
train_env = DummyVecEnv([partial(create_env)]) # wrappo dentro DummyVecEnv per SB3 perchè loro lavorano con immagini

print("Observation space: ", train_env.observation_space) # Observation space:  Dict('image': Box(-0.0, 1.0, (84, 84, 3, 1), float32), 'state_and_lidar': Box(-0.0, 1.0, (135,), float32))
### shape of 'state' --> shape = self.ego_state_obs_dim + self.navi_dim (9 + 5*2)

# Observation space:
#	Dict('image': Box(-0.0, 1.0, (84, 84, 3, 1), float32), 
#		 'state_and_lidar': Box(-0.0, 1.0, (135,), float32))

# ------------ ACTION NOISE FOR DDPG --------------
n_actions = train_env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(
	mean=np.zeros(n_actions),
	sigma=0.3 * np.ones(n_actions),
	theta=0.15
)

# ------------ CUSTOM POLICY KEYWORDS --------------
policy_kwargs = dict(
	features_extractor_class=CustomCNN,
	features_extractor_kwargs=dict(features_dim=512)
)

# ------------ MODEL DEFINITION --------------
if model_name.upper() == "PPO":
	print("Start training PPO model...")
	model = PPO("MultiInputPolicy", train_env, n_steps=n_steps, batch_size=batch_size, verbose=1, device="cuda", tensorboard_log=f"runs/{run.id}", ent_coef = 0.001, policy_kwargs=policy_kwargs)
elif model_name.upper() == "DDPG":
	print("Start training DDPG model...")
	model = DDPG("MultiInputPolicy", train_env, verbose=1, device="cuda", batch_size=batch_size, buffer_size=buffer_size, learning_starts=learning_starts, action_noise=action_noise, tensorboard_log=f"runs/{run.id}", policy_kwargs=policy_kwargs)
elif model_name.upper() == "TD3":
	print("Start training TD3 model...")
	model = TD3("MultiInputPolicy", train_env, verbose=1, device="cuda", batch_size=batch_size, buffer_size=buffer_size, tensorboard_log=f"runs/{run.id}", policy_kwargs=policy_kwargs)
elif model_name.upper() == "SAC":
	print("Start training SAC model...")
	model = SAC("MultiInputPolicy", train_env, verbose=1, device="cuda", tau=0.005, gamma=0.99, train_freq=1, batch_size=batch_size, buffer_size=buffer_size, tensorboard_log=f"runs/{run.id}")
elif model_name.upper() == "TQC":
	print("Start training TQC model...")
	model = TQC("MultiInputPolicy", train_env, learning_rate=3e-4, buffer_size=1000, batch_size=128, learning_starts=10_000, train_freq=1, top_quantiles_to_drop_per_net=2, gradient_steps=1, tau=0.005, gamma=0.99, verbose=1, tensorboard_log=f"runs/{run.id}")
else:
	raise ValueError(f"Unsupported model type: {model_name}")
	
model.set_logger(new_logger)
model.learn(total_timesteps=total_timesteps, log_interval=4, callback=[callback_train, wandb_callback])
print("Training is finished!")
clear_output()
wandb.finish()

# ----------- SAVE MODEL --------------
model_save_path = f"models/{model_name.lower()}_metadrive_multimodal"
model.save(model_save_path)

################################################

'''
Sample observation at timestep 700: OrderedDict([('image', array([[[[[0.6509804 ],
		  [0.6666667 ],
		  [0.65882355]],

		 [[0.654902  ],
		  [0.67058825],
		  [0.6627451 ]],

		 [[0.6509804 ],
		  [0.6666667 ],
		  [0.65882355]],

		 ...,

		 [[0.54901963],
		  [0.47058824],
		  [0.38431373]],

		 [[0.5529412 ],
		  [0.47058824],
		  [0.38431373]],

		 [[0.5529412 ],
		  [0.47058824],
		  [0.38431373]]],


		[[[0.6431373 ],
		  [0.654902  ],
		  [0.6392157 ]],

		 [[0.6509804 ],
		  [0.6627451 ],
		  [0.64705884]],

		 [[0.654902  ],
		  [0.6666667 ],
		  [0.6509804 ]],

		 ...,

		 [[0.54509807],
		  [0.4627451 ],
		  [0.3764706 ]],

		 [[0.5568628 ],
		  [0.46666667],
		  [0.3764706 ]],

		 [[0.5529412 ],
		  [0.4627451 ],
		  [0.3764706 ]]],


		[[[0.64705884],
		  [0.64705884],
		  [0.627451  ]],

		 [[0.64705884],
		  [0.6509804 ],
		  [0.6392157 ]],

		 [[0.6431373 ],
		  [0.654902  ],
		  [0.6392157 ]],

		 ...,

		 [[0.5411765 ],
		  [0.4509804 ],
		  [0.34901962]],

		 [[0.54509807],
		  [0.45882353],
		  [0.35686275]],

		 [[0.5568628 ],
		  [0.47058824],
		  [0.36862746]]],


		...,


		[[[0.6156863 ],
		  [0.58431375],
		  [0.59607846]],

		 [[0.62352943],
		  [0.5882353 ],
		  [0.6039216 ]],

		 [[0.61960787],
		  [0.5882353 ],
		  [0.6039216 ]],

		 ...,

		 [[0.58431375],
		  [0.5529412 ],
		  [0.5686275 ]],

		 [[0.5921569 ],
		  [0.56078434],
		  [0.5764706 ]],

		 [[0.6117647 ],
		  [0.5764706 ],
		  [0.5921569 ]]],


		[[[0.61960787],
		  [0.5882353 ],
		  [0.6       ]],

		 [[0.62352943],
		  [0.5921569 ],
		  [0.6039216 ]],

		 [[0.627451  ],
		  [0.59607846],
		  [0.60784316]],

		 ...,

		 [[0.6039216 ],
		  [0.5686275 ],
		  [0.58431375]],

		 [[0.5803922 ],
		  [0.54509807],
		  [0.56078434]],

		 [[0.5568628 ],
		  [0.5254902 ],
		  [0.5411765 ]]],


		[[[0.627451  ],
		  [0.59607846],
		  [0.60784316]],

		 [[0.627451  ],
		  [0.59607846],
		  [0.60784316]],

		 [[0.6313726 ],
		  [0.6       ],
		  [0.6117647 ]],

		 ...,

		 [[0.6117647 ],
		  [0.5803922 ],
		  [0.5921569 ]],

		 [[0.60784316],
		  [0.5764706 ],
		  [0.5921569 ]],

		 [[0.5921569 ],
		  [0.5568628 ],
		  [0.57254905]]]]], dtype=float32)), ('state_and_lidar', array([[1.4414546e-01, 4.3918788e-01, 5.3551280e-01, 1.7889541e-02,
		5.0765461e-01, 9.5927799e-01, 8.5712582e-01, 2.1305289e-04,
		6.8769294e-01, 8.8250619e-01, 5.0061655e-01, 0.0000000e+00,
		5.0000000e-01, 5.0000000e-01, 8.8902438e-01, 1.8590023e-01,
		8.4269744e-01, 1.0000000e+00, 9.2771590e-01, 6.3541347e-01,
		5.0623679e-01, 5.2905715e-01, 4.9250871e-01, 6.7225301e-01,
		4.8348245e-01, 5.3399354e-01, 4.7891948e-01, 6.7495084e-01,
		4.9188325e-01, 5.3247625e-01, 4.8026243e-01, 6.8105698e-01,
		4.6765783e-01, 5.3093433e-01, 4.7499466e-01, 1.0000000e+00,
		2.5754490e-01, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.0000000e+00,
		1.0000000e+00, 1.0000000e+00, 7.4440169e-01, 6.8892241e-01,
		7.1111643e-01, 1.0000000e+00, 1.0000000e+00, 6.1691970e-01,
		1.0000000e+00, 5.7150388e-01, 5.9005457e-01, 5.6154972e-01,
		5.0907940e-01, 5.0570995e-01, 4.9754983e-01, 4.4232917e-01,
		3.6612144e-01, 1.0000000e+00, 3.5188678e-01]], dtype=float32))])
'''