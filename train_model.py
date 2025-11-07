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
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from nets.custom_cnn import CustomCNN
from callback.training_callback import TrainingLoggerCallback
from obs.custom_obs import CustomizeObs
from env.custom_meta_env import MetaDriveEnvGymnasium


# ------------- READ CONFIG PARAMETERS --------------
# --- Load config file ---
with open("config.json", "r") as f:
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
log_dir = f"wandb/logs/{wandb.run.name}"
os.makedirs(log_dir, exist_ok=True)

# collega SB3 al logger wandb
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

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

# ----------- INIT ENV --------------
def create_env():
	env_base = MetaDriveEnvGymnasium(config) # creo l'ambiente --> Observation space:  Dict('image': Box(-0.0, 1.0, (84, 84, 3, 1), float32), 'state_and_lidar': Box(-0.0, 1.0, (275,), float32))
	env_monitored = Monitor(env_base)
	return env_monitored

# -------------- DEFINE CALLBACKS --------------
callback_train = TrainingLoggerCallback(verbose=1)
wandb_callback = WandbCallback(gradient_save_freq=5, model_save_path=f"wandb/models/{run.id}", verbose=2)

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
model_save_path = f"models_trained/{model_zip_file}"
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