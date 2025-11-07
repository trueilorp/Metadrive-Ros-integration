from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

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