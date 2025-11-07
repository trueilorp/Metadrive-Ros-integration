from metadrive.obs.observation_base import BaseObservation
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.image_obs import ImageObservation
from metadrive.obs.state_obs import LidarStateObservation
import gymnasium as gym
from gymnasium import spaces

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