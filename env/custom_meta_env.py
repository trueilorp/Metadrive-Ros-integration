from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from metadrive.envs import MetaDriveEnv

class MetaDriveEnvGymnasium(MetaDriveEnv):
	def reset(self, *, seed=None, options=None):
		obs, info = super().reset(seed=seed)
		return obs, info

	def reward_function(self, vehicle_id: str):
		vehicle = self.agents[vehicle_id]
		step_info = {}
		# Minimal reward: forward progress + speed
		if vehicle.lane in vehicle.navigation.current_ref_lanes:
			current_lane = vehicle.lane
			positive_road = 1
		else:
			current_lane = vehicle.navigation.current_ref_lanes[0]
			current_road = vehicle.navigation.current_road
			positive_road = 1 if not current_road.is_negative_road() else -1

		long_last, _ = current_lane.local_coordinates(vehicle.last_position)
		long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

		if self.config.get("use_lateral_reward", False):
			lateral_factor = max(0.0, 1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width())
		else:
			lateral_factor = 1.0

		reward = 0.0
		reward += self.config.get("driving_reward", 1.0) * (long_now - long_last) * lateral_factor * positive_road
		reward += self.config.get("speed_reward", 0.0) * (vehicle.speed_km_h / max(vehicle.max_speed_km_h, 1e-6)) * positive_road

		step_info["step_reward"] = reward

		if self._is_arrive_destination(vehicle):
			reward = +self.config.get("success_reward", 10)
		elif self._is_out_of_road(vehicle):
			reward = -self.config.get("out_of_road_penalty", 5)
		elif vehicle.crash_vehicle:
			reward = -self.config.get("crash_vehicle_penalty", 5)
		elif vehicle.crash_object:
			reward = -self.config.get("crash_object_penalty", 5)
		elif vehicle.crash_sidewalk:
			reward = -self.config.get("crash_sidewalk_penalty", 1)
		step_info["route_completion"] = vehicle.navigation.route_completion

		return reward, step_info