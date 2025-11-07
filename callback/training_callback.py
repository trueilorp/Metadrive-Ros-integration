from stable_baselines3.common.callbacks import BaseCallback

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