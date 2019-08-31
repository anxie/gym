import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.dynamic_mjc.wrist import wrist

class WristEnv(mujoco_env.MujocoEnv, utils.EzPickle):
	def __init__(self):
		self.im_width = 32
		self.im_height = 32

		model, _ = wrist()
		with model.asfile() as f:
			mujoco_env.MujocoEnv.__init__(self, f.name, 5)

	def step(self, a):
		reward = 0.0
		self.do_simulation(a, self.frame_skip)
		obs = self._get_obs()
		done = False
		frame = self._get_image_obs()
		return obs, reward, done, dict(frame=frame)

	def reset_model(self):
		model, objects = wrist()
		with model.asfile() as f:
			mujoco_env.MujocoEnv.__init__(self, f.name, 5)
		return objects

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat[:3],
			self.sim.data.qvel.flat[:3],
			self.get_body_com("obj_0")[:2],
			self.get_body_com("obj_1")[:2],
		]).reshape(-1)

	def _get_image_obs(self):
		return self.sim.render(self.im_width, self.im_height, camera_name="overheadcam")
