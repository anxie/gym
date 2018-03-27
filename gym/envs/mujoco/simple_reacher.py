import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gc
import glob
import os
from natsort import natsorted

class SimpleReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        gc.enable()
        utils.EzPickle.__init__(self)
        self.xml_paths = natsorted(glob.glob(os.path.join(os.path.dirname(__file__), "assets/sim_vision_reach_xmls/*")))
        self.xml_iter = iter(self.xml_paths)
        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset_model(self):
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

    def get_image(self, width=64, height=64):
        return self.sim.get_image(width, height)

    def next(self):
        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 2)
