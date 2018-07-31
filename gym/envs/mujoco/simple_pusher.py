import numpy as np
import glob
import os
from random import shuffle
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py

class SimplePusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.xml_paths = glob.glob(os.path.join(os.path.dirname(__file__), "assets/pushing/*"))
        shuffle(self.xml_paths)
        self.xml_iter = iter(self.xml_paths)
        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 5)

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, None, done, dict(reward_dist=reward_dist,
                reward_ctrl=reward_ctrl)

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])

    def get_image(self, width=64, height=64):
        return self.sim.render(width, height, camera_name="camera")

    # def get_image(self, width=100, height=100):
    #     _, depth_image = self.sim.render(width, height, camera_name="wrist_camera", depth=True)
    #     world_image = self.sim.render(width, height, camera_name="camera")
    #     return world_image, (depth_image * 255.0).astype(np.uint8)

    def next(self):
        mujoco_env.MujocoEnv.__init__(self, self.xml_iter.__next__(), 5)
