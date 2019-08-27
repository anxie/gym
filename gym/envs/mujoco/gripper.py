import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.dynamic_mjc.gripper import gripper


GOAL_POSITIONS = [np.asarray([-0.5, -0.5]),     # top left
                  np.asarray([-0.5, 0.5]),      # bottom left
                  np.asarray([0.5, -0.5]),      # top right
                  np.asarray([0.5, 0.5])        # bottom right
                  ]


class GripperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 seed=None,
                 substeps=20, 
                 im_height=64, 
                 im_width=64,
                 log_video=True,
                 video_substeps=10,
                 camera_name="overheadcam",
                 max_episode_steps=10):

        utils.EzPickle.__init__(self)

        self.substeps = substeps
        self.im_height = im_height
        self.im_width = im_width
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.camera_name = camera_name
        self.max_episode_steps = max_episode_steps
        self.curr_step = 0
        self.target_pos = None

        model, self.objects = gripper()
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):  
        video_frames = self.pos_control(a)
        obs = self._get_obs()
        done = self.curr_step >= self.max_episode_steps

        obj_pos = self.data.get_body_xpos('obj_1').copy()
        rew = -np.linalg.norm(obj_pos - self.target_pos)

        return obs, rew, done, dict(video_frames=video_frames)

    def pos_control(self, a):
        action = np.array(a)
        torque = 0.0

        if a[-1] == 1:
            torque = 0.1
            action = np.asarray([0, 0, 0, 0, torque])
            self.do_pos_simulation_with_substeps(action)

        action[-1] = torque

        video_frames = [self.do_pos_simulation_with_substeps(action)]
        return video_frames

    def do_pos_simulation_with_substeps(self, a, substeps=None):
        if substeps is None:
            substeps = self.substeps

        qpos_curr = self.sim.data.qpos[:4]
        a_pos = a[:4]

        if self.log_video:
            video_frames = np.zeros((int(substeps / self.video_substeps),
                                     self.im_height, self.im_width, 3))
        else:
            video_frames = None

        self.sim.data.ctrl[:-1] = qpos_curr + a_pos
        self.sim.data.ctrl[-1] = a[-1]

        for i in range(substeps):
            self.sim.step()

            if i % self.video_substeps == 0 and self.log_video:
                video_frames[int(i / self.video_substeps)] = self.sim.render(
                    self.im_height, self.im_width, camera_name=self.camera_name)

        return video_frames
        
    def reset_model(self):
        model, self.objects = gripper()
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

        self.curr_step = 0
        self.target_pos = np.random.choice(GOAL_POSITIONS)

        # qpos = self.init_qpos
        # qvel = self.init_qvel
        # self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.target_pos
        ])
