import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.dynamic_mjc.sweeper_with_gripper import sweeper_with_gripper

class SweeperWithGripperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
                 substeps=50, 
                 im_height=64, 
                 im_width=64,
                 log_video=True,
                 video_substeps=10,
                 camera_name="overheadcam"):

        utils.EzPickle.__init__(self)

        self.substeps = substeps
        self.im_height = im_height
        self.im_width = im_width
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.camera_name = camera_name

        self.cubes_pos = [0, 0]
        self.sweeper_pos = [0, 0]
        model = sweeper_with_gripper(cubes_pos=self.cubes_pos, sweeper_pos=self.sweeper_pos)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):  
        # compute reward
        video_frames = self.pos_control(a)
        ob = self._get_obs()
        done = False
        return ob, 0, done, dict(reward_tool=None, reward_obj=None, video_frames=video_frames)

    def pos_control(self, a):
        a = np.array(a)
        torque = 0.0

        # move gripper down at grip iteration
        if a[-1] == 1:
            torque = 0.1
            action = np.asarray([0, 0, 0, 0, torque])
            self.do_pos_simulation_with_substeps(action)

        if a[-1] == 1:
            z_com = 0
        else:
            z_com = a[2]

        action = np.asarray([a[0], a[1], z_com, 0, torque])

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
        self.cubes_pos = [np.random.uniform(-0.35,0.35), np.random.uniform(0.15, 0.3)]
        # self.sweeper_pos = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.05, 0.1)]
        self.sweeper_pos = [0,0]
        print(self.cubes_pos)
        model = sweeper_with_gripper(cubes_pos=self.cubes_pos, sweeper_pos=self.sweeper_pos)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
