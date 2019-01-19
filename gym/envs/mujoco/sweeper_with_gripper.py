import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.dynamic_mjc.sweeper_with_gripper import sweeper_with_gripper

class SweeperWithGripperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, 
                 seed=None,
                 substeps=20, 
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

        self._seed = seed

        if self._seed is not None:
            np.random.seed(self._seed)

    def step(self, a):  
        # compute reward
        video_frames = self.pos_control(a)
        obs = self._get_obs()
        done = False

        sweeper_pos = obs[6:8]
        cube_x_pos = np.average([obs[13], obs[20], obs[27]])
        cube_y_pos = np.average([obs[14], obs[21], obs[28]])
        cube_pos = [cube_x_pos, cube_y_pos]

        return obs, 0, done, dict(video_frames=video_frames, sweeper_pos=sweeper_pos, cube_pos=cube_pos)

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
        step = (a_pos - qpos_curr) / substeps

        if self.log_video:
            video_frames = np.zeros((int(substeps / self.video_substeps),
                                     self.im_height, self.im_width, 3))
        else:
            video_frames = None

        self.sim.data.ctrl[:-1] = qpos_curr + a_pos
        self.sim.data.ctrl[-1] = a[-1]

        for i in range(substeps):
            # self.sim.data.ctrl[:-1] = qpos_curr + (i+1)*step
            self.sim.step()

            if i % self.video_substeps == 0 and self.log_video:
                video_frames[int(i / self.video_substeps)] = self.sim.render(
                    self.im_height, self.im_width, camera_name=self.camera_name)

        return video_frames
        
    def reset_model(self):
        if self._seed is not None:
            np.random.seed(self._seed)

        ################# for data collection ######################
        # self.cubes_pos = [np.random.uniform(-0.3,0.3), np.random.uniform(-0.1, 0.3)]  # for data collection
        # self.sweeper_pos = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]  # for random data collection
        # self.sweeper_pos = [0,0]  # for expert data collection





        ################# for testing planning #####################
        # self.cubes_pos = [np.random.uniform(0.1, 0.3), np.random.uniform(0.0, 0.1)]
        # self.sweeper_pos = [np.random.uniform(-0.2, -0.1), np.random.uniform(-0.05, 0)]
        # self.cubes_pos = [np.random.uniform(-0.2, 0.2), np.random.uniform(0.2, 0.3)]
        
        # self.cubes_pos = [0.2, 0.3]
        # self.sweeper_pos = [-0.1, -0.025]




        ################# for collecting expert gripper data to fit ar dist ######################
        self.sweeper_pos = [np.random.uniform(-0.0625, 0.0625), np.random.uniform(-0.05, 0.05)]
        if self.sweeper_pos[0] < 0:
            self.cubes_pos = [np.random.uniform(self.sweeper_pos[0] + 0.15, 0.3), np.random.uniform(-0.1, 0.3)]
        else:
            self.cubes_pos = [np.random.uniform(-0.3, self.sweeper_pos[0] - 0.15), np.random.uniform(-0.1, 0.3)]




        ################# for testing slippage on 12/24 ##################
        # self.cubes_pos = [0.12383749802226976, 0.1021068223998294]
        # self.sweeper_pos = [-0.057314282267020115, -0.004435712340069557]





        # print(self.cubes_pos)
        # print(self.sweeper_pos)
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
