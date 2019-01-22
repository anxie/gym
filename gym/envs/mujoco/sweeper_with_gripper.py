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

        sweeper_pos = self.get_body_com("sweeper_handle")
        cubes_pos = [self.get_body_com("cube_0"), self.get_body_com("cube_1"), self.get_body_com("cube_2")]

        projected_sweeper_pos = np.squeeze(np.round(self.project_point(sweeper_pos))).astype(np.uint8)
        projected_cubes_pos = [np.squeeze(np.round(self.project_point(pos))).astype(np.uint8) for pos in cubes_pos]

        return obs, 0, done, dict(video_frames=video_frames,
            sweeper_pos=sweeper_pos,
            cubes_pos=cubes_pos,
            projected_sweeper_pos=projected_sweeper_pos,
            projected_cubes_pos=projected_cubes_pos)

    def project_point(self, point):
        model_matrix = np.zeros((4, 4))
        model_matrix[:3, :3] = self.sim.data.get_camera_xmat(self.camera_name).T
        model_matrix[-1, -1] = 1

        fovy_radians = np.deg2rad(self.sim.model.cam_fovy[self.sim.model.camera_name2id(self.camera_name)])
        uh = 1. / np.tan(fovy_radians / 2)
        uw = uh / (self.im_width / self.im_height)
        extent = self.sim.model.stat.extent
        far, near = self.sim.model.vis.map.zfar * extent, self.sim.model.vis.map.znear * extent
        view_matrix = np.array([[uw, 0., 0., 0.],                        # matrix definition from
                                [0., uh, 0., 0.],                        # https://stackoverflow.com/questions/18404890/how-to-build-perspective-projection-matrix-no-api
                                [0., 0., far / (far - near), -1.],
                                [0., 0., -2*far*near/(far - near), 0.]]) # Note Mujoco doubles this quantity

        MVP_matrix = view_matrix.dot(model_matrix)
        world_coord = np.ones((4, 1))
        world_coord[:3, 0] = point - self.sim.data.get_camera_xpos(self.camera_name)

        clip = MVP_matrix.dot(world_coord)
        ndc = clip[:3] / clip[3]  # everything should now be in -1 to 1!!
        col, row = (ndc[0] + 1) * self.im_width / 2, (-ndc[1] + 1) * self.im_height / 2

        return self.im_height - row, col                 # rendering flipped around in height

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
