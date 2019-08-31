from gym.envs.mujoco.dynamic_mjc.model_builder import MJCModel
import numpy as np
import os

SHAPES = ["box",
          "cylinder",
          "sphere",
          "ellipsoid"]

COLORS = [[1, 0, 0, 1],     # red
          [0, 0, 1, 1],     # blue
          [0, 1, 0, 1],     # green
          [1, 0, 1, 1],     # purple
          [0, 1, 1, 1]]     # yellow

POSITIONS = [[0.45, -0.125, -0.325],
             [0.55, 0.125, -0.325]]


def gen_fixed_objects():
    N = 2
    shapes, sizes, densities, colors, positions = [], [], [], [], []
    color_indices = np.random.choice(2, 2, replace=False)
    for i in range(N):
        shapes.append(SHAPES[1])
        # sizes.append(np.asarray([0.05, 0.05, 0.05]))
        sizes.append(np.random.uniform(0.04, 0.06, 3))
        colors.append(COLORS[color_indices[i]])
        positions.append(POSITIONS[i] + np.append(np.random.uniform(low=-.075, high=.075, size=2), 0))
    return N, shapes, sizes, densities, colors, positions, color_indices


def gen_random_objects():
    N = np.random.randint(3, 6)
    shapes, sizes, densities, colors, positions = [], [], [], []
    for i in range(N):
        shapes.append(np.random.choice(SHAPES, p=[0.5, 0.2, 0.1, 0.2]))
        sizes.append(np.random.uniform(0.06, 0.08, 3))
        densities.append(np.random.uniform(500, 1500))
        colors.append(np.random.choice(COLORS))
        positions.append(np.append(np.random.uniform(-0.25, 0.25, 2), 0.0))
    return N, shapes, sizes, densities, colors, positions


def wrist():
    mjcmodel = MJCModel('wrist')
    mjcmodel.root.compiler(inertiafromgeom="true",
        angle="radian",
        coordinate="local", 
        eulerseq="XYZ")

    mjcmodel.root.size(njmax=6000, nconmax=6000)
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(limited="true", damping="1", armature="0.04")
    default.geom(contype="0", conaffinity="0", condim="1", friction=".8 .1 .1", density="10", margin="0.002")

    worldbody = mjcmodel.root.worldbody()
    # worldbody.camera(name="maincam", mode="fixed", pos="0.4 0.5 0.8", euler="-0.375 0.0 0.0")
    worldbody.camera(name="overheadcam", mode="fixed", pos="0.4 0.0 1.0", euler="0.0 0.0 0.0")

    wrist_base = worldbody.body(name="wrist_base", pos=[0,0,-0.325])
    wrist_base.geom(contype="1", conaffinity="1", condim="1", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", type="capsule", fromto="0 -0.1 0 0.0 +0.1 0", size="0.02")
    wrist_base.geom(contype="1", conaffinity="1", condim="1", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", type="capsule", fromto="0 -0.1 0 0.1 -0.1 0", size="0.02")
    wrist_base.geom(contype="1", conaffinity="1", condim="1", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", type="capsule", fromto="0 +0.1 0 0.1 +0.1 0", size="0.02")

    wrist_base.joint(name="slide_x", type="slide", pos="0 0 0", axis="1 0 0", range="-10.3213 10.3", damping="0.5")
    wrist_base.joint(name="slide_y", type="slide", pos="0 0 0", axis="0 1 0", range="-10.3213 10.3", damping="0.5")
    wrist_base.joint(name="hinge_z", type="hinge", pos="0 0 0", axis="0 0 1", range="-10.3213 10.3", damping="0.5")

    fingers = wrist_base.body(name="fingers", pos=[0,0,0])
    finger_1 = fingers.body(name="finger_1", pos=[0.1,-0.1,0])
    finger_1.geom(contype="1", conaffinity="1", condim="1", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", type="sphere", size=".01")
    finger_2 = fingers.body(name="finger_2", pos=[0.1,0.1,0])
    finger_2.geom(contype="1", conaffinity="1", condim="1", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", type="sphere", size=".01")

    N, shapes, sizes, densities, colors, positions, color_indices = gen_fixed_objects()
    objects = []
    for i in range(N):
        objects.append(worldbody.body(name="obj_{}".format(i), pos=positions[i]))
        objects[i].joint(name="obj_{}_slide_x".format(i), type="slide", pos="0 0 0", axis="1 0 0", range="-10.3213 10.3")
        objects[i].joint(name="obj_{}_slide_y".format(i), type="slide", pos="0 0 0", axis="0 1 0", range="-10.3213 10.3")
        objects[i].geom(type=shapes[i], size=sizes[i], rgba=colors[i], density="0.0005", solimp="1 1 0.0", solref="0.01 1", friction="1.5 0.005 0.001", contype="1", conaffinity="1")

    worldbody.geom(conaffinity="1", contype="1", condim="1", name="table", pos="0 0.5 -0.325", size="10 10 0.1", type="plane")
    worldbody.geom(name="wall1", conaffinity="0", contype="1", pos="0.4 -0.5 -0.275", size="0.5 0.001 0.06", mass="1000", type="box", rgba="0.9 0.6 0.4 1")
    worldbody.geom(name="wall2", conaffinity="0", contype="1", pos="0.4 0.5 -0.275", size="0.5 0.001 0.06", mass="1000", type="box", rgba="0.9 0.6 0.4 1")
    worldbody.geom(name="wall3", conaffinity="0", contype="1", pos="-0.1 0 -0.275", size="0.001 0.5 0.06", mass="1000", type="box", rgba="0.9 0.6 0.4 1")
    worldbody.geom(name="wall4", conaffinity="0", contype="1", pos="0.9 0 -0.275", size="0.001 0.5 0.06", mass="1000", type="box", rgba="0.9 0.6 0.4 1")
    worldbody.geom(rgba="0.9 0.6 0.4 1", type="capsule", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", fromto="-0.1 0.5 -0.325 -0.1 0.5 -0.225", size="0.005", conaffinity="0", contype="1")
    worldbody.geom(rgba="0.9 0.6 0.4 1", type="capsule", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", fromto="-0.1 -0.5 -0.325 -0.1 -0.5 -0.225", size="0.005", conaffinity="0", contype="1")
    worldbody.geom(rgba="0.9 0.6 0.4 1", type="capsule", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", fromto="0.9 0.5 -0.325 0.9 0.5 -0.225", size="0.005", conaffinity="0", contype="1")
    worldbody.geom(rgba="0.9 0.6 0.4 1", type="capsule", solimp="1 1 0", solref="0.01 1", friction="1.5 0.005 0.001", fromto="0.9 -0.5 -0.325 0.9 -0.5 -0.225", size="0.005", conaffinity="0", contype="1")

    light = worldbody.body(name="light", pos=[0.4,0,1.5])
    light.light(name="light0", dir="0 0 -1", castshadow="false")

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="slide_x", ctrllimited="true", ctrlrange="-2.0 2.0")
    actuator.motor(joint="slide_y", ctrllimited="true", ctrlrange="-2.0 2.0")
    actuator.motor(joint="hinge_z", ctrllimited="true", ctrlrange="-2.0 2.0")

    asset = mjcmodel.root.asset()

    return mjcmodel, {'shapes': shapes, 'colors': color_indices}
