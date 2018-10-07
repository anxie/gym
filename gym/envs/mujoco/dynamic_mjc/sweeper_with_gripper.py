from gym.envs.mujoco.dynamic_mjc.model_builder import MJCModel
import numpy as np
import os

def sweeper_with_gripper(num_cubes=3, cubes_pos=[0.0, 0.3], sweeper_pos=[0.0, 0.0]):
    mjcmodel = MJCModel('sweeper')
    mjcmodel.root.compiler(inertiafromgeom="auto",
        angle="radian",
        coordinate="local", 
        eulerseq="XYZ", 
        texturedir=os.path.dirname(os.path.realpath(__file__)) + "/../assets/textures")

    mjcmodel.root.size(njmax=6000, nconmax=6000)
    mjcmodel.root.option(timestep="0.005", gravity="0 0 -9.81", iterations="50", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(limited="false", damping="1")
    default.geom(contype="1", conaffinity="1", condim="3", friction=".5 .1 .1", density="1000", margin="0.002")

    worldbody = mjcmodel.root.worldbody()

    worldbody.camera(name="maincam", mode="fixed", fovy="32", euler="0.7 0 0", pos="0 -1.1 1.3")
    worldbody.camera(name="overheadcam", mode= "fixed", pos="0. 0.7 1.0", euler="-0.55 0.0 0.0")

    gripper = worldbody.body(name="gripper", pos=[0,0,0.25])

    gripper.inertial(pos="0 0 0", mass="1", diaginertia="16.667 16.667 16.667")
    gripper.geom(type="box", size=".1 .03 .03", rgba="0.8 0.8 0.8 1", contype="0", conaffinity="0")

    gripper.joint(name="slide_x", type="slide", pos="0 0 0", axis="1 0 0", limited="true", range="-0.5 0.5", armature="0", damping="30", stiffness="0")
    gripper.joint(name="slide_y", type="slide", pos="0 0 0", axis="0 1 0", limited="true", range="-0.5 0.5", armature="0", damping="30", stiffness="0")
    gripper.joint(name="slide_z", type="slide", pos="0 0 0", axis="0 0 1",  limited="true", range="-0.08 0.15")
    gripper.joint(name="hinge_z", type="hinge", pos="0 0 0", axis="0 0 1", limited="true", range="-6.28 6.28", damping="30")

    fingers = gripper.body(name="fingers", pos=[0,0,0])
    finger_1 = fingers.body(name="finger_1", pos=[-0.08,0.0,-0.1])
    finger_1.joint(name="j_finger1", type="slide", pos="0 0 0", axis="1 0 0",  limited="true", range="0.0 0.0615")
    finger_1.geom(condim="6", contype="1", conaffinity="1", type="box", size=".01 .02 .1", rgba="0.8 0.8 0.8 1",  mass="0.08")
    finger_1.site(name="finger1_surf", pos="0.01 0 0", size=".0025 .0190 .095", type="box", rgba="0.0 1.0 0.0 0")

    finger_2 = fingers.body(name="finger_2", pos=[0.08,0.0,-0.1])
    finger_2.joint(name="j_finger2", type="slide", pos="0 0 0", axis="1 0 0")
    finger_2.geom(condim="6", contype="1", conaffinity="1",  type="box", size=".01 .02 .1", rgba="0.8 0.8 0.8 1", mass="0.08")
    finger_2.site(name="finger2_surf", pos="-0.01 0 0", size=".0025 .0190 .095", type="box", rgba="1.0 0.0 0.0 0")

    sweeper_handle = worldbody.body(name="sweeper_handle", pos=sweeper_pos + [0])
    sweeper_handle.joint(name="sweeper_free", type="free")
    start = str(sweeper_pos[0]) + ' ' + str(sweeper_pos[1] - 0.1) + ' 0.01'
    end = str(sweeper_pos[0]) + ' ' + str(sweeper_pos[1] + 0.1) + ' 0.01'
    sweeper_handle.geom(type="capsule", size="0.02", rgba="0 0 1 1", contype="1", conaffinity="1", fromto=start + ' ' + end)
    sweeper_flat = sweeper_handle.body(name="sweeper_flat", pos=[sweeper_pos[0],sweeper_pos[1]+0.1,0])
    sweeper_flat.site(name="tip_l", pos="-0.06 0 0", size="0.01")
    sweeper_flat.site(name="tip_r", pos="0.06 0 0", size="0.01")
    sweeper_flat.geom(type="box", pos="0 0 0.03", size="0.12 0.02 0.06", rgba="0 0 1 1", contype="1", conaffinity="1")

    cubes = []
    pos_list = []
    for i in range(num_cubes):
        new_pos = list(cubes_pos)
        # hard code for now
        if i == 0:
            new_pos[1] -= 0.05 + np.random.normal(loc=0.0, scale=0.01)
        elif i == 1:
            new_pos[0] -= 0.05 + np.random.normal(loc=0.0, scale=0.01)
        elif i == 2:
            new_pos[0] += 0.05 + np.random.normal(loc=0.0, scale=0.01)
        new_pos.append(0.0)
        cubes.append(worldbody.body(name="cube_{}".format(i), pos=list(new_pos)))
        cubes[i].joint(type="free")
        cubes[i].geom(type="box", size="0.03 0.03 0.03", rgba="1.0 0.0 0.0 1.0", contype="1", conaffinity="1")

    container = worldbody.body(name="container", pos=[0,0,-0.05])
    border_front = container.body(name="border_front", pos="0 -.5  0")
    border_front.geom(type="box", size=".5 .01 .1", rgba="0 .1 .9 .3")
    border_rear = container.body(name="border_rear", pos="0 .5  0")
    border_rear.geom(type="box", size=".5 .01 .1", rgba="0 .1 .9 .3")
    border_right = container.body(name="border_right", pos=".5 0. 0")
    border_right.geom(type="box", size=".01  .5 .1", rgba="0 .1 .9 .3")
    border_left = container.body(name="border_left", pos="-.5 0. 0")
    border_left.geom(type="box", size=".01  .5 .1", rgba="0 .1 .9 .3")
    table = container.body(name="table", pos="0 0 -.01")
    table.geom(type="box", size=".5  .5 .01", rgba="1 1 1 1", contype="7", conaffinity="7")
        
    light = worldbody.body(name="light", pos=[0,0,1])
    light.light(name="light0", mode="fixed", directional="false", active="true", castshadow="false")

    actuator = mjcmodel.root.actuator()
    actuator.position(joint="slide_x", ctrllimited="false", kp="200")
    actuator.position(joint="slide_y", ctrllimited="false", kp="200")
    actuator.general(joint="slide_z", gaintype="fixed", dyntype="none", dynprm="1 0 0",
                    gainprm ="100 0 0", biastype="affine", biasprm="10 -100 -4")
    actuator.position(joint="hinge_z", ctrllimited="false", kp="300")
    actuator.position(joint="j_finger1", ctrllimited="true", ctrlrange="0 0.1", kp="300")

    equality = mjcmodel.root.equality()
    equality.joint(joint1="j_finger1", joint2="j_finger2", polycoef="0 -1 0 0 0")

    asset = mjcmodel.root.asset()

    return mjcmodel
