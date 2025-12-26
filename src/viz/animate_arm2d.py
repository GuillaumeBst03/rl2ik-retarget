import numpy as np

from src.robotics.arm2d_kinematics import Arm2DParams, forward_kinematics, joint_positions

if __name__ == "__main__":
    params = Arm2DParams(link_lengths=np.array([0.5, 0.4, 0.3, 0.2]))
    q = np.array([0.2, -0.5, 0.9, 0.1])

    ee = forward_kinematics(params, q)
    pts = joint_positions(params, q)

    print("End-effector:", ee)
    print("All points:\n", pts)