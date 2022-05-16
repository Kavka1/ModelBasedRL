from typing import List, Tuple, Dict
import numpy as np
from gym.envs.mujoco.walker2d import Walker2dEnv


MISSINGABLE_JOINT = ['thigh', 'leg', 'foot']


class Missing_Joint_Vel_Walker(Walker2dEnv):
    def __init__(self, missing_joint: List) -> None:
        self.episode_length = 1000
        self.episode_step = 0
        for joint in missing_joint:
            assert joint in MISSINGABLE_JOINT, f"Invalid missing joint {joint}"
        self.missing_joint = missing_joint

        super().__init__()

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        feasible_q_vel = self._drop_infeasible_jnt_vel(qvel)
        
        return np.concatenate([qpos[1:], np.clip(feasible_q_vel, -10, 10)]).ravel()

    def step(self, action: np.array) -> Tuple:
        obs, r, done, info = super().step(action)
        if self.episode_step >= self.episode_length:
            done = True
        self.episode_step += 1
        return obs, r, done, info

    def reset(self) -> np.array:
        self.episode_step = 0
        return super().reset()
 
    def _unconcat_qpos_qvel(self, obs: np.array) -> Tuple:
        """
        Factorize the observation to qpos and qvel
        """
        dim_qpos = len(self.sim.data.qpos)
        qpos = obs[:dim_qpos-1]
        qvel = obs[dim_qpos-1:]
        return qpos, qvel

    def _process_obs(self, obs: np.array) -> np.array:
        """
        Process the original obs to missing info version
        """
        qpos, qvel = self._unconcat_qpos_qvel(obs)
        qvel = self._drop_infeasible_jnt_vel(qvel)
        return np.concatenate([qpos, qvel], axis=-1)

    def _drop_infeasible_jnt_vel(self, qvel: np.array) -> np.array:
        """
        Lack some observation info
        Original Obs: 
            [
                qpos: 
                    [0:2]: x, y, angle for generalized coordinate 1
                    [3:5]: x, y, angle for generalized coordinate 2
                    [6:8]: x, y, angle for generalized coordinate 3
                qvel:
                    velocity for 9 joints:
                        rootx, rooty, rootz,
                        thigh_joint, leg_joint, foot_joint,
                        thigh_left_joint, leg_left_joint, foot_joint
            ]
        """
        feasible_q_vel = np.copy(qvel)
        thigh_joint_index, leg_joint_index, foot_joint_index = [3,6], [4,7], [5,8]

        if 'thigh' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, thigh_joint_index)
            leg_joint_index = [leg_joint_index[i] - i - 1 for i in range(2)]
            foot_joint_index = [foot_joint_index[i] - i - 1 for i in range(2)]
        if 'leg' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, leg_joint_index)
            foot_joint_index = [foot_joint_index[i] - i - 1 for i in range(2)]
        if 'foot' in self.missing_joint:
            feasible_q_vel = np.delete(feasible_q_vel, foot_joint_index)

        return feasible_q_vel