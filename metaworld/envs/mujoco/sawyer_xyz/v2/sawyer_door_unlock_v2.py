import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os

import glob,random

class SawyerDoorUnlockEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        main_file = 'sawyer_door_lock.xml'
        generate = False
        if generate:
            mjcfs_dir = 'metaworld/envs/assets_v2/sawyer_xyz_multi/mjcfs/'+main_file.split('.')[0]
            if not os.path.isdir(mjcfs_dir):
                os.system('mkdir '+mjcfs_dir)
            multi_object = multi_object_man(init_file_name=main_file)

            main_envs_dir = 'metaworld/envs/assets_v2/sawyer_xyz/'
            xml_files = os.listdir(main_envs_dir)
            poses_list = [0,1,2]
            for pos in [0,1,2]:
                poses_list = [0,1,2]
                dx_idx = poses_list.pop(pos)
                for st_sec_file in xml_files:
                    if main_file == st_sec_file: pass
                    for nd_sec_file in xml_files:
                        if nd_sec_file == st_sec_file or nd_sec_file == main_file: pass  
                        try:
                            multi_object.get_new_env([st_sec_file,nd_sec_file] , dx_idx,poses_list)
                            self.file_name = multi_object.get_file_name()
                            super().__init__(
                                self.model_name,
                                hand_low=hand_low,
                                hand_high=hand_high,
                            )
                            multi_object.multi_env_loaded()
                            
                        except:
                            print('failed to load:',self.file_name)
                            multi_object.multi_env_not_loaded()

        else:
            env_txt_file = open('metaworld/all_envs/'+main_file.split('.')[0]+'.txt','r')
            env_txt_lines = env_txt_file.read().split('\n')
            
            env_txt_line = random.choice(env_txt_lines)
            
            self.file_name = env_txt_line
            main_env_pos = float(self.file_name.split(',')[1])        

        obj_low = (main_env_pos, 0.8, 0.15)
        obj_high = (main_env_pos, 0.85, 0.15)
        goal_low = (main_env_pos, 0.64, 0.2100)
        goal_high = (main_env_pos+0.2, 0.7, 0.2111)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.85, 0.15]),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.85, 0.1])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._lock_length = 0.1

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for(os.path.join('sawyer_xyz_multi',self.file_name))
        return full_v2_path_for('sawyer_xyz/sawyer_door_lock.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_open > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [
            ('goal_unlock', self._target_pos),
            ('goal_lock', np.array([10., 10., 10.]))
        ]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        return self._get_site_pos('lockStartUnlock')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('door_link')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        door_pos = self.init_config['obj_init_pos']

        if self.random_init:
            door_pos = self._get_state_rand_vec()

        self.sim.model.body_pos[self.model.body_name2id('door')] = door_pos
        self._set_obj_xyz(1.5708)

        self.obj_init_pos = self.get_body_com('lock_link')
        self._target_pos = self.obj_init_pos + np.array([.1, -.04, .0])

        return self._get_obs()

    def compute_reward(self, action, obs):
        del action
        gripper = obs[:3]
        lock = obs[4:7]

        # Add offset to track gripper's shoulder, rather than fingers
        offset = np.array([.0, .055, .07])

        scale = np.array([0.25, 1., 0.5])
        shoulder_to_lock = (gripper + offset - lock) * scale
        shoulder_to_lock_init = (
            self.init_tcp + offset - self.obj_init_pos
        ) * scale

        # This `ready_to_push` reward should be a *hint* for the agent, not an
        # end in itself. Make sure to devalue it compared to the value of
        # actually unlocking the lock
        ready_to_push = reward_utils.tolerance(
            np.linalg.norm(shoulder_to_lock),
            bounds=(0, 0.02),
            margin=np.linalg.norm(shoulder_to_lock_init),
            sigmoid='long_tail',
        )

        obj_to_target = abs(self._target_pos[0] - lock[0])
        pushed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._lock_length,
            sigmoid='long_tail',
        )

        reward = 2 * ready_to_push + 8 * pushed

        return (
            reward,
            np.linalg.norm(shoulder_to_lock),
            obs[3],
            obj_to_target,
            ready_to_push,
            pushed
        )
