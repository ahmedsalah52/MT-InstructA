import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os
import glob,random

class SawyerDoorLockEnvV2(SawyerXYZEnv):

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
            
            self.file_order  = random.choice(range(len(env_txt_lines)))
            
            self.file_name = env_txt_lines[self.file_order]
            main_env_pos = float(self.file_name.split(',')[1])        
        self.x_shift = main_env_pos

        obj_low = (main_env_pos, 0.8, 0.15)
        obj_high = (main_env_pos, 0.85, 0.15)

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

        goal_low = self.hand_low
        goal_high = self.hand_high

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
            ('goal_lock', self._target_pos),
            ('goal_unlock', np.array([10., 10., 10.]))
        ]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        return self._get_site_pos('lockStartLock')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('door_link')

    def reset_model(self):
        self._reset_hand()
        door_pos = self.init_config['obj_init_pos']

        if self.random_init:
            door_pos = self._get_state_rand_vec()

        self.sim.model.body_pos[self.model.body_name2id('door')] = door_pos
        for _ in range(self.frame_skip):
            self.sim.step()

        self.obj_init_pos = self.get_body_com('lock_link')
        self._target_pos = self.obj_init_pos + np.array([.0, -.04, -.1])

        return self._get_obs()

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.get_body_com('leftpad')

        scale = np.array([0.25, 1., 0.5])
        tcp_to_obj = np.linalg.norm((obj - tcp) * scale)
        tcp_to_obj_init = np.linalg.norm((obj - self.init_left_pad) * scale)

        obj_to_target = abs(self._target_pos[2] - obj[2])

        tcp_opened = max(obs[3], 0.0)
        near_lock = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.01),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        lock_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._lock_length,
            sigmoid='long_tail',
        )

        reward = 2 * reward_utils.hamacher_product(tcp_opened, near_lock)
        reward += 8 * lock_pressed

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_lock,
            lock_pressed
        )
