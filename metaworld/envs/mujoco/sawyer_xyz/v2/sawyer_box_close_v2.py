import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os
import glob,random

class SawyerBoxCloseEnvV2(SawyerXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        main_file = 'sawyer_box.xml'
        
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

        obj_low   = (main_env_pos, 0.5, 0.02)
        obj_high  = (main_env_pos, 0.55, 0.02)
        goal_low  = (main_env_pos-0.05, 0.7, 0.133)
        goal_high = (main_env_pos+0.05, 0.8, 0.133)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.55, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.75, 0.133])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._target_to_obj_init = None

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return full_v2_path_for(os.path.join('sawyer_xyz_multi',self.file_name))
        return full_v2_path_for('sawyer_xyz/sawyer_box.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(success),
            'near_object': reward_ready,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_success,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('BoxHandleGeom')

    def _get_pos_objects(self):
        return self.get_body_com('top_link')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('top_link')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        box_height = self.get_body_com('boxbody')[2]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.25:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self._target_pos = goal_pos[-3:]

        self.sim.model.body_pos[self.model.body_name2id('boxbody')] = np.concatenate((self._target_pos[:2], [box_height]))
        self._set_obj_xyz(self.obj_init_pos)

        return self._get_obs()

    @staticmethod
    def _reward_grab_effort(actions):
        return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _reward_quat(obs):
        # Ideal upright lid has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error/0.2, 0.0)

    @staticmethod
    def _reward_pos(obs, target_pos):
        hand = obs[:3]
        lid = obs[4:7] + np.array([.0, .0, .02])

        threshold = 0.02
        # floor is a 3D funnel centered on the lid's handle
        radius = np.linalg.norm(hand[:2] - lid[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )
        # grab the lid's handle
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - lid),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_lift = reward_utils.hamacher_product(above_floor, in_place)

        # now actually put the lid on the box
        pos_error = target_pos - lid
        error_scale = np.array([1., 1., 3.])  # Emphasize Z error
        a = 0.2  # Relative importance of just *trying* to lift the lid at all
        b = 0.8  # Relative importance of placing the lid on the box
        lifted = a * float(lid[2] > 0.04) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * error_scale),
            bounds=(0, 0.05),
            margin=0.25,
            sigmoid='long_tail',
        )

        return ready_to_lift, lifted

    def compute_reward(self, actions, obs):
        reward_grab = SawyerBoxCloseEnvV2._reward_grab_effort(actions)
        reward_quat = SawyerBoxCloseEnvV2._reward_quat(obs)
        reward_steps = SawyerBoxCloseEnvV2._reward_pos(obs, self._target_pos)

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_grab, reward_steps[0]),
            8.0 * reward_steps[1],
        ))

        # Override reward on success
        success = np.linalg.norm(obs[4:7] - self._target_pos) < 0.08
        if success:
            reward = 10.0

        # STRONG emphasis on proper lid orientation to prevent reward hacking
        # (otherwise agent learns to kick-flip the lid onto the box)
        reward *= reward_quat

        return (
            reward,
            reward_grab,
            *reward_steps,
            success,
        )
