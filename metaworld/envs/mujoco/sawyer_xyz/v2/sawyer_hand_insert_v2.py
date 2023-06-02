import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os , random


class SawyerHandInsertEnvV2(SawyerXYZEnv):
    TARGET_RADIUS = 0.05

    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        main_file = 'sawyer_table_with_hole.xml'
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

        obj_low = (-0.1, 0.6, 0.05)
        obj_high = (0.1, 0.7, 0.05)
        goal_low = (-0.04, 0.8, -0.0201)
        goal_high = (0.04, 0.88, -0.0199)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.05]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0., 0.84, -0.08], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_table_with_hole.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= 0.05),
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_success': float(
                self.touching_main_object and
                (tcp_open > 0) and
                (obj[2] - 0.02 > self.obj_init_pos[2])
            ),
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('obj')

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]

        # if self.random_init:
        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.15:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
        self._target_pos = goal_pos[-3:]

        self._set_obj_xyz(self.obj_init_pos)
        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]

        target_to_obj = np.linalg.norm(obj - self._target_pos)
        target_to_obj_init = np.linalg.norm(self.obj_init_pos - self._target_pos)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1. + 7. * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )
