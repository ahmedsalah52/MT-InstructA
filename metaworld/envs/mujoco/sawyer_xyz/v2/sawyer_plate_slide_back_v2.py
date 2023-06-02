import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os
import glob,random

class SawyerPlateSlideBackEnvV2(SawyerXYZEnv):

    def __init__(self):

        
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        main_file = 'sawyer_plate_slide.xml'
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

        goal_low = (main_env_pos, 0.6, 0.015)
        goal_high = (main_env_pos, 0.6, 0.015)
        obj_low = (main_env_pos, 0.85, 0.)
        obj_high = (main_env_pos, 0.85, 0.)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0., 0.85, 0.], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0., 0.6, 0.015])
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
        return full_v2_path_for(os.path.join('sawyer_xyz_multi',self.file_name))
        return full_v2_path_for('sawyer_xyz/sawyer_plate_slide.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': 0.0,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward
        }
        return reward, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('puck')

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('puck')).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:11] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = self.init_config['obj_init_pos']
        self._target_pos = self.goal.copy()

        if self.random_init:
            rand_vec = self._get_state_rand_vec()
            self.obj_init_pos = rand_vec[:3]
            self._target_pos = rand_vec[3:]

        self.sim.model.body_pos[self.model.body_name2id('puck_goal')] = self.obj_init_pos
        self._set_obj_xyz(np.array([0, 0.15]))

        return self._get_obs()

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(self.init_tcp - self.obj_init_pos)
        object_grasped = reward_utils.tolerance(tcp_to_obj,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=obj_grasped_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        reward = 1.5 * object_grasped

        if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
            reward = 2 + (7 * in_place)

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return [
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place
        ]
