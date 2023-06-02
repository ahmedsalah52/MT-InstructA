import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_v2 import SawyerDoorEnvV2
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os
import glob,random


class SawyerDoorCloseEnvV2(SawyerDoorEnvV2):
    def __init__(self):

        
        main_file = 'sawyer_door_pull.xml'
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

        goal_low = (main_env_pos+.2, 0.65, 0.1499)
        goal_high = (main_env_pos+.3, 0.75, 0.1501)
        

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0.1, 0.95, 0.15], dtype=np.float32),
            'hand_init_pos': np.array([-0.5, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.2, 0.8, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('handle')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy() + np.array([0.2, -0.2, 0.])
            self._target_pos = goal_pos

        self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos

        # keep the door open after resetting initial positions
        self._set_obj_xyz(-1.5708)

        return self._get_obs()

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, obj_to_target, in_place = self.compute_reward(action, obs)
        info = {
            'obj_to_target': obj_to_target,
            'in_place_reward': in_place,
            'success': float(obj_to_target <= 0.08),
            'near_object': 0.,
            'grasp_success': 1.,
            'grasp_reward': 1.,
            'unscaled_reward': reward,
        }
        return reward, info

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        target = self._target_pos

        tcp_to_target = np.linalg.norm(tcp - target)
        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_to_target = np.linalg.norm(obj - target)

        in_place_margin = np.linalg.norm(self.obj_init_pos - target)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='gaussian',)

        hand_margin = np.linalg.norm(self.hand_init_pos - obj) + 0.1
        hand_in_place = reward_utils.tolerance(tcp_to_target,
                                    bounds=(0, 0.25*_TARGET_RADIUS),
                                    margin=hand_margin,
                                    sigmoid='gaussian',)

        reward = 3 * hand_in_place + 6 * in_place

        if obj_to_target < _TARGET_RADIUS:
            reward = 10

        return [reward, obj_to_target, hand_in_place]
