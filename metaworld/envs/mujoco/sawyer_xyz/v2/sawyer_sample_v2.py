import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for,full_mix_path_for 
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
import random
from metaworld.envs.build_random_envs import build_env , multi_object_man
import os

class SawyerSampleEnvV2(SawyerXYZEnv):
    def __init__(self):
        poses_list = [0,1,2]
        dx_idx = poses_list.pop(random.randrange(len(poses_list)))
        self.dx_dict = {0:-0.4 , 1:0 , 2:0.4}
        dx = self.dx_dict[dx_idx]
        
        
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low  =  (dx+0.01, 0.85, 0.115)
        obj_high =  (dx-0.01, 0.9, 0.115)

        secondary_poses = [self.dx_dict[poses_list[0]],self.dx_dict[poses_list[1]]]
        print('main ',dx)
        print('secondary ',secondary_poses)

        main_env_name = 'sawyer_button_press.xml'
      
        #main_file = 'metaworld/envs/assets_v2/sawyer_xyz/sawyer_button_press.xml'
        
        multi_object = multi_object_man(init_file_name=main_env_name)
        main_envs_dir = 'metaworld/envs/assets_v2/sawyer_xyz/'
        delta_x_objects = 0.35
        xml_files = os.listdir(main_envs_dir)
        xml_files.remove(main_env_name)
        st_sec_file = random.sample(xml_files,1)[0]
        xml_files.remove(st_sec_file)
        nd_sec_file = random.sample(xml_files,1)[0]
        multi_object.get_new_env([st_sec_file,nd_sec_file] , dx_idx,poses_list)
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
        self.init_config = {
            'obj_init_pos': np.array([0., 0.9, 0.115], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.78, 0.12])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for(os.path.join('sawyer_xyz_multi',self.file_name))
        #return full_mix_path_for('metaworld/envs/assets_v2/sawyer_xyz/testing.xml',self.secondary_poses)

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
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('btnGeom')

    def _get_pos_objects(self):
        return self.get_body_com('button') + np.array([.0, -.193, .0])

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('button')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos

        self.sim.model.body_pos[
            self.model.body_name2id('box')] = self.obj_init_pos
        self._set_obj_xyz(0)
        self._target_pos = self._get_site_pos('hole')

        self._obj_to_target_init = abs(
            self._target_pos[1] - self._get_site_pos('buttonStart')[1]
        )

        return self._get_obs()

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[1] - obj[1])

        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._obj_to_target_init,
            sigmoid='long_tail',
        )

        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_button,
            button_pressed
        )
