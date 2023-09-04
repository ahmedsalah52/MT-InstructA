import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os,random


class SawyerPegUnplugSideEnvV2(SawyerXYZEnv):
    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        main_file = 'sawyer_peg_unplug_side.xml'
        generate = True
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

        obj_low = (-0.25, 0.6, -0.001)
        obj_high = (-0.15, 0.8, 0.001)
        goal_low = obj_low + np.array([.194, .0, .131])
        goal_high = obj_high + np.array([.194, .0, .131])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([-0.225, 0.6, 0.05]),
            'hand_init_pos': np.array(((0, 0.6, 0.2))),
        }
        self.goal = np.array([-0.225, 0.6, 0.0])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_peg_unplug_side.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward, grasp_success = (
            self.compute_reward(action, obs))
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self._get_site_pos('pegEnd')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('plug1')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos
        qpos[12:16] = np.array([1., .0, .0, .0])
        qvel[9:12] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        pos_box = self._get_state_rand_vec() if self.random_init else self.goal
        self.sim.model.body_pos[self.model.body_name2id('box')] = pos_box

        pos_plug = pos_box + np.array([.044, .0, .131])
        self._set_obj_xyz(pos_plug)
        self.obj_init_pos = self._get_site_pos('pegEnd')

        self._target_pos = pos_plug + np.array([.15, .0, .0])

        return self._get_obs()

    def compute_reward(self, action, obs):
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos
        tcp_to_obj = np.linalg.norm(obj - tcp)
        obj_to_target = np.linalg.norm(obj - target)
        pad_success_margin = 0.05
        object_reach_radius = 0.01
        x_z_margin = 0.005
        obj_radius = 0.025

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_margin,
            xz_thresh=x_z_margin,
            desired_gripper_effort=0.8,
            high_density=True)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)
        
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.05),
            margin=in_place_margin,
            sigmoid='long_tail',
        )
        grasp_success = (tcp_opened > 0.5 and 
            (obj[0] - self.obj_init_pos[0] > 0.015))
        
        
        reward = 2 * object_grasped

        if grasp_success and tcp_to_obj < 0.035:
            reward = 1 + 2 * object_grasped + 5 * in_place

        if obj_to_target <= 0.05:
            reward = 10.

        return reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place, float(
            grasp_success)
