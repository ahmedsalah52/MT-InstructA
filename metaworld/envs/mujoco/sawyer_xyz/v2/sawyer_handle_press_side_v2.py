import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import build_env , multi_object_man
import os , random


class SawyerHandlePressSideEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the end effector's wrist has a
        nub that got caught on the box before pushing the handle all the way
        down. There are a number of ways to fix this, e.g. moving box to right
        sie of table, extending handle's length, decreasing handle's damping,
        or moving the goal position slightly upward. I just the last one.
    Changelog from V1 to V2:
        - (8/05/20) Updated to new XML
        - (6/30/20) Increased goal's Z coordinate by 0.01 in XML
    """
    TARGET_RADIUS = 0.02

    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        main_file = 'sawyer_handle_press_sideways.xml'
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

        obj_low = (-0.35, 0.65, -0.001)
        obj_high = (-0.25, 0.75, +0.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([-0.3, 0.7, 0.0]),
            'hand_init_pos': np.array((0, 0.6, 0.2),),
        }
        self.goal = np.array([-0.2, 0.7, 0.14])
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
        return full_v2_path_for('sawyer_xyz/sawyer_handle_press_sideways.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward,
        tcp_to_obj,
        _,
        target_to_obj,
        object_grasped,
        in_place) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= self.TARGET_RADIUS),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': 1.,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self._get_site_pos('handleStart')

    def _get_quat_objects(self):
        return np.zeros(4)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = (self._get_state_rand_vec()
                             if self.random_init
                             else self.init_config['obj_init_pos'])

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._set_obj_xyz(-0.001)
        self._target_pos = self._get_site_pos('goalPress')
        self._handle_init_pos = self._get_pos_objects()

        return self._get_obs()

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[4:7]
        obj = self._get_pos_objects()
        tcp = self.tcp_center
        target = self._target_pos.copy()
        
        target_to_obj = (obj[2] - target[2])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self._handle_init_pos[2] - target[2])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self._handle_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        reward = 1 if target_to_obj <= self.TARGET_RADIUS else reward
        reward *= 10
        return (reward,
               tcp_to_obj,
               tcp_opened,
               target_to_obj,
               object_grasped,
               in_place)