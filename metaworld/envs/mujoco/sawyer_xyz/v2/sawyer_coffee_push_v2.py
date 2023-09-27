import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


from metaworld.envs.build_random_envs import Multi_task_env
import os
import glob,random

class SawyerCoffeePushEnvV2(SawyerXYZEnv):
    def __init__(self,main_pos_index=None , task_variant = None):
        Multi_task_env.__init__(self)
        self.main_pos_index = main_pos_index
        self.task_variant = task_variant

        hand_low = (-0.7, 0.2, 0.05)
        hand_high = (0.7, 1, 0.5)
        main_file = 'sawyer_coffee.xml'
        
        self.generate_env(main_file,main_pos_index,task_variant)
        obj_low =  (self.task_offsets_min[0], self.task_offsets_min[1] - 0.4, 0.001)
        obj_high = (self.task_offsets_max[0], self.task_offsets_max[1] - 0.4, 0.001)

        goal_low  = (self.task_offsets_min[0], self.task_offsets_min[1] - 0.25,  0.001)
        goal_high = (self.task_offsets_max[0], self.task_offsets_max[1] - 0.25,  0.001)
    
        SawyerXYZEnv.__init__(
            self,
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )
        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': (np.array(obj_low)+np.array(obj_high))/2,
            'hand_init_pos': np.array(self.hand_init_pos_, dtype=np.float32),
        }
        self.goal = (np.array(goal_low)+np.array(goal_high))/2
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
        return full_v2_path_for('sawyer_xyz/sawyer_coffee.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place = self.compute_reward(
            action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_open > 0))

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,

        }

        return reward, info

    @property
    def _target_site_config(self):
        return [('coffee_goal', self._target_pos)]

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('mug')
        ).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        pos_mug_init = self.init_config['obj_init_pos']
        pos_mug_goal = self.goal

        if self.random_init:
            pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2]) < 0.15:
                pos_mug_init, pos_mug_goal = np.split(
                    self._get_state_rand_vec(),
                    2
                )

        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init

        pos_machine = pos_mug_goal + np.array([.0, .22, .0])
        self.sim.model.body_pos[self.model.body_name2id(
            'coffee_machine'
        )] = pos_machine

        self._target_pos = pos_mug_goal
        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        target = self._target_pos.copy()

        # Emphasize X and Y errors
        scale = np.array([2., 2., 1.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, 0.05),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.04,
            obj_radius=0.02,
            pad_success_thresh=0.05,
            xz_thresh=0.05,
            desired_gripper_effort=0.7,
            medium_density=True
        )

        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.04 and tcp_opened > 0:
            reward += 1. + 5. * in_place
        if target_to_obj < 0.05:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            np.linalg.norm(obj - target),  # recompute to avoid `scale` above
            object_grasped,
            in_place
        )
