import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

from metaworld.envs.build_random_envs import Multi_task_env
import os



class SawyerPlateSlideEnvV2(SawyerXYZEnv,Multi_task_env):

    OBJ_RADIUS = 0.04

    def __init__(self,main_pos_index=None , task_variant = None):
        Multi_task_env.__init__(self)
        self.main_pos_index = main_pos_index
        self.task_variant = task_variant

        hand_low = (-0.6, 0.40, 0.05)
        hand_high = (0.6, 1, 0.5)
        main_file = 'sawyer_plate_slide.xml'
        self.generate_env(main_file,main_pos_index,task_variant)

        

        obj_low   = (self.task_offsets_min[0]   , self.task_offsets_min[1] + 0.55, 0)
        obj_high  = (self.task_offsets_max[0]   , self.task_offsets_max[1] + 0.55, 0)
        goal_low  = (self.task_offsets_min[0]   , self.task_offsets_min[1] + 0.90, 0.)
        goal_high = (self.task_offsets_max[0]   , self.task_offsets_min[1] + 0.90, 0.)
        SawyerXYZEnv.__init__(
            self,
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos' : np.array([0.5, 0.6, 0.], dtype=np.float32),
            'hand_init_pos': np.array(self.hand_init_pos_, dtype=np.float32),
        }
        self.goal = np.array([0.5, 0.85, 0.02])
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
            'grasp_reward': object_grasped,
            'grasp_success': 0.0,
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
            self.init_tcp = self.tcp_center
            self.obj_init_pos = rand_vec[:3]
            self._target_pos = rand_vec[3:]

        self.sim.model.body_pos[
            self.model.body_name2id('puck_goal')] = self._target_pos
        self._set_obj_xyz(np.zeros(2))

        return self._get_obs()

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(self.init_tcp - self.obj_init_pos)

        object_grasped = reward_utils.tolerance(tcp_to_obj,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=obj_grasped_margin,
                                    sigmoid='long_tail',)

        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = 8 * in_place_and_object_grasped

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
