import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.pybullet.utils import transformations

class CloseLoopPomdpBlockStackingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.pick_place_stage = 0
    # self.post_pose_reached = False
    # pos, rot, primitive
    self.current_target = None
    self.target_obj = None

  def getNextAction(self, target_obj_idx):
    if not self.env._isHolding():
      # pre-pick
      primative = constants.PICK_PRIMATIVE

      delta_height = 0.0

      # pre-pick
      if target_obj_idx in [0, 1]:
        delta_height = 0.1

      # actual pick
      if target_obj_idx in [2, 3]:
        delta_height = 0.0

      obj_idx = target_obj_idx

      if obj_idx >= 2:
        obj_idx -= 2

    else:
      if target_obj_idx >= 2:
        target_obj_idx -= 2
      obj_idx = 1 - target_obj_idx
      primative = constants.PLACE_PRIMATIVE

      # pre-place
      if target_obj_idx in [0, 1]:
        delta_height = 0.1

      # actual place
      if target_obj_idx in [2, 3]:
        delta_height = 0.0

      delta_height = 0.1

    block_pos = self.env.objects[obj_idx].getPosition()
    object_rot = list(transformations.euler_from_quaternion(self.env.objects[obj_idx].getRotation()))

    block_pos[2] += delta_height

    gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
    while object_rot[2] - gripper_rz > np.pi / 4:
      object_rot[2] -= np.pi / 2
    while object_rot[2] - gripper_rz < -np.pi / 4:
      object_rot[2] += np.pi / 2

    current_target = (block_pos, object_rot, primative)

    return self.getNextActionToCurrentTarget(current_target)
    
  def getNextActionToCurrentTarget(self, current_target):
    x, y, z, r = self.getActionByGoalPose(current_target[0], current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
    else:
      primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100
