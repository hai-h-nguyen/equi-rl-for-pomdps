import numpy as np

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.pybullet.utils import transformations

class CloseLoopPomdpDrawerOpeningHardPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0
    self.current_target = None

    self.moveup_cnt = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      self.current_target = None
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self, target_obj_idx):

    if target_obj_idx == 0:
      handle_pos = self.env.drawer.getHandlePosition()
    else:
      handle_pos = self.env.locked_drawer.getHandlePosition()
    drawer_rot = self.env.drawer_rot
    m = np.array(transformations.euler_matrix(0, 0, drawer_rot))[:3, :3]
    handle_pos = handle_pos + m[:, 0] * 0.025
    handle_pos = handle_pos - m[:, 2] * 0.025
    pull_pos = handle_pos - m[:, 0] * 0.2
    pre_pos = np.copy(handle_pos)
    pre_pos[2] += 0.1

    drawer_rot += np.pi/2
    while drawer_rot > np.pi/2:
      drawer_rot -= np.pi
    while drawer_rot < -np.pi/2:
      drawer_rot += np.pi
    rot = [0, 0, drawer_rot]

    if self.stage == 0:
      # moving to pre
      self.stage = 1
      self.current_target = (pre_pos, rot, constants.PICK_PRIMATIVE)
    elif self.stage == 1:
      # moving to handle
      self.stage = 2
      self.current_target = (handle_pos, rot, constants.PICK_PRIMATIVE)
    elif self.stage == 2:
      self.stage = 0
      self.current_target = (pull_pos, rot, constants.PICK_PRIMATIVE)

  def getNextAction(self, target_obj_idx):
    if(self.env.drawer.isDrawerOpen(0.13)):
      self.moveup_cnt += 1
      if self.moveup_cnt >= 5:
        return self._getNextActionPickObj()
      else:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    if self.env.current_episode_steps == 1 or target_obj_idx == 2:
      self.stage = 0
      self.moveup_cnt = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      if target_obj_idx == 2:
        target_obj_idx = 1
      self.setNewTarget(target_obj_idx)
      return self.getNextActionToCurrentTarget()

  def _getNextActionPickObj(self):
    if not self.env._isHolding():
      block_pos = self.env.objects[0].getPosition()
      block_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())

      x, y, z, r = self.getActionByGoalPose(block_pos, block_rot)

      if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
        primitive = constants.PICK_PRIMATIVE
      else:
        primitive = constants.PLACE_PRIMATIVE

    else:
      x, y, z = 0, 0, self.dpos
      r = 0
      primitive = constants.PICK_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100
