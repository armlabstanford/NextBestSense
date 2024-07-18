import time
import rospy
import numpy as np
from pyquaternion import Quaternion
from geometry_msgs.msg import PoseStamped, Pose
from math import pi
from std_srvs.srv import Empty
from scipy.spatial.transform import Rotation as sciR

import kdl_parser_py.urdf as kdl_parser
import PyKDL

class PoseSolver(object):
  def __init__(self,
               base_name="base_link", ee_name="tool_frame",
               cache_size=100):
    robot_description = rospy.get_param(rospy.get_namespace() + "/robot_description")
    ret, kdl_tree = kdl_parser.treeFromString(robot_description)

    if not ret:
      rospy.logerr("Could not parse the URDF")
      return

    self.chain = kdl_tree.getChain(base_name, ee_name)
    self.fk = PyKDL.ChainFkSolverPos_recursive(self.chain)
    self.ik = PyKDL.ChainIkSolverPos_LMA(self.chain)
    self.JacSolver = PyKDL.ChainJntToJacSolver(self.chain)

    self.rng = np.random.default_rng(12345)

    # joints, pose, visit count
    # pre-defined some cache data
    self.cache_result = [
      [(-0.1128, -0.2298, -3.0260, -2.0978, 0.0404, -0.99, 1.53), (0.3088, -0.0099, 0.2494, 0.7079, 0.6921, 0.0975, 0.1013), 1],
      [(-5.9122e-5, 0.2602, 3.1399, -2.2700, 9.6080e-5, 0.9598, 1.5701), (0.6561, 0.0023, 0.4341, 0.4997, 0.5, 0.5, 0.5), 1]
    ]
    self.cache_size = cache_size

  def calcJac(self, joints) -> np.ndarray:
    """ Calculate the Jacobian matrix for the given joint positions 
    
    Args:
      joints (List[float]): The joint positions
    
    Return:
      np.ndarray: The Jacobian matrix (6, num_joints)
    """
    assert len(joints) == self.chain.getNrOfJoints(), "Joint Mismatch; the chain has {} joints, \
                                                        while the input has {} joins".format(self.chain.getNrOfJoints(), len(joints))
    
    q = PyKDL.JntArray(len(joints))
    for i in range(len(joints)):
      q[i] = joints[i]
    jac = PyKDL.Jacobian(len(joints))
    self.JacSolver.JntToJac(q, jac)

    jac_np = np.array([jac.getColumn(k) for k in range(jac.columns())])
    jac_np = jac_np.T # (6, num_joints)
    return jac_np
  
  @staticmethod
  def computePoseDistance(pose1, pose2):
    """ Compute the distance between two poses """
    assert len(pose1) == 7 and len(pose2) == 7, "The input poses should have 7 elements"
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    p1 = np.array(pose1[:3])
    p2 = np.array(pose2[:3])
    
    q1 = np.array(pose1[[6, 3, 4, 5]]) # change to qw, qx, qy, qz
    q2 = np.array(pose2[[6, 3, 4, 5]])

    # copmute quaternion distance
    quat1 = Quaternion(q1)
    quat2 = Quaternion(q2)

    rel_q = quat1.inverse * quat2
    theta = rel_q.radians

    return np.linalg.norm(p1 - p2) + theta

  def __storeCache(self, joint, pose):
    """ Store the joint and pose in the cache """
    
    # check the distance with poses inside cache

    # if the cache is full, remove the least visit element
    if len(self.cache_result) >= self.cache_size:
      visits = [stats[2] for stats in self.cache_result]
      visits = np.array(visits)
      min_idx = np.argmin(visits)

      # remove
      self.cache_result.pop(min_idx)

    self.cache_result.append([joint, pose, 1])

  def calcFK(self, joints) -> np.ndarray:
    """ Calculate the Forward Kinematics for the given joint positions
    
    Args:
      joints (List[float]): The joint positions
    
    Return:
      np.ndarray: The pose (x y z qx qy qz qw)
    """
    assert len(joints) == self.chain.getNrOfJoints(), "Joint Mismatch; the chain has {} joints, \
                                                        while the input has {} joins".format(self.chain.getNrOfJoints(), len(joints))
    q = PyKDL.JntArray(len(joints))
    for i in range(len(joints)):
      q[i] = joints[i]
    frame = PyKDL.Frame()
    self.fk.JntToCart(q, frame)
    
    pos = np.array([frame.p[k] for k in range(3)])
    quat = np.asarray(frame.M.GetQuaternion())
    pose = np.concatenate((pos, quat))
    return pose
  
  def calcIK(self, pose, init_joints=None) -> np.ndarray:
    """ Calculate the Inverse Kinematics for the given pose
    
    Args:
      pose (List[float]): The pose (x y z qx qy qz qw)

    Return:
      np.ndarray: The joint positions
    """
    assert len(pose) == 7, "Pose Mismatch; the input pose should have 7 elements"

    frame = PyKDL.Frame()
    frame.p = PyKDL.Vector(pose[0], pose[1], pose[2])
    frame.M = PyKDL.Rotation.Quaternion(pose[3], pose[4], pose[5], pose[6])
    q = PyKDL.JntArray(self.chain.getNrOfJoints())
    init_qs = []
  
    if init_joints is not None:
      init_q = PyKDL.JntArray(self.chain.getNrOfJoints())
      for i in range(self.chain.getNrOfJoints()):
        init_q[i] = init_joints[i]
      init_qs.append(init_q)
    
    elif len(self.cache_result) > 0:
      rospy.loginfo(" Use Cache result {} for solving ".format(len(self.cache_result)))
      distances = []
      for joint, ee, visit in self.cache_result:
        distance = RandomPoseGenerator.computePoseDistance(ee, pose)
        distances.append(distance)

      distances = np.array(distances)
      # put into the init qs
      sorted_args = np.argsort(distances)
      for ind in sorted_args:
        init_q = PyKDL.JntArray(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
          init_q[i] = self.cache_result[ind][0][i]
          # increase visit
          self.cache_result[ind][2] += 1
        init_qs.append(init_q)

    # iterate through all cache results
    for init_q in init_qs:
      rospy.loginfo(" Try IK solver with {} results".format(len(init_qs)))
      ret = self.ik.CartToJnt(init_q, frame, q)
      
      if ret >= 0:
        joints = np.array([q[i] for i in range(q.rows())])
        
        cache_joints = np.array([stats[0] for stats in self.cache_result])
        joint_distance = np.linalg.norm(cache_joints - joints, axis=1)
        if np.min(joint_distance) > 0.1:
          self.__storeCache(joints, pose)
        
        return joints
      
    rospy.logerr("IK failed")
    return None

class RandomPoseGenerator(PoseSolver):
    def __init__(self, 
                 base_name="base_link", 
                 ee_name="tool_frame",
                cache_size=100 ):
        super().__init__(base_name, ee_name, cache_size)

    def sampleSphere(self, center_pos, radius):
        """ Sample a point on the sphere with given center and radius """
        # sample a point on the sphere
        theta = np.pi / 2 + self.rng.random() * np.pi
        phi = self.rng.random() * np.pi / 6 + np.pi / 3
        offset = np.array([radius * np.sin(phi) * np.cos(theta), radius * np.sin(phi) * np.sin(theta), radius * np.cos(phi)])
        pos = center_pos + offset

        # compute the orientation
        vec_z = -1 * offset / np.linalg.norm(offset)
        
        # y axis is horizontal
        vec_y = np.array([1, -vec_z[0] / vec_z[1], 0])
        vec_y = vec_y / np.linalg.norm(vec_y)

        # x_axis
        vec_x = np.cross(vec_y, vec_z)

        if vec_x[2] > 0:
            vec_x = -vec_x
            vec_y = -vec_y

        R_matrix = np.stack([vec_x, vec_y, vec_z], axis=1)

        # get the quaternion
        quat = sciR.from_matrix(R_matrix).as_quat()    
        pose = np.concatenate((pos, quat))

        return pose

    