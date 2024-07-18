#!/usr/bin/env python3

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_move_it_trajectories.py __ns:=my_gen3

import sys
sys.path.append('/miniconda/envs/densetact/lib/python3.8/site-packages')

import time
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from pyquaternion import Quaternion
from geometry_msgs.msg import PoseStamped, Pose
from math import pi
from std_srvs.srv import Empty
from scipy.spatial.transform import Rotation as sciR

import kdl_parser_py.urdf as kdl_parser
import PyKDL

from typing import List

TOPVIEW = [0.656, 0.002, 0.434, 0.707, 0.707, 0., 0.]
OBJECT_CENTER = np.array([0.4, 0., 0.1])

class RandomPoseGenerator(object):
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

class ExampleMoveItTrajectories(object):
  """ExampleMoveItTrajectories"""
  def __init__(self):

    # Initialize the node
    super(ExampleMoveItTrajectories, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('example_move_it_trajectories')

    try:
      self.is_gripper_present = rospy.get_param(rospy.get_namespace() + "is_gripper_present", False)
      if self.is_gripper_present:
        gripper_joint_names = rospy.get_param(rospy.get_namespace() + "gripper_joint_names", [])
        self.gripper_joint_name = gripper_joint_names[0]
      else:
        self.gripper_joint_name = ""
      self.degrees_of_freedom = rospy.get_param(rospy.get_namespace() + "degrees_of_freedom", 7)

      # Create the MoveItInterface necessary objects
      arm_group_name = "arm"
      self.robot = moveit_commander.RobotCommander("robot_description")
      self.scene = moveit_commander.PlanningSceneInterface(ns=rospy.get_namespace())
      self.arm_group = moveit_commander.MoveGroupCommander(arm_group_name, ns=rospy.get_namespace())
      self.display_trajectory_publisher = rospy.Publisher(rospy.get_namespace() + 'move_group/display_planned_path',
                                                    moveit_msgs.msg.DisplayTrajectory,
                                                    queue_size=20)

      if self.is_gripper_present:
        gripper_group_name = "gripper"
        self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

      rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
    except Exception as e:
      print (e)
      self.is_init_success = False
    else:
      self.is_init_success = True

    self.pose_generator = RandomPoseGenerator()

  def reach_named_position(self, target):
    arm_group = self.arm_group
    
    # Going to one of those targets
    rospy.loginfo("Going to named target " + target)
    # Set the target
    arm_group.set_named_target(target)
    # Plan the trajectory
    (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
    # Execute the trajectory and block while it's not finished
    return arm_group.execute(trajectory_message, wait=True)

  def reach_joint_angles(self, joint_positions, tolerance=0.05):
    arm_group = self.arm_group
    success = True

    # Set the goal joint tolerance
    self.arm_group.set_goal_joint_tolerance(tolerance)
    arm_group.set_joint_value_target(joint_positions)
    
    # Plan and execute in one command
    success &= arm_group.go(wait=True)

    # Show joint positions after movement
    new_joint_positions = arm_group.get_current_joint_values()
    rospy.loginfo("Printing current joint positions after movement :")
    for p in new_joint_positions: rospy.loginfo(p)
    return success

  def get_cartesian_pose(self) -> Pose:
    arm_group = self.arm_group

    # Get the current pose and display it
    # return geometry_msgs PoseStamped
    pose:PoseStamped = arm_group.get_current_pose() 
    rospy.loginfo("Actual cartesian pose is : ")
    rospy.loginfo(pose.pose)

    return pose.pose

  def reach_cartesian_pose(self, pose, tolerance, constraints):
    arm_group = self.arm_group
    
    # Set the tolerance
    arm_group.set_goal_position_tolerance(tolerance)

    # Set the trajectory constraint if one is specified
    if constraints is not None:
      arm_group.set_path_constraints(constraints)

    # Get the current Cartesian Position
    arm_group.set_pose_target(pose)

    # Plan and execute
    rospy.loginfo("Planning and going to the Cartesian Pose")
    return arm_group.go(wait=True)

  def reach_gripper_position(self, relative_position):
    gripper_group = self.gripper_group
    
    # We only have to move this joint because all others are mimic!
    gripper_joint = self.robot.get_joint(self.gripper_joint_name)
    gripper_max_absolute_pos = gripper_joint.max_bound()
    gripper_min_absolute_pos = gripper_joint.min_bound()
    try:
      val = gripper_joint.move(relative_position * (gripper_max_absolute_pos - gripper_min_absolute_pos) + gripper_min_absolute_pos, True)
      return val
    except:
      return False 

def main():
  example = ExampleMoveItTrajectories()

  # For testing purposes
  success = example.is_init_success
  try:
      rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
  except:
      pass
  
  sample_pose = rospy.Publisher("/sample_pose", PoseStamped, queue_size=1, latch=True)

  # if success:
  #   rospy.loginfo("Reaching Named Target Vertical...")
  #   success &= example.reach_named_position("vertical")
  #   print (success)
  
  # if success:
  #   rospy.loginfo("Reaching Joint Angles...")  
  #   success &= example.reach_joint_angles(tolerance=0.01) #rad
  #   print (success)
  
  if success:
    rospy.loginfo("Reaching Named Target Home...")
    success &= example.reach_named_position("home")
    print(success)
  
  for _ in range(12):
    import pdb; pdb.set_trace()
    pose = example.pose_generator.sampleSphere(OBJECT_CENTER, 0.2)
    joints = example.pose_generator.calcIK(pose)
    print(joints)

    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "base_link"
    pose_msg.header.stamp = rospy.Time.now()

    pose_msg.pose.position.x = pose[0]
    pose_msg.pose.position.y = pose[1]
    pose_msg.pose.position.z = pose[2]
    pose_msg.pose.orientation.x = pose[3]
    pose_msg.pose.orientation.y = pose[4]
    pose_msg.pose.orientation.z = pose[5]
    pose_msg.pose.orientation.w = pose[6]
    sample_pose.publish(pose_msg)

    if joints is not None:
      try:
        success &= example.reach_joint_angles(joints)
      except:
        rospy.logwarn("Fail to Execute")

  # if success:
  #   rospy.loginfo("Reaching Cartesian Pose...")
  #   success &= example.reach_cartesian_pose(pose=TOPVIEW, tolerance=0.01, constraints=None)
  #   print(success)

  # # For testing purposes
  # rospy.set_param("/kortex_examples_test_results/moveit_general_python", success)

  # if not success:
  #     rospy.logerr("The example encountered an error.")

if __name__ == '__main__':
  main()
