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
# Author: Acorn Pooley, Mike Lautman, Boshu Lei, Matt Strong

# Inspired from http://docs.ros.org/kinetic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html
# Modified by Alexandre Vannobel to test the FollowJointTrajectory Action Server for the Kinova Gen3 robot

# To run this node in a given namespace with rosrun (for example 'my_gen3'), start a Kortex driver and then run : 
# rosrun kortex_examples example_move_it_trajectories.py __ns:=my_gen3

import sys

from matplotlib import pyplot as plt
sys.path.append('/miniconda/envs/densetact/lib/python3.8/site-packages')

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest, Empty
from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest, SaveModel, SaveModelResponse, SaveModelRequest

from scipy.spatial.transform import Rotation as sciR
from kinova_control_py.pose_util import RandomPoseGenerator

import kdl_parser_py.urdf as kdl_parser
import PyKDL

from typing import List

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

import pickle


TOPVIEW = [0.656, 0.002, 0.434, 0.707, 0.707, 0., 0.]
OBJECT_CENTER = np.array([0.4, 0., 0.1])
BOX_DIMS = (0.15, 0.15, 0.13)

EXP_POSES = {
  "starting_joints": [],
  "starting_poses": [],
  "candidate_joints": [],
  "candidate_poses": []
}

PICKLE_PATH_FULL = '/home/user/NextBestSense/data/EXP_POSES_1.pkl'

class ExampleMoveItTrajectories(object):
  """ExampleMoveItTrajectories"""
  def __init__(self):

    # Initialize the node
    super(ExampleMoveItTrajectories, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('example_move_it_trajectories')
    rospy.loginfo("Initializing ExampleMoveItTrajectories")
    
    # read pickle file if it exists
    # if it does not exist, create it
    try:
      with open(PICKLE_PATH_FULL, "rb") as f:
        self.exp_poses = pickle.load(f)
        self.exp_poses_available = True
    except FileNotFoundError:
      self.exp_poses_available = False

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
      # set max acc scaling factor
      self.arm_group.set_max_acceleration_scaling_factor(0.3)
      
      box_pose = geometry_msgs.msg.PoseStamped()
      box_pose.pose.orientation.w = 1.0
      box_pose.pose.position.x = OBJECT_CENTER[0]
      box_pose.pose.position.y = OBJECT_CENTER[1]
      box_pose.pose.position.z = OBJECT_CENTER[2]
      box_pose.header.frame_id = 'base_link'
      box_name = "box"
      # add box to the scene. In the future, resize to object size in GS
      self.scene.add_box(box_name, box_pose, size=BOX_DIMS)
      rospy.loginfo("Added box to the scene")

      if self.is_gripper_present:
        gripper_group_name = "gripper"
        self.gripper_group = moveit_commander.MoveGroupCommander(gripper_group_name, ns=rospy.get_namespace())

      rospy.loginfo("Initializing node in namespace " + rospy.get_namespace())
    except Exception as e:
      print (e)
      self.is_init_success = False
    else:
      self.is_init_success = True
      
      
    rospy.loginfo("Initialization done. Generating Poses ...")

    self.pose_generator = RandomPoseGenerator()
    self.num_poses = 10
    
    # wait for vision node service
    rospy.loginfo("Waiting for Vision Node Services...")
    
    rospy.wait_for_service("/add_view")
    rospy.wait_for_service("/next_best_view")
    rospy.wait_for_service("/save_model")

    self.add_view_client = rospy.ServiceProxy("/add_view", Trigger)
    self.nbv_client = rospy.ServiceProxy("/next_best_view", NBV)
    self.save_model_client = rospy.ServiceProxy("/save_model", SaveModel)

    rospy.loginfo("Vision Node Services are available")
    
    
    self.finish_training_service = rospy.Service("finish_training", Trigger, self.finishTrainingCb)
    
    self.training_done = False
    
    
  def finishTrainingCb(self, req) -> TriggerResponse:
      """ AddVision Cb 
      
      return TriggerResponse:
          bool success
          message:
              form: Error Code X: [Error Message]
              X=1 -> Training thread is still running
              X=2 -> Unsupported image encoding
              X=3 -> Failed to lookup transform from camera to base_link
      """
      
      res = TriggerResponse()
      res.success = True
      res.message = "Robot finished training GS. Now adding test views."
      
      self.training_done = True
      rospy.loginfo("Training Done")
      
      return res
    
  def get_exp_poses_at(self, i, view_type):
    rospy.loginfo("Loading EXP_POSES")
    if view_type == 0:
      candidate_joints = self.exp_poses["starting_joints"]
      poses = self.exp_poses["starting_poses"]
      return [candidate_joints[i]], [poses[i]]
    else:
      candidate_joints = self.exp_poses["candidate_joints"]
      poses = self.exp_poses["candidate_poses"]
      idx = i - len(self.exp_poses["starting_joints"])
      return candidate_joints[idx], poses[idx]

  def reach_named_position(self, target):
    arm_group = self.arm_group
    
    # Going to one of those targets
    rospy.loginfo("Going to named target " + target)
    # Set the target
    arm_group.set_named_target(target)
    # Plan the trajectory
    (success_flag, trajectory_message, planning_time, error_code) = arm_group.plan()
    print(success_flag, planning_time, error_code)
    # Execute the trajectory and block while it's not finished
    return arm_group.execute(trajectory_message, wait=True)

  def reach_joint_angles(self, joint_positions, tolerance=0.001):
    arm_group = self.arm_group
    success = True

    # Set the goal joint tolerance
    self.arm_group.set_goal_joint_tolerance(tolerance)
    arm_group.set_joint_value_target(joint_positions)
    
    # Plan and execute in one command
    success &= arm_group.go(wait=True)
    arm_group.stop()

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

  def send_req_helper(self, client, req):
    """ Send request helper """
    while True:
      res = client(req)

      if res.success:
        break
      else:
        # error analysis
        error_code = int(res.message.split(":")[0].split(" ")[2])

        if error_code in [2, 3]:
          rospy.logwarn(res.message)
          exit()
        elif error_code == 1:
          # wait for training loop to be finised
          rospy.loginfo(res.message)
          rospy.sleep(1)
        else:
          raise NotImplementedError("Error Code {} not implemented".format(error_code))
        
    return res

  def convertNumpy2PoseStamped(self, pose:np.ndarray) -> PoseStamped:
    pose_msg = PoseStamped()
    # relative to world
    pose_msg.header.frame_id = "base_link"
    pose_msg.header.stamp = rospy.Time.now()

    # convert R7 to msg
    if pose.ndim == 1:
      pose_msg.pose.position.x = pose[0]
      pose_msg.pose.position.y = pose[1]
      pose_msg.pose.position.z = pose[2]
      pose_msg.pose.orientation.x = pose[3]
      pose_msg.pose.orientation.y = pose[4]
      pose_msg.pose.orientation.z = pose[5]
      pose_msg.pose.orientation.w = pose[6]
    
    # convert 4x4 to msg
    elif pose.ndim == 2:
      pose_msg.pose.position.x = pose[0, 3]
      pose_msg.pose.position.y = pose[1, 3]
      pose_msg.pose.position.z = pose[2, 3]

      Rot = sciR.from_matrix(pose[:3, :3])
      quat = Rot.as_quat()

      pose_msg.pose.orientation.x = quat[0]
      pose_msg.pose.orientation.y = quat[1]
      pose_msg.pose.orientation.z = quat[2]
      pose_msg.pose.orientation.w = quat[3]

    return pose_msg
  
  def run(self):
    """ Run Method (Main Thread) """
    
    success = self.is_init_success
    try:
        rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
    except:
        pass
    
    if success:
      rospy.loginfo("Reaching Named Target Home...")
      success &= self.reach_named_position("home")
      
    start_views = 4
    total_views_to_add = 25
    view_type_ids = []
    for i in range(start_views):
      view_type_ids.append(0)
    for i in range(total_views_to_add):
      view_type_ids.append(1)
      
    total_iters = len(view_type_ids)
    i = 0
    
    while i < total_iters:
      # Next Best View
      pose_req = NBVRequest()
      candidate_joints = []
      rospy.loginfo(view_type_ids[i])
      
      # Sample views near the sphere until we have 10 poses
      if self.exp_poses_available:
        candidate_joints, pose_req.poses = self.get_exp_poses_at(i, view_type_ids[i])
       
      else:
        pose_cnt = 0
        while pose_cnt < self.num_poses:
          pose = self.pose_generator.sampleInSphere(OBJECT_CENTER, 0.15, 0.6)
          joints = self.pose_generator.calcIK(pose) 

          # make plans to reach the pose
          success = False
          while not success:
            try:
              success, trajectory, planning_time, err_code = self.arm_group.plan(joints)
            except: 
              success = False
              rospy.logwarn("Fail to Plan Trajectory")
              pose = self.pose_generator.sampleInSphere(OBJECT_CENTER, 0.15, 0.6)
              joints = self.pose_generator.calcIK(pose) 
          
          # if plan succeeds, we can add the pose as valid
          if success:
            pose_cnt += 1

            pose_msg:PoseStamped = self.convertNumpy2PoseStamped(pose)
            pose_req.poses.append(pose_msg)
            candidate_joints.append(joints)
      if view_type_ids[i] == 0:
        # choose random joints
        if self.exp_poses_available:
          joints = candidate_joints[0]
          pose = pose_req.poses[0]
        else:
          rand_idx = np.random.randint(0, self.num_poses)
          joints = candidate_joints[rand_idx]
          pose = pose_req.poses[rand_idx]
        
      elif view_type_ids[i] == 1:
        # send the request for next best view
        rospy.loginfo("Ready to call NBV")
        res: NBVResponse = self.send_req_helper(self.nbv_client, pose_req)
        scores = np.array(res.scores)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_joints = [candidate_joints[i] for i in sorted_indices]
        
        joints = sorted_joints[0]
        joint_config_idx = 0
        
      # reach the view
      if joints is not None:
        try:
          success &= self.reach_joint_angles(joints)
          
          if view_type_ids[i] == 1:
            if not success:
              rospy.logwarn("Fail to Reach Joint Angles. Try Next Views")
              # skip other code and go to next view
              while not success and joint_config_idx < len(sorted_joints):
                joint_config_idx += 1
                joints = sorted_joints[joint_config_idx]
                success = self.reach_joint_angles(joints) 
          
          if view_type_ids[i] == 1:
            rospy.loginfo("Went to Next Best View")
            
        except:
          rospy.logwarn("Fail to Execute Joint Trajectory")
          
        if not success:
          rospy.logwarn("Fail to Reach Joint Angles. Try Next View")
          # skip other code and go to next view
          continue
        else: 
          if view_type_ids[i] == 0:
            EXP_POSES["starting_joints"].append(joints)
            EXP_POSES["starting_poses"].append(pose)
            
          else:
            # list of lists
            EXP_POSES["candidate_joints"].append(candidate_joints)
            EXP_POSES["candidate_poses"].append(pose_req.poses)
            
          # continually write EXP_POSES to file pickle
          rospy.loginfo("Writing EXP_POSES")
          # write pickle file
          with open(PICKLE_PATH_FULL, "wb") as f:
            pickle.dump(EXP_POSES, f)
          
          i += 1
          
        # Add the view
        req = TriggerRequest()
        res = self.send_req_helper(self.add_view_client, req)
        
        # train the model at the end of the first few views
        if i >= start_views:
          rospy.loginfo("Saving Model with new pose ...")
          req = SaveModelRequest()
          req.success = success
          res = self.send_req_helper(self.save_model_client, req)
          if "Test" in res.message:
            rospy.loginfo("Model is in test mode, exiting gracefully")
            exit()
        else:
          rospy.loginfo("Adding new train view...")
        
    # Save the model
    rospy.loginfo("Save Model ...")
    req = TriggerRequest()
    self.send_req_helper(self.save_model_client, req)

    return success

def main():
  example = ExampleMoveItTrajectories()
  example.run()

if __name__ == '__main__':
  main()
