#!/usr/bin/env python3

# Author: Acorn Pooley, Mike Lautman, Boshu Lei, Matt Strong

import os

import sys
from typing import List, Callable
from functools import partial
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import binary_dilation, disk
import json as js
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as sciR
from matplotlib import pyplot as plt
sys.path.append('/miniconda/envs/densetact/lib/python3.8/site-packages')

import rospy
import moveit_commander
import cv2
import os
from os import path as osp
from cv_bridge import CvBridge

import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Point


from voxblox_msgs.srv import FilePathRequest, FilePath
from voxblox_msgs.srv import QueryTSDFRequest, QueryTSDF
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from kortex_driver.msg import TwistCommand
import message_filters

from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest, Empty
from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest, SaveModel, SaveModelResponse, SaveModelRequest
import open3d as o3d
import rosservice

# tf buffer
import tf2_ros

from kinova_control_py.pose_util import RandomPoseGenerator, calcAngDiff
from kinova_control_py.status import *
from kinova_control_py.april_tag_detector import detect_img

from open3d.pipelines.integration import ScalableTSDFVolume
if 'tsdf_at' in dir(ScalableTSDFVolume):
  Open3D_installed = True
else:
  Open3D_installed = False

Open3D_installed = False

TOPVIEW = [0.656, 0.002, 0.434, 0.707, 0.707, 0., 0.]
# OBJECT_CENTER = np.array([0.4, 0., 0.1]
OBJECT_CENTER = np.array([0.5, 0., 0.0])
# BOX_DIMS = (0.15, 0.15, 0.01)
BOX_DIMS = (0.07, 0.07, 0.01)

TOUCH_DATA_DIR = "/home/user/NextBestSense/data/touch_data/touch"


STARTING_VIEW_ID = 0
ADDING_VIEW_ID = 1

EXP_POSES = {
  "starting_joints": [],
  "starting_poses": [],
  "candidate_joints": [],
  "candidate_poses": []
}

PICKLE_PATH_FULL = 'EXP_POSES.pkl'

def mask_filter(vertices, gaussian_data_dir):
  """ Mask Filter using the SAM2 Data 
  
  Args:
    vertices: (N, 3) Vertices of the Mesh in the world frame
    gaussian_data_dir: The directory where the gaussian data is saved
  """

  vertices_cu = torch.from_numpy(vertices).float().cuda()
  vertices_cu = torch.cat([vertices_cu, torch.ones((vertices_cu.shape[0], 1)).cuda()], dim=1) # (N, 4)
  vertices_cu = vertices_cu.T # (4, N)

  # read json file
  with open(osp.join(gaussian_data_dir, 'transforms.json')) as f:
      data = js.load(f)
  
  intrinsic = np.eye(3)
  intrinsic[0, 0] = data["fl_x"]
  intrinsic[1, 1] = data["fl_y"]
  intrinsic[0, 2] = data["cx"]
  intrinsic[1, 2] = data["cy"]
  intrinsic_cu = torch.from_numpy(intrinsic).float().cuda()

  W, H = data["w"], data["h"]

  frames = data["frames"]
  sampled_masks = []
  for frame in frames:
    mask_path = frame["mask_path"]
    mask_img = cv2.imread(osp.join(gaussian_data_dir, mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255. 
    mask_img = torch.from_numpy(binary_dilation(mask_img, disk(24))).float()[None, None].cuda()

    c2w = np.array(frame["transform_matrix"])
    w2c = np.linalg.inv(c2w)
    w2c = w2c[:3, :]
    w2c_cu = torch.from_numpy(w2c).float().cuda()

    # get vertices in the camera frame
    vertices_c = intrinsic_cu @ w2c_cu @ vertices_cu # (4, N)
    pix_coords = vertices_c[:2, :] / (vertices_c[2, :].unsqueeze(0) + 1e-6)
    pix_coords = pix_coords.permute(1, 0) # (N, 2)
    pix_coords[..., 0] = pix_coords[..., 0] / W
    pix_coords[..., 1] = pix_coords[..., 1] / H
    pix_coords = pix_coords * 2 - 1
    valid = ((pix_coords > -1) & (pix_coords < 1)).all(dim=1).float()

    sampled_mask = \
          F.grid_sample(mask_img, pix_coords[None, None], mode='nearest', align_corners=True, padding_mode="zeros")
    sampled_mask = sampled_mask.squeeze()
    sampled_mask = sampled_mask + (1 - valid)
    sampled_masks.append(sampled_mask)
  
  sampled_mask = torch.stack(sampled_masks, dim=-1)
  mask = (sampled_mask > 0.).all(dim=-1).cpu().numpy()

  return mask

class TouchGSController(object):
  """TouchGSController"""
  def __init__(self):

    # Initialize the node
    super(TouchGSController, self).__init__()
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('touch-gs-controller')
    rospy.loginfo("Initializing Touch-GS controller")
    self.bridge = CvBridge()

    self.tfBuffer = tf2_ros.Buffer()
    self.listener = tf2_ros.TransformListener(self.tfBuffer)
    
    """
    Description of below parameters:
    starting_views: number of views to start with. If should_collect_test_views in the launch file is True, we will collect this num of views and gracefully exit.
    
    added_views: number of views to add after the starting views. If should_collect_test_views is false, we will add this num of views.
    
    should_collect_experiment: if True, we will collect the experiment data and save it to a pickle file, or read the data in. If not, we do not create any pickle file and just add random views.
    """
    
    self.starting_views = int(rospy.get_param("~starting_views", "5"))
    self.added_views = int(rospy.get_param("views_to_add", "10"))
    self.num_touches_to_add = int(rospy.get_param("touches_to_add", "10"))
        
    self.should_collect_experiment = bool(rospy.get_param("~should_collect_experiment", "False"))
    self.use_touch = bool(rospy.get_param("~use_touch", "False"))
    
    # if should_collect_experiment is True, then we will collect the experiment data or read it in from pickle file
    self.exp_poses_available = False
    if self.should_collect_experiment:
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
      print(e)
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
    rospy.wait_for_service("/get_gs_data_dir")

    self.add_view_client = rospy.ServiceProxy("/add_view", Trigger)
    self.nbv_client = rospy.ServiceProxy("/next_best_view", NBV)
    self.save_model_client = rospy.ServiceProxy("/save_model", SaveModel)
    self.get_gs_data_dir_client = rospy.ServiceProxy("/get_gs_data_dir", Trigger)

    self.dt_deform_thresh = False # if True, means the DT sensor exceeds the threshold, and should be stopped
    if self.use_touch:
      self.DT_DEPTH_TOPIC = "/RunCamera/imgDepth_show"
      img: Image = rospy.wait_for_message(self.DT_DEPTH_TOPIC, Image)
      self.depth_undeformed = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

      self.dt_deform_threshold_value = rospy.get_param("~dt_deform_threshold_value", 95)
      rospy.loginfo("Depth Deformation Threshold Value: {}".format(self.dt_deform_threshold_value))
      self.depthImg_sub = rospy.Subscriber(self.DT_DEPTH_TOPIC, Image, self.depthImgCb)

      self.touch_json_header = {
        "w": 640,
        "h": 640,
        "near": 1e-4,
        "far": 3.27,
        "camera_angle_x": 0.5236,
        "frames": []
      }

    if Open3D_installed:
      # create sync rgb and depth 
      self.integrator = ScalableTSDFVolume(
        voxel_length=0.004,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
      )
      self.depth_sub = message_filters.Subscriber("/camera/depth/image_rect_raw", Image)
      self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
      self.info_sub = message_filters.Subscriber("/camera/color/camera_info", CameraInfo)

      self.tsdf_sync = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.info_sub], 10, 0.1)
      self.tsdf_sync.registerCallback(self.RGBDCallback)    

    rospy.loginfo("Vision Node Services are available")
    
    self.finish_training_service = rospy.Service("finish_training", Trigger, self.finishTrainingCb)
    self.training_done = False

    self.create_view_types_list()

  def RGBDCallback(self, depth_msg:Image, color_msg:Image, info_msg:CameraInfo):
    """ RGBD Callback """
    # convert to numpy array
    color_np = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")
    depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
    depth_np = depth_np.astype(np.float32) / 1000.0
    depth_np = cv2.resize(depth_np, (info_msg.width, info_msg.height))

    # get transform from camera to base_link
    try:
      transform = self.tfBuffer.lookup_transform("base_link", depth_msg.header.frame_id, rospy.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
      rospy.logwarn("Fail to lookup transform from camera to base_link")
      return EXECUTION_FAILURE

    c2w = np.eye(4)
    c2w[:3, 3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    q = transform.transform.rotation
    q = Quaternion(q.w, q.x, q.y, q.z)
    c2w[:3, :3] = q.rotation_matrix

    extrinsincs = np.linalg.inv(c2w)

    # get instrinsics
    K = np.array(info_msg.K).reshape(3, 3)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(info_msg.width, info_msg.height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_np), o3d.geometry.Image(depth_np), depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

    self.integrator.integrate(rgbd, intrinsic, extrinsincs)

  def pointcloudCb(self, msg:PointCloud2):
    pass

    # get transform from camera to base_link
    try:
      transform = self.tfBuffer.lookup_transform("base_link", msg.header.frame_id, rospy.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
      rospy.logwarn("Fail to lookup transform from camera to base_link")
      return EXECUTION_FAILURE
  
    c2w = np.eye(4)
    c2w[:3, 3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    q = transform.transform.rotation
    q = Quaternion(q.w, q.x, q.y, q.z)
    c2w[:3, :3] = q.rotation_matrix

    self.integrator.integratePointCloud(msg, c2w)

  def depthImgCb(self, msg:Image):
    """ DT Depth Image Callback """
    # convert to numpy array
    img_np = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    depth_deformation = np.abs(img_np - self.depth_undeformed).mean()
    if depth_deformation > self.dt_deform_threshold_value and not self.dt_deform_thresh:
      rospy.logwarn("Depth Deformation is too high DIff {}".format(depth_deformation))
    
    self.dt_deform_thresh = depth_deformation > self.dt_deform_threshold_value
  
  def get_gs_data_dir(self):
    """ Get the GS Data Directory """
    req = TriggerRequest()
    res = self.get_gs_data_dir_client(req)
    return res.message

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
    """Reaches Cartesian Pose given the pose, tolerance and constraints"""
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
    """ Send request helper with ROS service"""
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
    """ Convert Numpy to PoseStamped 
      If pose is 1D, it is a 7D pose, in the order of x y z qx qy qz qw
      
      If pose is 2D, it is a 4x4 pose
    """
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
  
  def vel_control(self):
    """ Velocity Control """
    vel_pub = rospy.Publisher("/my_gen3/in/cartesian_velocity", TwistCommand, queue_size=10)
    command = TwistCommand()
    # always choose the world frame
    command.reference_frame = 0
    command.twist.linear_x = 0.01
    command.twist.linear_y = 0.
    command.twist.linear_z = 0.
    command.twist.angular_x = 0.
    command.twist.angular_y = 0.
    command.twist.angular_z = 0.
    
    for i in range(10):
      vel_pub.publish(command)
      rospy.sleep(0.1)
    
    return True
  
  def vel_approach(self, target_pose, start_pose, exec_time=10,
                   trans_tolerate=0.002, ang_tolerate=0.01, dt_safety_check=True):
    """ 
    Use velocity control to approach the target pose 
    
    Use the Linear Interpolation for velocity control
      A simple P controller is adopted to compute the velocity command.
      
    Should use feedback from the robot to determine the velocity command.
    
    Start Pose: The pose where the robot starts (x y z qx qy qz qw)
    Target Pose: The pose where the robot wants to reach (x y z qx qy qz qw)
    dt_safety_check: If True, when the depth sensor exceeds the threshold, stop the robot
                  disable it when you are sure the sensor is moving away from the object 
    """ 
      
    rospy.loginfo("Starting Velocity Control")
    rospy.loginfo("Move to Start Pose")
    
    joints = self.pose_generator.calcIK(start_pose) 
    success, trajectory, planning_time, err_code = self.arm_group.plan(joints)
    if not success:
      rospy.logwarn("Fail to Plan Trajectory")
      return PLAN_FAILURE
    
    success = self.reach_joint_angles(joints) 
    if not success:
      rospy.logwarn("Fail to Reach Joint Angles")
      return EXECUTION_FAILURE
    
    # start velocity control
    vel_pub = rospy.Publisher("/my_gen3/in/cartesian_velocity", TwistCommand, queue_size=10)
    command = TwistCommand()
    
    rate = rospy.Rate(30)
    start_time = rospy.Time.now().to_sec()
    q_target = Quaternion(target_pose[6], target_pose[3], target_pose[4], target_pose[5])
    q_start = Quaternion(start_pose[6], start_pose[3], start_pose[4], start_pose[5])

    status = SUCCESS
    while True:
      current_time = rospy.Time.now().to_sec()
      t = current_time - start_time
      if t > exec_time + 0.1:
        rospy.logwarn("Time Exceeded")
        status = TIME_EXCEEDED
        break

      if dt_safety_check and self.dt_deform_thresh:
        rospy.logwarn("Depth Deformation Exceeded")
        status = DT_THRESHOLD_EXCEED
        break
      
      # compute the execution time
      t = min(t, exec_time)
      
      # compute pose error 
      # get current pose
      current_pose = self.get_cartesian_pose()
      xcur = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
      angcur = np.array([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
      q_cur = Quaternion(angcur[3], angcur[0], angcur[1], angcur[2])
      
      trans_error = np.linalg.norm(xcur - target_pose[:3])
      relative_q = q_cur.inverse * q_target
      ang_error = abs(relative_q.angle)
      
      if trans_error < trans_tolerate and ang_error < ang_tolerate:
        rospy.loginfo("Reached Target Pose")
        break
      
      # linear interpolation
      xdes = t / exec_time * (target_pose[:3] - start_pose[:3]) + start_pose[:3]
      vdes = (target_pose[:3] - start_pose[:3]) / exec_time
       
      # P controller
      v = 5 * (xdes - xcur) + vdes
      # rospy.loginfo("Current v: {}".format(v))
      
      # compute the desired orientation
      qdes = (t / exec_time) * q_target + (1 - t / exec_time) * q_start
      qdes = qdes.normalised
      q_dot = (q_target - q_start) / exec_time
      ang_des = 2 * q_dot * qdes.inverse
      ang_vdes = np.array([ang_des.x, ang_des.y, ang_des.z])
      
      rospy.loginfo("Current Ang Vel: {}".format(ang_vdes))
      
      Rdes = qdes.rotation_matrix
      Rcur = q_cur.rotation_matrix
      ang_diff = calcAngDiff(Rdes, Rcur)
      ang_vel = 5 * ang_diff + ang_vdes
      
      rospy.loginfo("Current qdes: {}, qcur {}".format(qdes, q_cur))
      rospy.loginfo("Current Ang Diff: {}".format(ang_diff))
      
      # convert to EE frame
      ang_vel = Rcur.T @ ang_vel
      
      # set the velocity
      # the velocity is in the base frame
      # the angular velocity is in the end effector frame
      command.reference_frame = 0
      command.twist.linear_x = v[0]
      command.twist.linear_y = v[1]
      command.twist.linear_z = v[2]
      command.twist.angular_x = ang_vel[0]
      command.twist.angular_y = ang_vel[1]
      command.twist.angular_z = ang_vel[2]
      
      vel_pub.publish(command)
      rate.sleep()   
      
    # stop the robot
    command.twist.linear_x = 0.
    command.twist.linear_y = 0.
    command.twist.linear_z = 0.
    command.twist.angular_x = 0.
    command.twist.angular_y = 0.
    command.twist.angular_z = 0.
    vel_pub.publish(command) 
    
    # compute the final pose error
    current_pose = self.get_cartesian_pose()
    xcur = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
    angcur = np.array([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
    q_cur = Quaternion(angcur[3], angcur[0], angcur[1], angcur[2])
    
    trans_error = np.linalg.norm(xcur - target_pose[:3])
    relative_q = q_cur.inverse * q_target
    ang_error = abs(relative_q.angle)
    rospy.loginfo("Trans Error: {}, Ang Error {}".format(trans_error, ang_error))

    return status

  def touch_pose(self, pose, distance = 0.1, cb_func:Callable = None):
    """ 
    Get One touch at the pose
    
    pose: The pose where the robot starts (x y z qx qy qz qw)
    distance: The distance to move the robot in the direction of the pose
    """
    # update undeformed depth image
    img: Image = rospy.wait_for_message(self.DT_DEPTH_TOPIC, Image)
    self.depth_undeformed = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    touch_q = Quaternion(pose[6], pose[3], pose[4], pose[5])
    touch_pos = np.array([pose[0], pose[1], pose[2]])
    rot = touch_q.rotation_matrix

    # the z-axis is the normal
    axis_z = rot[:, 2]
    start_pos = touch_pos - distance * axis_z

    start_pose = np.array([start_pos[0], start_pos[1], start_pos[2], 
                           pose[3], pose[4], pose[5], pose[6]])

    status = self.vel_approach(pose, start_pose, exec_time=6)

    if status == SUCCESS or status == DT_THRESHOLD_EXCEED:
      # TODO Call the Touch Service 
      # get the touch sensor pose now
      status = cb_func()

      # return to the start pose
      # get current pose 
      current_pose = self.get_cartesian_pose()
      current_pose = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z, 
                               current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
      self.vel_approach(start_pose, current_pose, exec_time=3, dt_safety_check=False)
    
    elif status == PLAN_FAILURE or status == EXECUTION_FAILURE:
      rospy.logwarn("Fail to Reach Target Pose")
      # since the robot does not move, just return is OK

    elif status == TIME_EXCEEDED:
      # pure time exceed, no idea now
      pass

    return status
  
  def get_aruco_marker_coord(self):
    """ Get Aruco Marker Board Coord in world frame"""
    # go to home pose
    self.reach_named_position("home")

    current_pose = self.get_cartesian_pose()
    q_cur = Quaternion(current_pose.orientation.w, current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z)
    delta_q = Quaternion(axis=[0, 1, 0], angle=np.pi / 2)
    q_start = delta_q * q_cur # * delta_q

    # move to the observe pose
    observe_pose = np.array([current_pose.position.x, current_pose.position.y, current_pose.position.z,
                              q_start.x, q_start.y, q_start.z, q_start.w])
    joints = self.pose_generator.calcIK(observe_pose) 
    success, trajectory, planning_time, err_code = self.arm_group.plan(joints)
    if not success:
      rospy.logwarn("Fail to Plan Trajectory")
      return PLAN_FAILURE
    
    success = self.reach_joint_angles(joints) 
    if not success:
      rospy.logwarn("Fail to Reach Joint Angles")
      return EXECUTION_FAILURE

    # sleep one sec
    rospy.sleep(1)
    img = rospy.wait_for_message("/camera/color/image_raw", Image)
    img_np = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    # get the camera matrix
    camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
    K = np.array(camera_info.K).reshape(3, 3)

    # detect the aruco marker
    success, w2c = detect_img(img_np, K)

    if not success:
      rospy.logwarn("Fail to detect Aruco Marker")
      return EXECUTION_FAILURE
  
    # get camera link to base link
    try:
      transform = self.tfBuffer.lookup_transform("base_link", img.header.frame_id, rospy.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
      rospy.logwarn("Fail to lookup transform from camera to base_link")
      return EXECUTION_FAILURE

    # get the pose of the aruco marker in the base link
    c2b = np.eye(4)
    c2b[:3, 3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    q = transform.transform.rotation
    q = Quaternion(q.w, q.x, q.y, q.z)
    c2b[:3, :3] = q.rotation_matrix

    # get the pose of the aruco marker in the world frame
    w2b = c2b @ w2c

    return w2b
      
  def get_board_touch_poses(self):
    """ 
    Get Board Touch Poses 
    
    Return:
      poses: List of Poses to Touch
        (x, y, z, qx, qy, qz, qw)
    """
    w2b = self.get_aruco_marker_coord()

    # since this pose is facing downward, we can also take this
    current_pose = self.get_cartesian_pose()
    q_start = Quaternion(current_pose.orientation.w, current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z)

    # sample pose on the board
    board_x = np.linspace(0.5, 7.5, 4) / 100
    board_y = np.linspace(0.5, 7.5, 4) / 100

    board_coord = np.meshgrid(board_x, board_y)
    board_coord = np.array(board_coord).reshape(2, -1).T
    height = 0.05
    board_coord = np.hstack([board_coord, height * np.ones((board_coord.shape[0], 1))])

    # get the pose in the world frame
    board_coord = np.hstack([board_coord, np.ones((board_coord.shape[0], 1))])
    board_coord = w2b @ board_coord.T
    board_coord = board_coord[:3].T

    # up right down towards the board
    # rot = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    # vertice touch pose
    touch_q = q_start

    poses = []
    for coord in board_coord:
      pose = np.array([coord[0], coord[1], coord[2], touch_q.x, touch_q.y, touch_q.z, touch_q.w])
      poses.append(pose)
    
    return poses
  
  def get_mesh(self, service_name:str = "/voxblox_node/generate_mesh", w2b = np.eye(4)):
    # check gen_mesh service is available or not
    service_list = rosservice.get_service_list()
    
    if Open3D_installed:
      mesh = self.integrator.extract_triangle_mesh()
    elif service_name in service_list:
      mesh_filename = "/home/user/Documents/map.ply"
      req = FilePathRequest()
      req.file_path = mesh_filename
      save_mesh_client = rospy.ServiceProxy(service_name, FilePath)
      res = save_mesh_client(req)
      mesh = o3d.io.read_triangle_mesh(mesh_filename)
    else:
      raise NotImplementedError("Service {} is not available".format(service_name))
    
    # TODO remove comment for visualization

    # coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1, origin=[0., 0, 0])
    # coord.transform(w2b)
    # o3d.visualization.draw_geometries([mesh, coord])

    return mesh
  
  def get_tsdf_value(self, points):
    """ Get TSDF Value """
    if Open3D_installed:
      # Open3D TSDF is actually distance * tsdf_trunc_inv
      tsdfs = self.integrator.tsdf_at(points) * self.integrator.sdf_trunc
      return tsdfs
    
    else:
      # query the tsdf at the given point
      req = QueryTSDFRequest()
      for p in points:
        p_ros = Point()
        p_ros.x = p[0]
        p_ros.y = p[1]
        p_ros.z = p[2]
        req.points.append(p_ros)
    
      tsdf_client = rospy.ServiceProxy("/voxblox_node/query_tsdf", QueryTSDF)
      res = tsdf_client(req)
      return res.tsdf
  
  def box_filter(self, vertices, x_min, x_max, y_min, y_max, z_min, z_max):
    """ Box Filter, make sure the marker board is in the region """
    # get the board region here .. if you want to use box filter
    w2b = self.get_aruco_marker_coord()

    vertices_board = w2b[:3, :3].T @ (vertices.T - w2b[:3, [3]])
    vertices_board = vertices_board.T # (N, 3)
    # mesh.vertices = o3d.utility.Vector3dVector(vertices)

    def box_filtering(vertices, x_min, x_max, y_min, y_max, z_min, z_max):
      mask = (vertices[:, 0] > x_min) & (vertices[:, 0] < x_max) & \
             (vertices[:, 1] > y_min) & (vertices[:, 1] < y_max) & \
             (vertices[:, 2] > z_min) & (vertices[:, 2] < z_max)
      return mask

    return box_filtering(vertices_board, x_min, x_max, y_min, y_max, z_min, z_max)
  
  def get_tsdf_touch_poses(self):
    """ 
    Get TSDF Touch Poses 
    
    Return:
      poses: List of Poses to Touch
        (x, y, z, qx, qy, qz, qw)
    """
    # board_center = np.array([0.1, 0.1, 0.06, 1.0])
    # obs_center = w2b @ board_center
    # OBJECT_CENTER = obs_center[:3]

    # get some poses for TSDF integration,
    # comment this with vision phase.
    for i in range(5):
      # always random sa
      candidate_joints, pose_req = self.get_candidate_joints_and_poses(0)
      joints, pose = self.select_starting_view(candidate_joints, pose_req)
 
      # reach the view
      success = self.goto_pose(i, joints, None)
        
      if not success:
        rospy.logwarn("Fail to Reach Joint Angles. Iterating again...")
        continue
    
    mesh = self.get_mesh()

    # filter vertices in Region of Interest
    vertices_ = np.asarray(mesh.vertices)
    vertices_mask = mask_filter(vertices_, self.get_gs_data_dir())
    # vertices_mask = self.box_filter(vertices_, 0, 0.2, 0, 0.2, 0., 0.4)
    
    mesh_faces = np.asarray(mesh.triangles)
    faces = mesh_faces.reshape(-1)

    faces_mask = vertices_mask[faces]
    faces_mask = faces_mask.reshape(-1, 3)
    faces_mask = np.prod(faces_mask, axis=1)

    # select faces based on mask
    filter_faces = mesh_faces[faces_mask > 0]
    
    # Filter the mesh
    mesh.triangles = o3d.utility.Vector3iVector(filter_faces)

    # compute normal
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)

    # random select 100 faces
    select_face_num = 100
    distance_along_normal = 0.03

    face_idx = np.arange(normals.shape[0])
    np.random.shuffle(face_idx)
    select_face = filter_faces[face_idx[:select_face_num]]
    select_normals = normals[face_idx[:select_face_num]]

    # compute the center of the face
    center = np.mean(vertices_[select_face], axis=1)

    # move along the normal direction
    center = center + select_normals * distance_along_normal

    tsdfs = self.get_tsdf_value(center)

    sample_coords = []
    for i in range(select_face_num):
        c_w = center[i]
        tsdf = tsdfs[i]
        
        # filter by tsdf value
        if tsdf <= 0.01 or tsdf > 0.1:
          continue

        x_axis = select_normals[i] * -1

        # take poses only pointing downward
        if x_axis[2] > 0:
            continue

        dummy = np.cross(np.array([0, 0, 1]), x_axis)
        dummy2 = np.cross(np.array([0, 1, 0]), x_axis)
        
        axis = dummy if np.linalg.norm(dummy) > np.linalg.norm(dummy2) else dummy2
        z_axis = np.cross(x_axis, axis)
        if z_axis[2] < 0:
            z_axis = -z_axis
        
        y_axis = np.cross(z_axis, x_axis)
        R = np.array([x_axis, y_axis, z_axis]).T
        u, s, vh = np.linalg.svd(R, full_matrices=True)
        s = np.ones((3, ))
        R = u @ (s[..., None] * vh)

        touch_poses = np.eye(4)
        touch_poses[:3, :3] = R
        touch_poses[:3, 3] = c_w

        ee_poses = self.convert_pose("touch", "base_link", touch_poses)
        joints = self.pose_generator.calcIK(ee_poses) 
        if joints is None:
          continue

        try:
          # plan reaching the pose
          success, trajectory, planning_time, err_code = self.arm_group.plan(joints)
        except:
          success = False
          rospy.logwarn("Fail to Plan Trajectory")

        if success:
          sample_coords.append(touch_poses)

    # visualization 
    original_mesh = self.get_mesh()
    coords = []
    for transform in sample_coords:
      sample_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=.02, origin=[0., 0, 0])
      sample_coord.transform(transform)
      coords.append(sample_coord)
    
    o3d.visualization.draw_geometries([original_mesh, *coords])

    return sample_coords

  def board_demo(self):
    """ Board Demo using Aruco Marker"""
    poses = self.get_board_touch_poses()

    gaussian_splatting_data_dir = "/home/user/Documents/gs_data"
  
    # move to the board
    for touch_pose in poses:
      
      # put a dummy function here
      status = self.touch_pose(touch_pose, cb_func=partial(self.save_touch_data, gaussian_splatting_data_dir))
      if status != SUCCESS and status != DT_THRESHOLD_EXCEED:
        rospy.logwarn("Fail to Touch the Board")
        return status

  def delete_test_result_param(self):
    try:
      rospy.delete_param("/kortex_examples_test_results/moveit_general_python")
    except:
      pass

  def create_view_types_list(self):
    """
    Create a list of view types to add
    """
    self.view_type_ids = []

    for _ in range(self.starting_views):
      self.view_type_ids.append(0)
    for _ in range(self.added_views):
      self.view_type_ids.append(1)

    self.total_views = self.starting_views + self.added_views

  def generate_poses(self):
    """ Generate Poses """

    pose_req = NBVRequest()
    candidate_joints = []

    pose_cnt = 0
    while pose_cnt < self.num_poses:
      pose = self.pose_generator.sampleInSphere(OBJECT_CENTER, 0.3, 0.6)
      joints = self.pose_generator.calcIK(pose) 

      # plan reaching the pose
      success = False
      while not success:
        try:
          success, trajectory, planning_time, err_code = self.arm_group.plan(joints)
        except: 
          success = False
          rospy.logwarn("Fail to Plan Trajectory")
          pose = self.pose_generator.sampleInSphere(OBJECT_CENTER, 0.3, 0.6)
          joints = self.pose_generator.calcIK(pose) 

        # if plan succeeds, we can add the pose as valid
        if success:
          pose_cnt += 1

          pose_msg:PoseStamped = self.convertNumpy2PoseStamped(pose)
          pose_req.poses.append(pose_msg)
          candidate_joints.append(joints)

    return candidate_joints, pose_req.poses
  
  def get_candidate_joints_and_poses(self, i):
    """ Get Candidate Joints and Poses """
    pose_req = NBVRequest()
    if self.exp_poses_available and self.should_collect_experiment:
        candidate_joints, pose_req.poses = self.get_exp_poses_at(i, self.view_type_ids[i])
    else:
      candidate_joints, pose_req.poses = self.generate_poses()

    return candidate_joints, pose_req
  
  def call_nbv(self, pose_req, candidate_joints, sensor_type="rgb"):
    """ 
    Call Next Best View
    
    Args:
      pose_req: NBVRequest the poses are ee link pose
      candidate joints: List of candidate joints corresponding to the poses
      sensor_type: The sensor type to use for NBV
          could be either ["rgb", "touch"]
     """
    rospy.loginfo("Calling NBV...")
    pose_req.sensor = sensor_type
    res: NBVResponse = self.send_req_helper(self.nbv_client, pose_req)
    scores = np.array(res.scores)
    sorted_indices = np.argsort(scores)[::-1]
    return [candidate_joints[i] for i in sorted_indices]

  def select_starting_view(self, candidate_joints, pose_req):
    """ Select Starting View """
    if self.exp_poses_available:
      # follow previous trial init poses
      joints = candidate_joints[0]
      pose = pose_req.poses[0]
    else:
      # use random gen pose
      rand_idx = np.random.randint(0, self.num_poses)
      joints = candidate_joints[rand_idx]
      pose = pose_req.poses[rand_idx]

    return joints, pose
  
  def goto_pose(self, i, joints, sorted_by_score_joints=None):
    joint_configuration_idx = 0
    success = True
    try:
      success &= self.reach_joint_angles(joints)
      if self.view_type_ids[i] == ADDING_VIEW_ID:
        if success:
          rospy.logwarn("Success; reached next view! ")

          # Keep trying to reach the joints, in order of highest score.
          while not success and joint_configuration_idx < len(sorted_by_score_joints):
            joint_configuration_idx += 1
            joints = sorted_by_score_joints[joint_configuration_idx]
            success = self.reach_joint_angles(joints) 
        else:
          rospy.loginfo("Went to Next Best View")
    except:
        rospy.logwarn("Fail to Execute Joint Trajectory")

    return success

  def add_to_experiment_if_needed(self, joints, pose, candidate_joints, i, pose_req):
    if self.should_collect_experiment and not self.exp_poses_available:
      if self.view_type_ids[i] == STARTING_VIEW_ID:
        EXP_POSES["starting_joints"].append(joints)
        EXP_POSES["starting_poses"].append(pose)
      else:
        EXP_POSES["candidate_joints"].append(candidate_joints)
        EXP_POSES["candidate_poses"].append(pose_req.poses)
        
      rospy.loginfo("Writing EXP_POSES")
      # write pickle file
      with open(PICKLE_PATH_FULL, "wb") as f:
        pickle.dump(EXP_POSES, f)

  def call_add_view_client(self):
    req = TriggerRequest()
    rospy.loginfo("Adding View ...")
    res = self.send_req_helper(self.add_view_client, req)
    return res
  
  def update_gs_model(self, success):
    rospy.loginfo("Saving Model with new pose ...")
    req = SaveModelRequest()
    req.success = success
    res = self.send_req_helper(self.save_model_client, req)
    if "Test" in res.message:
      rospy.loginfo("Model is in test mode, exiting gracefully")
      exit()

  def vision_phase(self):
    """ Vision Phase. Starting with a few random views, perform FisherRF to get the next best view """
    import pdb; pdb.set_trace()
    i = 0
    while i < self.total_views:
      candidate_joints, pose_req = self.get_candidate_joints_and_poses(i)

      if self.view_type_ids[i] == STARTING_VIEW_ID:
        joints, pose = self.select_starting_view(candidate_joints, pose_req)

      elif self.view_type_ids[i] == ADDING_VIEW_ID:
        sorted_joints  = self.call_nbv(pose_req, candidate_joints)
        joints = sorted_joints[0]
        
      # reach the view
      if joints is not None:
        sorted_by_score_joints = sorted_joints if self.view_type_ids[i] == ADDING_VIEW_ID else None
        success = self.goto_pose(i, joints, sorted_by_score_joints)
          
        if not success:
          rospy.logwarn("Fail to Reach Joint Angles. Iterating again...")
          continue
        else: 
          self.add_to_experiment_if_needed(joints, pose, candidate_joints, i, pose_req)
          i += 1
        
        self.call_add_view_client()

        if i >= self.starting_views:
          self.update_gs_model(success)

  def convert_pose(self, source_frame, target_frame, pose):
    """ Convert Touch Poses to EE Poses """
    try:
      transform = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
      rospy.logwarn("Fail to lookup transform from {} to {}".format(source_frame, target_frame))
      return np.eye(4)
    
    t2e = np.eye(4)
    t2e[:3, 3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    q = transform.transform.rotation
    q = Quaternion(q.w, q.x, q.y, q.z)
    t2e[:3, :3] = q.rotation_matrix

    # convert the pose to the base link
    # in order x y z qx qy qz qw
    if pose.ndim == 1:
      t2w = np.eye(4)
      t2w[:3, 3] = pose[:3]
      q = Quaternion(pose[6], pose[3], pose[4], pose[5])
      t2w[:3, :3] = q.rotation_matrix
    else:
      t2w = pose

    e2w = t2w @ np.linalg.inv(t2e)

    # convert back to the x y z qx qy qz qw
    e2w_pose = np.zeros(7)
    e2w_pose[:3] = e2w[:3, 3]
    q = Quaternion(matrix=e2w[:3, :3])
    e2w_pose[3:] = np.array([q.x, q.y, q.z, q.w])

    return e2w_pose

  def save_touch_data(self, gaussian_save_dir):
    """ Callback Function to Save Touch Data """

    # create the directory
    os.makedirs(gaussian_save_dir, exist_ok=True)
    os.makedirs(osp.join(gaussian_save_dir, "touch"), exist_ok=True)
    
    # get the current pose
    try:
      transform = self.tfBuffer.lookup_transform("base_link", "touch", rospy.Time())
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
      rospy.logwarn("Fail to lookup transform from touch to base_link")
      return EXECUTION_FAILURE

    # get the pose of the touch sensor in the base link
    c2b = np.eye(4)
    c2b[:3, 3] = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
    q = transform.transform.rotation
    q = Quaternion(q.w, q.x, q.y, q.z)
    c2b[:3, :3] = q.rotation_matrix

    # grab the depth image
    img = rospy.wait_for_message("/RunCamera/imgDepth", Image)
    img_np = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

    # the image_np here is already in uint16 format
    # save the depth image
    image_name = osp.join("touch", "{:04d}.png".format(len(self.touch_json_header["frames"])))
    depth_path = osp.join(gaussian_save_dir, image_name)
    cv2.imwrite(depth_path, img_np)

    c2b_list = [list(row) for row in c2b]

    # save the pose
    self.touch_json_header["frames"].append({
      "file_path": image_name,
      "transformation": c2b_list
    })

    # save the json file
    json_path = osp.join(gaussian_save_dir, "touch", "transforms.json")
    with open(json_path, "w") as f:
      js.dump(self.touch_json_header, f)

    return SUCCESS

  def touch_phase(self, gaussian_splatting_data_dir):
    """ 
    Touch Phase 
    
    At this point, we have set all views. We now continue to the Touch phase.

    1. Generate candidate poses from GS for touch (provided with segmented object in GS)
    2. Send candidate poses to GS to compute the next best touch pose.
    3. Get the next best touch pose and move the robot to that pose for touch.
    4. Get touch data and save it to the GS model. This includes directly injecting Gaussians into the scene and updating the views.
    5. Train model n steps and repeat the process.
    """
    
    # pose generation phase

    # to get touches near the surface, we need esdf volume to sample poses near the surface

    # phase 1. get board pose board 2 baseline
    # TODO 
    touch_poses = self.get_board_touch_poses()

    # convert these poses to EE pose
    ee_poses = list(map(partial(self.convert_pose, "touch", "base_link"), touch_poses))

    # sample uniformly on the board
    # check the feasibility of the touch poses
    pose_req = NBVRequest()
    for pose in ee_poses:
      
      joints = self.pose_generator.calcIK(pose) 

      # plan reaching the pose
      success, trajectory, planning_time, err_code = self.arm_group.plan(joints)
      if success:
        pose_msg:PoseStamped = self.convertNumpy2PoseStamped(pose)
        pose_req.poses.append(pose_msg)

    # call the next best touch pose
    pose_req.sensor_type = "touch"
    res: NBVResponse = self.send_req_helper(self.nbv_client, pose_req)
    scores = np.array(res.scores)
    sorted_indices = np.argsort(scores)[::-1]
    nbt_pose = touch_poses[sorted_indices[0]]
    
    # do the touch
    self.touch_pose(nbt_pose, cb_func=partial(self.save_touch_data, gaussian_splatting_data_dir))


  def run(self):
    """ Run Controller Method to get new views """
    success = self.is_init_success
    self.delete_test_result_param()

    # go home
    if success:
      rospy.loginfo("Reaching Named Target Home...")
      success &= self.reach_named_position("home")
      
    # Board Demo for the touch sensor
    # self.get_tsdf_touch_poses()
    # self.board_demo()

    # Phase 1: Vision
    self.vision_phase()
    
    # self.num_touches_to_add is the amount of touches to add

    # gaussian_splatting_data_dir = self.get_gs_data_dir()
    
    # Phase 2: Touch
    # self.touch_phase('test')

    return success

def main():
  controller = TouchGSController()
  controller.run()
  # rospy.spin()

if __name__ == '__main__':
  main()
