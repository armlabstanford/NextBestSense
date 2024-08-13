#!/usr/bin/env python3

import os
import datetime
import os.path as osp
from matplotlib import pyplot as plt
import numpy as np
import cv2
import json
from typing import List, Union


import tf2_ros as tf2
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as sciR

import rospy

from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
from sensor_msgs.msg import Image, CameraInfo

from gaussian_splatting_py.splatfacto3d import Splatfacto3D as splatfacto

from gaussian_splatting_py.vision_utils.vision_utils import convert_intrinsics, learn_scale_and_offset_raw, warp_image
from gaussian_splatting_py.load_yaml import load_config
from gaussian_splatting_py.monocular_depth import MonocularDepth

from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest, NBVPoses, NBVPosesResponse, NBVPosesRequest, NBVResult, NBVResultRequest, NBVResultResponse, SaveModel, SaveModelRequest, SaveModelResponse

class VisionNode(object):

    def __init__(self) -> None:
        rospy.init_node("vision_node")
        rospy.loginfo("Vision Node Initializing")

        # fetch parameter
        self.CAMERA_TOPIC = rospy.get_param("~image_topic", "/image")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/camera_info")
        
        # depth cam topic and cam info
        self.DEPTH_CAMERA_TOPIC = rospy.get_param("~depth_image_topic", "/image")
        self.depth_cam_info_topic = rospy.get_param("~depth_cam_info_topic", "/camera_info")
        
        self.save_data = rospy.get_param("~save_data", "True")
        self.save_data_dir = rospy.get_param("~save_data_dir", "/home/user/NextBestSense/data")
        self.gs_data_dir = rospy.get_param("~gs_data_dir", "/home/user/touch-gs-data/bunny_blender_data")
        
        # GS model
        self.gs_training = False
        self.gs_model = splatfacto(data_dir=self.gs_data_dir)
        
        # construct monocular depth model
        param_filename = osp.join(osp.dirname(osp.abspath(__file__)), "config.yml") 
        self.monocular_depth = MonocularDepth(load_config(param_filename))
        
        rospy.loginfo("Camera Topic: {}".format(self.CAMERA_TOPIC))
        rospy.loginfo("Depth Camera Topic: {}".format(self.DEPTH_CAMERA_TOPIC))
        rospy.loginfo("Save Data: {}".format(self.save_data))
        rospy.loginfo("Save Data Dir: {}".format(self.save_data_dir))

        # add service
        rospy.loginfo("Adding services")
        self.addview_srv = rospy.Service("add_view", Trigger, self.addVisionCb)
        
        self.nbv_srv = rospy.Service("next_best_view", NBV, self.NextBestView)
        self.nbv_get_poses_srv = rospy.Service("get_poses", NBVPoses, self.getNBVPoses)
        self.nbv_get_poses_srv = rospy.Service("receive_nbv_scores", NBVResult, self.receiveNBVScoresGS)
        self.savemodel_srv = rospy.Service("save_model", SaveModel, self.saveModelCb)
        
        # wait for image topic
        rospy.loginfo("Waiting for camera topic")
        rospy.wait_for_message(self.CAMERA_TOPIC, Image)
        rospy.loginfo("Camera topic found")
        
        # wait for depth image topic
        rospy.loginfo("Waiting for depth camera topic")
        rospy.loginfo(self.DEPTH_CAMERA_TOPIC)
        
        rospy.wait_for_message(self.DEPTH_CAMERA_TOPIC, Image)
        rospy.loginfo("Depth Camera topic found")
        self.bridge = CvBridge()
        
        # get camera infos 
        rospy.loginfo("Waiting for camera info")
        self.color_cam_info: CameraInfo = rospy.wait_for_message(self.cam_info_topic, CameraInfo)
        self.depth_cam_info: CameraInfo = rospy.wait_for_message(self.depth_cam_info_topic, CameraInfo)

        # store images in (H, W, 3) [0 - 255]
        self.images: List[np.ndarray] = []
        # store depth images in (H, W) [0 - max val of uint16]
        self.depths: List[np.ndarray] = []
        # store original camera to world pose 
        self.poses: List[np.ndarray] = []
        # store poses for next best view to send to GS
        self.poses_for_nbv: List[np.ndarray] = []
        
        self.scores: List[float] = []

        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)
        
        self.idx = 0
        
        self.data_base_dir = None
        self.gs_training_dir = None
        self.only_generate_test_views = False

        rospy.loginfo("Vision Node Initialized")
        

    def convertPose2Numpy(pose:Union[PoseStamped, Pose]) -> np.ndarray:
        if isinstance(pose, PoseStamped):
            pose = pose.pose

        c2w = np.eye(4)
        quat = sciR.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        return c2w
    
    def convertTransform2Numpy(transform:Union[TransformStamped, Transform]) -> np.ndarray:
        if isinstance(transform, TransformStamped):
            transform = transform.transform

        c2w = np.eye(4)
        quat = sciR.from_quat([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]

        return c2w
    
    def convertNumpy2PoseStamped(c2w:np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = c2w[0, 3]
        pose.pose.position.y = c2w[1, 3]
        pose.pose.position.z = c2w[2, 3]
        
        quat = sciR.from_matrix(c2w[:3, :3])
        pose.pose.orientation.x = quat.as_quat()[0]
        pose.pose.orientation.y = quat.as_quat()[1]
        pose.pose.orientation.z = quat.as_quat()[2]
        pose.pose.orientation.w = quat.as_quat()[3]
        
        return pose
    
    def saveModelCb(self, req) -> SaveModelResponse:
        """ Save Model Callback """
        res = SaveModelResponse()
        res.success = req.success
        
        if self.save_data:
            gs_data_dir = self.save_images()
            
        if self.only_generate_test_views:
            res.message = "Test views done."
            return res
        
        if self.gs_training:
            rospy.wait_for_service('continue_training')
            rospy.loginfo("Calling continue_training service")
            try:
                continue_training_srv = rospy.ServiceProxy('continue_training', Trigger)
                request = TriggerRequest()
                response = continue_training_srv(request)
                if response.success:
                    rospy.loginfo(f"Successfully called continue_training service")
                else:
                    rospy.logfatal("Failed to call continue_training service")
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed with error {e}")
            
        rospy.loginfo("Saving Model...")
        
        # start training model
        if not self.gs_training:
            self.saveModel(gs_data_dir)
            
        
        res.message = "Success"
        return res
    
    
    def align_depth(self, depth: np.ndarray, predicted_depth: np.ndarray, 
                    rgb: np.ndarray, use_sam: bool = False) -> np.ndarray:
        scale, offset = learn_scale_and_offset_raw(predicted_depth, depth)
        depth_np = (scale * predicted_depth) + offset
        
        # perform SAM2 semantic alignment if use_sam is True
        if use_sam:
            # call SAM2 process in python3.11. Save the image, depth, and predicted depth
            img_path = osp.join(self.save_data_dir, "sam2_img.png")
            depth_path = osp.join(self.save_data_dir, "sam2_depth.png")
            mde_depth_path = osp.join(self.save_data_dir, "sam2_mde_depth.png")
            
            cv2.imwrite(img_path, rgb)
            depth = (depth * 1000).astype(np.uint16)
            predicted_depth = (predicted_depth * 1000).astype(np.uint16)
            cv2.imwrite(depth_path, depth)
            cv2.imwrite(mde_depth_path, predicted_depth)
            
            # hack with python3.11 to run SAM2 depth alignment
            os.system(f"python3.11 /home/user/NextBestSense/src/gaussian_splatting/gaussian_splatting_py/run_sam2.py --img_path {img_path} --real_depth {depth_path} --mde_depth_path {mde_depth_path}")
            
            # read from mde_depth_aligned.png
            depth_np = cv2.imread(osp.join(self.save_data_dir, "mde_depth_aligned.png"), cv2.IMREAD_UNCHANGED) / 1000.0
        
        # remove bad values
        depth_np[depth_np < 0] = 0
        
        rospy.loginfo(f"Scale: {scale}, Offset: {offset}")
        return depth_np

    def addVisionCb(self, req) -> TriggerResponse:
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
        rospy.sleep(5)

        # grab the image message
        img: Image = rospy.wait_for_message(self.CAMERA_TOPIC, Image)
        img_np = self.bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

        # grab the depth image message
        depth_img: Image = rospy.wait_for_message(self.DEPTH_CAMERA_TOPIC, Image)
        depth = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough") / 1000.0
        # if depth value is > 20, set to 0
        depth[depth > 20] = 0
        
        new_intrinsics = np.array(self.color_cam_info.K)
        new_intrinsics_tup = (new_intrinsics[0], new_intrinsics[4], new_intrinsics[2], new_intrinsics[5])
        
        # convert instrinsics
        depth = convert_intrinsics(depth, new_size=(self.color_cam_info.width, self.color_cam_info.height), new_intrinsics=new_intrinsics_tup)
        
        try:
            cam2cam_transform: TransformStamped = self.tfBuffer.lookup_transform("camera_color_frame", "camera_depth_frame", rospy.Time())
        except Exception as e:
            rospy.logerr("Failed to lookup transform from camera to base_link")
            res.success = False
            res.message = "Error Code 3: Failed to lookup transform from camera to base_link"
        cam2cam_transform = VisionNode.convertTransform2Numpy(cam2cam_transform)
        K = np.array(self.depth_cam_info.K).reshape(3, 3)
        
        
        realsense_depth = warp_image(depth, K, cam2cam_transform[:3, :3], cam2cam_transform[:3, 3])
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        # run MDE to get the depth image
        output = self.monocular_depth(img_np)
        predicted_depth = output['depth']
        
        # align depths 
        depth_np = self.align_depth(realsense_depth, predicted_depth, img_np, use_sam=True)
        
        # save fig of the depth image
        full_mde_path = osp.join(self.save_data_dir, f"mde_depth_img{self.idx}.png")
        full_rs_path = osp.join(self.save_data_dir, f"rs_depth_img{self.idx}.png")
        self.idx += 1
        
        plt.figure()
        plt.imshow(depth_np, cmap='viridis')
        plt.colorbar(label='MDE Depth')
        plt.imsave(full_mde_path, depth_np, cmap='viridis')
        
        plt.figure()
        plt.imshow(realsense_depth, cmap='viridis')
        plt.colorbar(label='Realsense Depth')
        plt.imsave(full_rs_path, realsense_depth, cmap='viridis')
        cv2.imwrite(full_rs_path, (realsense_depth * 1000).astype(np.uint16))
        
        try:
            transform: TransformStamped = self.tfBuffer.lookup_transform("base_link", "camera_link", rospy.Time())
        except Exception as e:
            rospy.logerr(f"Failed to lookup transform from camera to base_link: {e}")
            res.success = False
            res.message = "Error Code 3: Failed to lookup transform from camera to base_link"
            
        if res.success:
            c2w = VisionNode.convertTransform2Numpy(transform)
            self.poses.append(c2w)
            self.images.append(img_np)
            self.depths.append(depth_np)
            
            res.message = "Success"
            rospy.loginfo(f"Added view to the dataset with {len(self.images)} images")
        return res
    
    def getNBVPoses(self, req: NBVPosesRequest) -> NBVPosesResponse:
        """ Get Next Best View Poses to send to GS

        Args:
            req (NBVPosesRequest): Request to get the next best view poses
        Returns:
            NBVPosesResponse: _description_
        """
        response = NBVPosesResponse()
        
        if len(self.poses_for_nbv) == 0:
            # if there are no poses for next best view, return false so that GS continues training as is.
            response.success = False
            response.message = "No poses for Next Best View"
            return response
         
        response.success = True
        response.message = f"Sent {len(self.poses_for_nbv)} poses for Next Best View to GS."
        response.poses = self.poses_for_nbv
        
        return response
    
    def receiveNBVScoresGS(self, req: NBVResultRequest) -> NBV:
        """ 
        Receive the NBV Scores from GS 
        
        """
        response = NBVResultResponse()
        if len(req.scores) == 0:
            response.success = False
            response.message = "No scores provided. GS will continue training."
            return response
        
        response.success = True
        response.message = "Successfully received scores from GS"
        self.scores = req.scores
        self.done = True
        return response

    def NextBestView(self, req:NBVRequest) -> NBVResponse:
        """ Next Best View Service Callback 
            Waits for GS to be trained, then proceeds
        """
        poses = req.poses
        res = NBVResponse()

        if len(poses) == 0:
            res.success = True
            res.message = "No-Op -- No poses provided"
            return None
        
        scores = self.EvaluatePoses(poses)

        # return response
        res.success = True
        res.message = "Success"
        res.scores = list(scores)
        return res
    
    def save_images(self):
        """ Save the captured images as a NeRF Synthetic Dataset format
            When new views exist, """
        # get camera info
        # get the date format in Year-Month-Day-Hour-Minute-Second
        
        if self.gs_training_dir is None:
        
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d-%H-%M-%S")

            data_base_dir = osp.join(self.save_data_dir, date_str)
            os.makedirs(data_base_dir, exist_ok=True)
            os.makedirs(osp.join(data_base_dir, "images"), exist_ok=True)
            self.gs_training_dir = data_base_dir
        

        cam_info:CameraInfo = rospy.wait_for_message(self.cam_info_topic, CameraInfo)

        cam_K = np.array(cam_info.K).reshape(3, 3)
        fovx = 2 * np.arctan(cam_K[0, 2] / cam_K[0, 0])
        fovy = 2 * np.arctan(cam_K[1, 2] / cam_K[1, 1])
        focal_x = cam_K[0, 0]
        focal_y = cam_K[1, 1]

        cam_height = cam_info.height
        cam_width = cam_info.width

        json_txt = {
            "w": cam_width,
            "h": cam_height,
            "fl_x": focal_x,
            "fl_y": focal_y,
            "cx": cam_K[0, 2],
            "cy": cam_K[1, 2],
            "camera_angle_x": fovx,
            "camera_angle_y": fovy,
            "frames": [],
        }

        for img_idx, (pose, image, depth) in enumerate(zip(self.poses, self.images, self.depths)):
            # save the image
            image_path = osp.join(self.gs_training_dir, "images", "{:04d}.png".format(img_idx))
            depth_path = osp.join(self.gs_training_dir, "images", "{:04d}_depth.png".format(img_idx))
            
            depth = (depth * 1000).astype(np.uint16)

            cv2.imwrite(image_path, image)
            cv2.imwrite(depth_path, depth)

            # save the pose
            pose_list = pose.tolist()
            pose_info = {
                "file_path": osp.join("images", "{:04d}.png".format(img_idx)),
                "depth_file_path": osp.join("images", "{:04d}_depth.png".format(img_idx)),
                "mask_path": osp.join("masks", "{:04d}.png".format(img_idx)),
                "transform_matrix": pose_list,
            }
            json_txt["frames"].append(pose_info)

        # dump to json file
        json_file = osp.join(self.gs_training_dir, "transforms.json")
        with open(json_file, "w") as f:
            json.dump(json_txt, f, indent=4)
            
        rospy.loginfo(f"Saved all images to {self.gs_training_dir}. Now generating masks in SAM2...")
        
        # construct sam2 masks
        os.system(f"python3.11 /home/user/NextBestSense/src/gaussian_splatting/gaussian_splatting_py/frames_sam2.py --data_dir {self.gs_training_dir}")
        
        return self.gs_training_dir
    
    def invertTransform(self, transform:np.ndarray) -> np.ndarray:
        """ Invert the transformation matrix """
        inv_transform = np.eye(4)
        inv_transform[:3, :3] = transform[:3, :3].T
        inv_transform[:3, 3] = -inv_transform[:3, :3] @ transform[:3, 3]
        return inv_transform

    def EvaluatePoses(self, poses:List[PoseStamped]) -> np.ndarray:
        """ 
        Evaluate poses. Waits a few minutes for GS to reach 2k steps, then requests a pose from GS with FisherRF
        """
        self.done = False
        
        cam_poses: List[PoseStamped] = []
        
        # compute transform from EE to camera
        try:
            transform: TransformStamped = self.tfBuffer.lookup_transform("end_effector_link", "camera_link", rospy.Time())
        except Exception as e:
            rospy.logerr(f"Failed to lookup transform from camera to end_effector_link: {e}")
            return None
        
        for pose in poses:
            try:
                # convert to pose stamped of the cam pose given the EE pose.
                pose.pose.position.x += transform.transform.translation.x
                pose.pose.position.y += transform.transform.translation.y
                pose.pose.position.z += transform.transform.translation.z

                # get transformation matrix from pose (4 x 4)
                cam_pose = VisionNode.convertPose2Numpy(pose)
                
                # convert to pose
                new_pose = VisionNode.convertNumpy2PoseStamped(cam_pose)
                cam_poses.append(new_pose)
            except Exception as e:
                rospy.logerr(f"Failed to transform pose: {e}")
                return None
        
        self.poses_for_nbv = cam_poses
        self.scores = []
        
        rate = rospy.Rate(1)  # 1 Hz
        # loop until GS hits 2k steps and requests a pose
        while not rospy.is_shutdown() and not self.done:
            rospy.loginfo("GS running...")
            rate.sleep()
        rospy.loginfo("GS Done")
        
        # result is obtained, return the scores, which should be populated
        return np.array(self.scores)

    def saveModel(self, gs_data_dir: str):
        """ Save the model  """
        # send request to NS to continue training
        self.gs_training = True
        
        # call training in data_dir
        self.gs_model.start_training(gs_data_dir)
        


if __name__ == "__main__":
    node = VisionNode()
    rospy.spin()