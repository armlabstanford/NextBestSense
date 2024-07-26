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
from scipy.spatial.transform import Rotation as sciR

import rospy

from cv_bridge import CvBridge
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
# from gaussian_splatting.gaussian_splatting_py.splatfacto3d import Splatfacto3D
from gaussian_splatting_py.splatfacto3d import Splatfacto3D as splatfacto
from gaussian_splatting_py.vision_utils.vision_utils import convert_intrinsics, warp_image
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
        
        self.save_data = rospy.get_param("~save_data", "False")
        self.save_data_dir = rospy.get_param("~save_data_dir", "/home/user/NextBestSense/data")
        self.gs_data_dir = rospy.get_param("~gs_data_dir", "/home/user/touch-gs-data/bunny_blender_data")
        
        # GS model
        self.gs_training = False
        self.gs_model = splatfacto(data_dir=self.gs_data_dir)
        
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
        # store original camera to world pose 
        self.poses: List[np.ndarray] = []
        # store poses for next best view to send to GS
        self.poses_for_nbv: List[np.ndarray] = []
        
        self.scores: List[float] = []

        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)

        # TODO initialize the radiance field

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
    
    def saveModelCb(self, req) -> SaveModelResponse:
        """ Save Model Callback """
        res = SaveModelResponse()
        res.success = req.success
        
        rospy.loginfo("Saving Model")

        if self.save_data:
            self.save_images()

        # save the model
        self.saveModel()

        res.message = "Success"
        return res

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

        # grab the image message
        img: Image = rospy.wait_for_message(self.CAMERA_TOPIC, Image)

        # grab the depth image message
        depth_img: Image = rospy.wait_for_message(self.DEPTH_CAMERA_TOPIC, Image)
        depth_img = self.bridge.imgmsg_to_cv2(depth_img, desired_encoding="passthrough")
        
        # get raw depth value
        depth = np.clip(depth_img, 0, 3000) / 3000
        
        # convert depth image to color frame with known transform and camera intrinsics
        print(self.depth_cam_info)
        print(self.color_cam_info)
        
        # convert instrinsics
        depth = convert_intrinsics(depth, new_size=(self.color_cam_info.width, self.color_cam_info.height))
        K = np.array(self.color_cam_info.K).reshape(3, 3)
        
        try:
            # self.listener.waitForTransform("base_link", "camera_color_frame", rospy.Time(), rospy.Duration(1))
            cam2cam_transform: TransformStamped = self.tfBuffer.lookup_transform("camera_color_frame", "camera_depth_frame", rospy.Time())
        except Exception as e:
            rospy.logerr("Failed to lookup transform from camera to base_link")
            res.success = False
            res.message = "Error Code 3: Failed to lookup transform from camera to base_link"
            
        cam2cam_transform = VisionNode.convertTransform2Numpy(cam2cam_transform)
        # final warped depth to directly align with color image
        depth = warp_image(depth, K, cam2cam_transform[:3, :3], cam2cam_transform[:3, 3])

        # process the image
        height = img.height
        width = img.width
        encoding = img.encoding
        
        rospy.loginfo("Image Encoding: {}".format(encoding))

        if encoding == "rgb8":
            img_np = np.frombuffer(img.data, dtype=np.uint8).reshape((height, width, 3))
        elif encoding == "rgb16":
            img_np = np.frombuffer(img.data, dtype=np.uint16).reshape((height, width, 3))
        elif encoding == "rgba8":
            img_np = np.frombuffer(img.data, dtype=np.uint8).reshape((height, width, 4))
        elif encoding == "rgba16":
            img_np = np.frombuffer(img.data, dtype=np.uint16).reshape((height, width, 4))
        elif encoding == "mono8":
            img_np = np.frombuffer(img.data, dtype=np.uint8).reshape((height, width, 1))
        elif encoding == "mono16":
            img_np = np.frombuffer(img.data, dtype=np.uint16).reshape((height, width, 1))
        else:
            rospy.logerr("Unsupported image encoding: {}".format(encoding))
            res.success = False
            res.message = "Error Code 2: Unsupported image encoding: {}".format(encoding)

        # loop up the transform from camera to base_link
        try:
            # self.listener.waitForTransform("base_link", "camera_color_frame", rospy.Time(), rospy.Duration(1))
            transform:TransformStamped = self.tfBuffer.lookup_transform("base_link", "camera_color_frame", rospy.Time())
        except Exception as e:
            rospy.logerr("Failed to lookup transform from camera to base_link")
            res.success = False
            res.message = "Error Code 3: Failed to lookup transform from camera to base_link"
        if res.success:
            # convert to numpy array
            # process the pose
            c2w = VisionNode.convertTransform2Numpy(transform)

            self.poses.append(c2w)
            self.images.append(img_np)
            
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
        
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d-%H-%M-%S")

        data_base_dir = osp.join(self.save_data_dir, date_str)
        os.makedirs(data_base_dir, exist_ok=True)
        os.makedirs(osp.join(data_base_dir, "images"), exist_ok=True)

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

        for img_idx, (pose, image) in enumerate(zip(self.poses, self.images)):
            # save the image
            image_path = osp.join(data_base_dir, "images", "{:04d}.png".format(img_idx))

            image = image[:, :, ::-1] # RGB to BGR
            cv2.imwrite(image_path, image)

            # save the pose
            pose_list = [list(row) for row in pose]
            pose_info = {
                "file_path": osp.join("images", "{:04d}.png".format(img_idx)),
                "transform_matrix": pose_list,
            }

            json_txt["frames"].append(pose_info)

        # dump to json file
        json_file = osp.join(data_base_dir, "transforms.json")
        with open(json_file, "w") as f:
            json.dump(json_txt, f, indent=4)
            
        rospy.loginfo(f"Saved images to {data_base_dir}")

    def EvaluatePoses(self, poses:List[PoseStamped]) -> np.ndarray:
        """ 
        Evaluate poses. Waits a few minutes for GS to reach 2k steps, then requests a pose from GS with FisherRF
        """
        self.done = False
        self.poses_for_nbv = poses
        self.scores = []
        rate = rospy.Rate(1)  # 1 Hz
        # loop until GS hits 2k steps and requests a pose
        while not rospy.is_shutdown() and not self.done:
            rospy.loginfo("GS running...")
            rate.sleep()
            
        # result is obtained, return the scores
        return np.array(self.scores)

    def saveModel(self):
        """ Save the model  """
        # send request to NS to continue training
        self.gs_training = True
        
        self.gs_model.start_training()


if __name__ == "__main__":
    node = VisionNode()
    rospy.spin()