#!/usr/bin/env python3

import os
import datetime
import os.path as osp
import numpy as np
import cv2
import json
from typing import List, Union

import tf2_ros as tf2
from scipy.spatial.transform import Rotation as sciR

import threading
import rospy
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose, Transform, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from gaussian_splatting.srv import NBV, NBVResponse, NBVRequest

class VisionNode(object):

    def __init__(self) -> None:
        rospy.init_node("vision_node")

        # fetch parameter
        self.CAMERA_TOPC = rospy.get_param("~image_topic", "/image")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/camera_info")
        self.save_data = rospy.get_param("~save_data", "False")
        self.save_data_dir = rospy.get_param("~save_data_dir", "/home/user/Documents/data")
        rospy.loginfo("Camera Topic: {}".format(self.CAMERA_TOPC))
        rospy.loginfo("Save Data: {}".format(self.save_data))
        rospy.loginfo("Save Data Dir: {}".format(self.save_data_dir))

        # add service
        rospy.loginfo("Adding services")
        self.addview_srv = rospy.Service("add_view", Trigger, self.addVisionCb)
        self.nbv_srv = rospy.Service("next_best_view", NBV, self.NextBestView)
        self.savemodel_srv = rospy.Service("save_model", Trigger, self.saveModelCb)

        # wait for image topic
        rospy.loginfo("Waiting for camera topic")
        rospy.wait_for_message(self.CAMERA_TOPC, Image)

        # store images in (H, W, 3) [0 - 255]
        self.images: List[np.ndarray] = []
        # store original camera to world pose 
        self.poses: List[np.ndarray] = []

        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)

        # TODO initialize the radiance field


        # training thread
        self.training_thread = threading.Thread(target=self.training_loop)

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
    
    def saveModelCb(self, req) -> TriggerResponse:
        """ Save Model Callback """
        res = TriggerResponse()
        res.success = True

        if self.training_thread.is_alive():
            res.success = False
            res.message = "Error Code 1: Training thread is still running"
            return res

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

        if self.training_thread.is_alive():
            res.success = False
            res.message = "Error Code 1: Training thread is still running"
            return res

        # grap the image message
        img:Image = rospy.wait_for_message(self.CAMERA_TOPC, Image)

        # convert to numpy array
        # process the image
        height = img.height
        width = img.width
        encoding = img.encoding

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

            # start the training thread
            self.training_thread = threading.Thread(target=self.training_loop)
            self.training_thread.start()

            res.message = "Success"
        
        return res

    def NextBestView(self, req:NBVRequest) -> NBVResponse:
        """ Next Best View Service Callback """
        poses = req.poses
        res = NBVResponse()

        if len(poses) == 0:
            res.success = True
            res.message = "Success"
            return None

        if self.training_thread.is_alive():
            res.success = False
            res.message = "Error Code 1: Training thread is still running"
            return res

        # convert to numpy array
        poses_np = np.array([VisionNode.convertPose2Numpy(pose) for pose in poses]) # (N, 4, 4)
        scores = self.EvaluatePoses(poses_np)

        # return response
        res.success = True
        res.message = "Success"
        res.scores = list(scores)
        return res
    
    def save_images(self):
        """ Save the captured images as a NeRF Synthetic Dataset format """
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
            json.dump(json_txt, f)

    def training_loop(self):
        """ Train the Radiance Field """
        rospy.sleep(5)
        pass

    def EvaluatePoses(self, poses:np.ndarray) -> np.ndarray:
        """ Evaluate poses  """
        
        # TODO update to query GS to get scores for each pose
        return np.random.rand(poses.shape[0])

    def saveModel(self):
        """ Save the model  """
        
        pass


if __name__ == "__main__":
    node = VisionNode()
    rospy.spin()