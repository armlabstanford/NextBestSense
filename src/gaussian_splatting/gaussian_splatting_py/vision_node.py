#!/usr/bin/env python3

import numpy as np
from typing import List, Union

import tf2
from scipy.spatial.transform import Rotation as sciR

import threading
import rospy
from std_srvs.srv import Trigger, TriggerResponse, TriggerRequest
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Image
from gaussian_splatting.srv import NBVResponse, NBVRequest

class VisionNode(object):
    CAMERA_TOPC = "image"

    def __init__(self) -> None:
        rospy.init_node("vision_node")

        # add service
        rospy.loginfo("Adding services")
        self.addview_srv = rospy.Service("add_view", Trigger, self.addVisionCb)
        self.nbv_srv = rospy.Service("next_best_view", NBVRequest, self.NextBestView)

        # wait for image topic
        rospy.loginfo("Waiting for camera topic")
        rospy.wait_for_message(self.CAMERA_TOPC, Image)

        # store images in (H, W, 3) [0 - 255]
        self.images:List[np.ndarray] = []
        # store original camera to world pose 
        self.poses:List[np.ndarray] = []

        self.tfBuffer = tf2.Buffer()
        self.listener = tf2.TransformListener(self.tfBuffer)

        # initialize the radiance field


        # training thread
        self.training_thread = threading.Thread(target=self.training_loop)

    def convertPose2Numpy(pose:Union[PoseStamped, Pose]) -> np.ndarray:
        if isinstance(pose, PoseStamped):
            pose = pose.pose

        c2w = np.eye(4)
        quat = sciR.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        c2w[:3, :3] = quat.as_matrix()

        # position
        c2w[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        return c2w

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
            self.listener.waitForTransform("base_link", "camera", rospy.Time(), rospy.Duration(1))
            pose:PoseStamped = self.tfBuffer.lookup_transform("base_link", "camera", rospy.Time())
        except:
            rospy.logerr("Failed to lookup transform from camera to base_link")
            res.success = False
            res.message = "Error Code 3: Failed to lookup transform from camera to base_link"

        if res.success:
            # convert to numpy array
            # process the pose
            c2w = VisionNode.convertPose2Numpy(pose)

            self.poses.append(c2w)
            self.images.append(img_np)

            # start the training thread
            self.training_thread.start()

            res.message = "Success"
        
        return res
    
    def training_loop(self):
        """ Train the Radiance Field """
        pass

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


    def EvaluatePoses(self, poses:np.ndarray) -> np.ndarray:
        pass


if __name__ == "__main__":
    node = VisionNode()
    rospy.spin()