"""
ROS Splatfacto abstract class
ARMLab 2024
"""
from abc import ABC, abstractmethod
import subprocess

import rospy

# future potential work: send poses to select for next best pose, resample around the top k poses to see if we can get a better view

class ROSSplatfacto(ABC):
    """ 
    3D GS model from Nerfstudio. Connects to ROS1
    """
    def __init__(self, data_dir='bunny_blender_dir', render_uncertainty=False,
                 train_split_fraction=0.5, depth_uncertainty_weight=1.0, rgb_uncertainty_weight=1.0):
        """
        initialize Gaussian Splatting
        When the model trains, it will train to 2K steps, then a service will be called to get the list of poses. [GS waits]
        
        The feasible poses will be sent back to GS, and the next best view will be selected.
        
        This view will be sent to the pose for the robot to go. [GS waits]
        
        When add view service is called, we take the current image, add it to the transforms.json file, and save the image. [GS waits]
        
        Then we resume GS training.
        
        """
        self.data_dir = data_dir
        self.render_uncertainty = render_uncertainty
        self.train_split_fraction = train_split_fraction
        self.depth_uncertainty_weight = depth_uncertainty_weight
        self.rgb_uncertainty_weight = rgb_uncertainty_weight
        
        rospy.loginfo("Adding GS services")
        # service for Nerfstudio to get the list of poses
        self.get_
        
        
    
    @abstractmethod
    def start_training(steps=15000):
        """
        Start training Gaussian Splatting. Runs ns-train depth-splatfacto3d or depth-splatfacto2d
        """
        pass
