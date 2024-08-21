#!/usr/bin/env python3

"""
Touch Node framework for Gaussian Splatting.
"""

class TouchNode(object):
    def __init__(self) -> None:
        # vision node framework
        pass

    def add_touch(self, touch: Touch) -> None:
        # add a touch point cloud to GS
        pass

    def get_segmented_object(self):
        """
        Gets the segmented object from Nerfstudio for Touch.
        """
        pass

    def send_touch_poses_to_gs(self):
        """
        Send touch poses to Gaussian Splatting for next best touch.
        """
        pass

    def receive_touch_poses_scores_from_gs(self):
        """
        Receive touch pose scores from Gaussian Splatting.
        """
        pass

    def get_next_best_touch(self, touch_poses):
        """
        Get the next best touch pose from Gaussian Splatting.

        Returns: scores of touches
        """
        pass