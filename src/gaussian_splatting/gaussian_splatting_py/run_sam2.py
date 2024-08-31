"""
On an RGB image, gets the automatic masks in SAM.

"""
import argparse
from typing import Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2



from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

SAM2_PATH = "/home/user/segment-anything-2/checkpoints/sam2_hiera_large.pt"
MODEL_CFG = "sam2_hiera_l.yaml"

# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def learn_scale_and_offset_raw(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    A = np.vstack([dense_depth_valid, np.ones_like(dense_depth_valid)]).T
    b = sparse_depth_valid

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    scale, offset = x
    return scale, offset

def learn_scale_raw(dense_depth, sparse_depth):
    dense_depth_flat = dense_depth.flatten()
    sparse_depth_flat = sparse_depth.flatten()

    valid_mask = sparse_depth_flat > 0
    dense_depth_valid = dense_depth_flat[valid_mask]
    sparse_depth_valid = sparse_depth_flat[valid_mask]

    # Ensure there are valid values and avoid division by zero
    if np.sum(dense_depth_valid ** 2) == 0 or len(dense_depth_valid) == 0:
        scale = 0
    else:
        # Compute the scale factor directly without an offset term
        scale = np.sum(sparse_depth_valid * dense_depth_valid) / np.sum(dense_depth_valid ** 2)
    
    offset = 0  # Offset is enforced to be zero
    return scale, offset


def warp_image(image, K, R, t):
    """
    Warp an image from the perspective of camera 1 to camera 2.

    :param image: Input image from camera 1
    :param K: Intrinsic matrix of both cameras
    :param R: Rotation matrix from camera 1 to camera 2
    :param t: Translation vector from camera 1 to camera 2
    :return: Warped image as seen from camera 2
    """
    # Compute the homography matrix
    H = compute_homography(K, R, t)

    # Warp the image using the homography
    height, width = image.shape[:2]
    warped_image = cv2.warpPerspective(image, H, (width, height))

    return warped_image


def compute_homography(K, R, t):
    """
    Compute the homography matrix given intrinsic matrix K, rotation matrix R, and translation vector t.
    """
    K_inv = np.linalg.inv(K)
    H = np.dot(K, np.dot(R - np.dot(t.reshape(-1, 1), K_inv[-1, :].reshape(1, -1)), K_inv))
    return H


class SAM2(SAM2AutomaticMaskGenerator):
    def __init__(self, points_per_side=64, pred_iou_thresh=0.95, use_m2m=False):
        sam2 = build_sam2(MODEL_CFG, SAM2_PATH, device ='cuda', apply_postprocessing=False)
        # build sam2 with points and iou threshold
        super().__init__(sam2, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh, use_m2m=use_m2m)
        print("SAM2 initialized.")

    def generate(self, img_path: str, mde_depth_path: str, real_depth: str,
                 is_challenge_object: bool = False, object_mask_path=None,
                 original_mde_depth_path: str = None) -> Union[None, list]:
        """
        Generate the masks for the image.

        Args:
            image: The input image.
            mde_depth: The depth image from monocular depth estimation.
            real_depth: The real depth image.
            mask_object: If not None, the object to mask and include in the background table segmentation

        Returns:
            Aligned depth image.
        """
        image = np.asarray(Image.open(img_path))

        mde_depth = cv2.imread(mde_depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
        real_depth = cv2.imread(real_depth, cv2.IMREAD_UNCHANGED) / 1000.0
        
        # get difference between the two depths' filtered 0s
        sparse_mask = real_depth > 0
        diff = np.mean(np.abs(mde_depth[sparse_mask] - real_depth[sparse_mask]))
        print(f"Diff: {diff}")
        
        results = super().generate(image)
        all_masks = []
        background_mask = np.ones_like(image)
        
        # remove last dim
        background_mask = background_mask[:, :, 0]
        for result in results:
            bool_mask = result['segmentation']
            # convert mas to 1 and 0
            mask = bool_mask.astype(np.uint8)
            # reshape mask to image size
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # perform alignment on mask
            mde_mask = mde_depth[bool_mask]
            real_mask = real_depth[bool_mask]
            scale, offset = learn_scale_and_offset_raw(mde_mask, real_mask)
            
            # update mde
            mde_depth[bool_mask] = mde_depth[bool_mask] * scale + offset
            # remove negative values
            mde_depth[mde_depth < 0] = 0
            background_mask = background_mask * (1 - mask)
        
        background_mask = background_mask.astype(bool)
        
        mde_mask = mde_depth[background_mask]
        real_mask = real_depth[background_mask]
        
        scale, offset = learn_scale_and_offset_raw(mde_mask, real_mask)
        # mde_depth = mde_depth * scale + offset
        
        mde_depth[background_mask] = mde_depth[background_mask] * scale + offset
        mde_depth[mde_depth < 0] = 0
        
        if is_challenge_object:
            # read in old mde depth
            old_mde_depth = cv2.imread(original_mde_depth_path, cv2.IMREAD_UNCHANGED) / 1000.0
            old_mde_mask = old_mde_depth[background_mask]
            
            scale, offset = learn_scale_and_offset_raw(old_mde_mask, real_mask)
            mask = cv2.imread(object_mask_path, cv2.IMREAD_UNCHANGED)
            mask = mask.astype(bool)
            mde_depth[mask] = old_mde_depth[mask] * scale + offset
            mde_depth[mde_depth < 0] = 0
            

        diff = np.mean(np.abs(mde_depth[sparse_mask] - real_depth[sparse_mask]))
        
        img_diff = np.abs(mde_depth - real_depth)
        # set values of sparse mask to 0 in img_diff
        img_diff[~sparse_mask] = 0
        
        print(f"Diff: {diff}")
        print("Masks generated.")
        
        # save the final aligned mde depth and remove negative values
        mde_depth = (mde_depth * 1000).astype(np.uint16)
        
        # get root path from img_path
        root_path = img_path.split("/")[:-1]
        root_path = "/".join(root_path)
        print(f"{root_path}/mde_depth_aligned.png")
        cv2.imwrite(f"{root_path}/mde_depth_aligned.png", mde_depth)
        
        return all_masks
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process an image with depth information.")
    parser.add_argument('--img_path', type=str, required=True, help="Path to the image file")
    parser.add_argument('--mde_depth_path', type=str, required=True, help="Path to the MDE depth file")
    parser.add_argument('--real_depth', type=str, required=True, help="Path to the real depth file")
    parser.add_argument("--is_challenge_object", action="store_true", help="If the object is a challenge object")
    parser.add_argument("--object_mask_path", type=str, help="Path to the object mask")
    parser.add_argument("--original_mde_depth_path", type=str, help="Path to the original MDE depth file")
    
    args = parser.parse_args()
    
    Sam2 = SAM2()
    Sam2.generate(img_path=args.img_path, 
                  mde_depth_path=args.mde_depth_path, 
                  real_depth=args.real_depth,
                  is_challenge_object=args.is_challenge_object,
                  object_mask_path=args.object_mask_path,
                  original_mde_depth_path=args.original_mde_depth_path)