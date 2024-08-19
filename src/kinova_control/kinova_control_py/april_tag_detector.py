import cv2
import numpy as np
import os
import os.path as osp
import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sciR

PREV_POSE_NUM = 5

y_bias = 0.2
# is in cm
tagsCoord = {
    4: np.array([[-4.55, -5.75, 0.], [-4.55, -5.75+3.5, 0], [-4.55+3.5, -5.75+3.5, 0], [-4.55+3.5, -5.75, 0]]),
    1: np.array([[-4.55, -10.3, 0.], [-4.55, -10.3+3.5, 0], [-4.55+3.5, -10.3+3.5, 0],[-4.55+3.5, -10.3, 0]]),
    2: np.array([[-0., -10.3, 0.], [-0., -10.3+3.5, 0], [-0.+3.5, -10.3+3.5, 0],  [-0.+3.5, -10.3, 0]]),
    3: np.array([[4.55, -10.3, 0.], [4.55, -10.3+3.5, 0], [4.55+3.5, -10.3+3.5, 0], [4.55+3.5, -10.3, 0]]),
    5: np.array([[4.55*2, -10.3, 0.], [4.55*2, -10.3+3.5, 0], [4.55*2+3.5, -10.3+3.5, 0], [4.55*2+3.5, -10.3, 0]]),
    # 5: np.array([[4.55*3, -10.3, 0.], [4.55*3+3.5, -10.3, 0], [4.55*3+3.5, -10.3+3.5, 0], [4.55*3, -10.3+3.5, 0]]),
}

def detect_img(img, cameraMatrix):
    # side of the marker is 5 mm
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.markerBorderBits = 1

    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco_dict, parameters=arucoParams)

    if len(corners) < 5:
        print("No markers found in image: ", path)
        return False, None

    # img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 3.5, cameraMatrix, distCoeffs)

    ids = ids.flatten()
    # # compute the pose
    model_points = np.concatenate([tagsCoord[i] for i in ids], axis=0)
    cam_points = np.concatenate(corners, axis=0)
    cam_points = cam_points.reshape(-1, 2)

    # tag_points = np.array([[0, 0, 0], [0, 3.5, 0], [3.5, 3.5, 0], [3.5, 0, 0]], dtype=np.float32)
    # rvet, init_rvecs, init_tvecs = cv2.solvePnP(tag_points, cam_points[:4], cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_AP3P)
    
    model_points[:, 1] += y_bias
    model_points = model_points / 100 # convert to meters
    distCoeffs = np.zeros((8,))
    rvet, rvecs, tvecs = cv2.solvePnP(model_points, cam_points, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE) #, rvec=init_rvecs, tvec=init_tvecs, useExtrinsicGuess=True)

    rvecs = rvecs.reshape(-1)
    tvecs = tvecs.reshape(-1)

    rot = cv2.Rodrigues(rvecs)
    w2c = np.eye(4)
    w2c[:3, :3] = rot[0]
    w2c[:3, 3] = tvecs

    return True, w2c

if __name__ == "__main__":
    imgs = glob.glob("images/*.JPG")
    imgs.sort()
    print("Images found: ", len(imgs))

    # side of the marker is 5 mm
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters()
    arucoParams.markerBorderBits = 1

    cameraMatrix = np.array([[3099.37, 0.0, 2016.0],
                             [0.0, 3099.37, 1512.0],
                             [0.0, 0.0, 1.0]])
    distCoeffs = np.zeros((8,))

    camera2tags = {}
    cv2.namedWindow('img', cv2.WINDOW_GUI_NORMAL)
    for idx, path in tqdm(enumerate(imgs), desc="Processing images"):
        img = cv2.imread(path)
        
        if img.shape[0] > img.shape[1]:
            img=cv2.transpose(img)
            img=cv2.flip(img,flipCode=0)

        ret, w2c = detect_img(img, cameraMatrix)

        c2w = np.linalg.inv(w2c)
        camera2tags[idx + 1] = c2w

        # draw the coordinate axis here (Uncomment to see the axis)
        tvecs = c2w[:3, 3]
        rvecs = cv2.Rodrigues(c2w[:3, :3])[0]
        img = cv2.drawFrameAxes(img, cameraMatrix, distCoeffs, rvecs, tvecs, 10)
        cv2.imshow('img', img)
        cv2.waitKey(2)
    
    cv2.destroyAllWindows()