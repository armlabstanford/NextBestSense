#! /usr/bin/python3

import sys
import os, re

from cv_bridge import CvBridge, CvBridgeError
from concurrent.futures import thread
import cv2
import numpy as np
import threading
import time
import os
import yaml
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import WrenchStamped, Vector3

from timeit import default_timer as timer

### model setting 
import torch
from DenseTact.Img2Depth.img2depthforce import getDepth, getForce
from DenseTact.Img2Depth.networks.DenseNet import DenseDepth
from DenseTact.Img2Depth.networks.STForce import DenseNet_Force


from skimage.metrics import structural_similarity as compare_ssim

import rospy

def convertSpherical_np_cam(xyz):
    """
    convert x, y, z into r, phi, theta where theta is defined in xy plane
    This ftn assumes that the y component has been flipped.
    """
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    new_pts = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    new_pts[:,0] = np.sqrt(xy + xyz[:,2]**2)
    new_pts[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #new_pts[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up

    # to compensate cam effect 
    new_pts[:,2] = np.arctan2(-xyz[:,1], xyz[:,0])
    return new_pts

def convertxyz_np(sphere):
    # ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    new_pts = np.zeros(sphere.shape)
    new_pts[:,0] = sphere[:,0]*np.sin(sphere[:,1])*np.cos(sphere[:,2])
    new_pts[:,1] = sphere[:,0]*np.sin(sphere[:,1])*np.sin(sphere[:,2]) # for elevation angle defined from Z-axis down
    new_pts[:,2] = sphere[:,0]*np.cos(sphere[:,1])
    return new_pts

class RunCamera:
    def __init__(self, port1, sensornum, netuse, camopen = True):
        super(RunCamera, self).__init__()
        # exit flag
        self.should_exit = False

        # array for determining whether the sensor can do position estimation and force estimation or not
        sen_pf = np.array([[1,1,1],
                        [2,1,1],
                        [3,0,1],
                        [4,0,0],
                        [5,0,1],
                        [6,1,0],
                        [101,1,0],
                        [102,1,0]])
        # whether it use pos or force
        self.ispos = sen_pf[sen_pf[:,0]==sensornum][0,1]
        self.isforce = sen_pf[sen_pf[:,0]==sensornum][0,2]
        
        self.id = port1

        self.cen_x, self.cen_y, self.exposure = self.get_sensorinfo(sensornum)
        self.flag = 0
        self.camopened = True
        self.netuse = netuse
        # Params
        self.image = None
        self.img_noncrop = np.zeros((768,1024))
        self.maxrad = 16.88
        self.minrad = 12.23
        self.input_width = 640
        self.imgsize = int(self.input_width / 2)

        # get root path for masks
        self.rootpath = os.path.dirname(os.path.abspath(__file__))
        self.device_num = 0
        self.imgidx = np.load(os.path.join(self.rootpath, 'Img2Depth/calib_idx/mask_idx_{}.npy'.format(sensornum)))
        self.radidx = np.load(os.path.join(self.rootpath, 'Img2Depth/calib_idx/pts_2ndmask_{}_80deg.npy'.format(sensornum)))
        self.rayvector = np.load(os.path.join(self.rootpath, 'Img2Depth/calib_idx/pts_masked_{}.npy'.format(sensornum)))[:, 3:]

        # Read Base Image
        # self.baseimg = cv2.imread('../data/sen_{}_basic.jpg'.format(sensornum))
        # if self.baseimg is None:
        #     print("No base Img")
        #     self.baseimg = cv2.imread('../data/sen_{}_basic_uncropped.jpg'.format(sensornum))
        #     self.baseimg = self.rectifyimg(self.baseimg)
        #     cv2.imwrite('../data/sen_{}_basic.jpg'.format(sensornum), self.baseimg)

        ########### For ROS and camera id setting ###########
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        self.device_num = 0
        if os.path.exists(self.id):
            device_path = os.path.realpath(self.id)
            device_re = re.compile("\/dev\/video(\d+)")
            info = device_re.match(device_path)
            if info:
                self.device_num = int(info.group(1))
                # Need to add -1 because it matches the metadata
                if self.device_num %2 != 0 :
                    print("Need to adjust device num with -1")
                    self.device_num -= 1

                print(self.id+ " corresponds to the /dev/video" + str(self.device_num))
        ################## CAM setting ################

        print("done?")
        # self.cap = cv2.VideoCapture(self.device_num)
        self.cap = cv2.VideoCapture(self.device_num, cv2.CAP_V4L2)

        
        if not (self.cap.isOpened()):
            print("Cannot open the camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))


        #########   ######### CAM setting done ################
        modelname = rospy.get_param('~ckpt')
        rospy.loginfo("Load Model Path: {}".format(modelname))
        if netuse: 
            ######## model setting ######
            if self.ispos == 1:
                self.model_pos = DenseDepth(max_depth = 256, pretrained = False)
                self.ispf = 'pos'
                self.model_pos = torch.nn.DataParallel(self.model_pos)
                checkpoint_pos = torch.load(modelname)
                self.model_pos.load_state_dict(checkpoint_pos['model'])
                self.model_pos.eval()
                self.model_pos.cuda()
                # self.imgDepth = self.img2Depth(np.ones((640,640,3)))

            if self.isforce == 1:
                self.model_force = DenseNet_Force(pretrained = False)
                self.ispf = 'force'
                self.model_force = torch.nn.DataParallel(self.model_force)
                checkpoint_force = torch.load(modelname)
                self.model_force.load_state_dict(checkpoint_force['model'])
                self.model_force.eval()
                self.model_force.cuda()
                # self.imgForce = self.img2Force(np.ones((640,640,3)))

        ############ define ros publisher ############
        self.img_pub = rospy.Publisher("/RunCamera/image_raw_1", Image, queue_size=2)

        # publish depth
        self.img_pub_depth = rospy.Publisher("/RunCamera/imgDepth", Image, queue_size=2)
        self.img_pub_depth_show = rospy.Publisher("/RunCamera/imgDepth_show", Image, queue_size=2)
        # publish wrenchstamped force/torque
        self.force_pub = rospy.Publisher("/RunCamera/force", WrenchStamped, queue_size=2)
        self.pub_caminfo = rospy.Publisher("/RunCamera/camera_info", CameraInfo, queue_size=3)
        
        cam_yaml_file = rospy.get_param('~caminfo')
        self.camera_info = self.yaml_to_CameraInfo(cam_yaml_file)

        print("camera & network setup done")
        # Subscribe to the camera image and depth topics and set
        # the appropriate callbacks
        # self.image_sub = rospy.Subscriber("/RunCamera/image_raw", Image, self.image_callback)

        # rospy.loginfo("Waiting for image topics...")
        
        # start thread for camera
        rospy.loginfo("Starting camera thread...")
        if camopen:    
            self.th1 = threading.Thread(target = self.CAM_camerashow)
            self.th1.start()

    def yaml_to_CameraInfo(self, yaml_fname):
        """
        Parse a yaml file containing camera calibration data (as produced by 
        rosrun camera_calibration cameracalibrator.py) into a 
        sensor_msgs/CameraInfo msg.
        
        Parameters
        ----------
        yaml_fname : str
            Path to yaml file containing camera calibration data
        Returns
        -------
        camera_info_msg : sensor_msgs.msg.CameraInfo
            A sensor_msgs.msg.CameraInfo message containing the camera calibration
            data
        """
        print('*************file name:     ' ,yaml_fname)
        # Load data from file
        with open(yaml_fname, "r") as file_handle:
            calib_data = yaml.load(file_handle, Loader=yaml.Loader)
        # Parse'

        camera_info_msg = CameraInfo()
        camera_info_msg.width = calib_data["image_width"]
        camera_info_msg.height = calib_data["image_height"]
        camera_info_msg.K = calib_data["camera_matrix"]["data"]
        camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
        camera_info_msg.R = calib_data["rectification_matrix"]["data"]
        camera_info_msg.P = calib_data["projection_matrix"]["data"]
        camera_info_msg.distortion_model = calib_data["distortion_model"]
        return camera_info_msg


    def get_sensorinfo(self, calibnum):
        """
        get center of each sensor.
        """
        brightness = 160
        senarr = np.array([[6, 520, 389, 150, 320],
                            [1, 522, 343, 100, 314],
                            [2, 520, 389, 150, 322],
                            [3, 522, 343, 100, 316],
                            [4, 522, 343, 100, 307],
                            [5, 547, 384, 159, 303],
                            [101, 512, 358, 100, 298],
                            [102, 545, 379, 100, 300],
                            [103, 522, 343, 100, 300]])

        cen_x = senarr[senarr[:,0]==calibnum][0,1]
        cen_y = senarr[senarr[:,0]==calibnum][0,2]
        brightness = senarr[senarr[:,0]==calibnum][0,3]
        self.radius = senarr[senarr[:,0]==calibnum][0,4]
        return cen_x, cen_y, brightness

    def set_manual_exposure(self, video_id, exposure_time):
        '''
        Another option to set the manual exposure
        '''
        commands = [
            ("v4l2-ctl --device /dev/video"+str(video_id)+" -c exposure_auto=3"),
            ("v4l2-ctl --device /dev/video"+str(video_id)+" -c exposure_auto=1"),
            ("v4l2-ctl --device /dev/video"+str(video_id)+" -c exposure_absolute="+str(exposure_time)),
       ]
        for c in commands: 
            os.system(c)

    def rectresult(self, frame):

        mask = np.zeros((640,640), dtype="uint8")
        cv2.circle(mask, (320, 320), self.radius, 255, -1)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        return cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)

    def rectifyimg(self, frame2):
        '''
            function for rectifying the image based on the given camera node 
            Now the function manually get the center of each circular shape and match the function. 

            Key is to match the center of pixel correctly so that we can get the right match process with original sensor.

        '''
        beforeRectImg2 = frame2.copy()
        (h, w) = beforeRectImg2.shape[:2]

        img_reshape = beforeRectImg2.reshape(w*h, 3)
        mask = np.ones(img_reshape.shape[0], dtype=bool)
        mask[self.imgidx[self.radidx]] = False
        img_reshape[mask, :] = np.array([0, 0, 0])
        img2 = img_reshape.reshape(h, w, 3)

        beforeRectImg2 = img2[self.cen_y-self.imgsize:self.cen_y+self.imgsize,self.cen_x-self.imgsize:self.cen_x+self.imgsize]
        
        rectImg2 = beforeRectImg2
        return rectImg2

    def image_callback(self, image):
        # make depth img without using ROS

        self.imgDepth2 = self.img2Depth(self.image_original2)
        imgDepth_rgb2 = cv2.cvtColor(self.imgDepth2, cv2.COLOR_GRAY2RGB)

        self.img_pub2.publish(self.bridge.cv2_to_imgmsg(imgDepth_rgb2, "rgb8"))

        # isidx = 2
        self.depth2Ptcloud(self.imgDepth2, 2)

        self.updateDepth2 = 0


    def getPSNR(self, img1, img2):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (score, diff) = compare_ssim(gray1, gray2, full=True)
            diff = (diff * 255).astype("uint8")
            # print("SSIM: {}".format(score))
            return score


    def CAM_camerashow(self):
        time.sleep(1)
        print('reading..')

        depthImg = self.img_noncrop
        forceEst = 0
        while not rospy.is_shutdown() and not self.should_exit:  
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            # rectify img based on the param on the launch file 
            rectImg = self.rectifyimg(frame)

            msg_tact = self.br.cv2_to_imgmsg(rectImg, "bgr8")
            msg_tact.header.stamp = rospy.get_rostime()
            self.img_pub.publish(msg_tact)
            self.pub_caminfo.publish(self.camera_info)

            ################## depth / force ###########
            if netuse: 
                if self.isforce == 1:
                    forceEst = getForce(self.model_force, rectImg)
                    # print("Force: ", forceEst)
                    wrench_stamped_msg = WrenchStamped()
                        # Set the force and torque values in the message
                    wrench_stamped_msg.header.stamp = rospy.Time.now()
                    wrench_stamped_msg.wrench.force = Vector3(*forceEst[:3])
                    wrench_stamped_msg.wrench.torque = Vector3(*forceEst[3:])
                    self.force_pub.publish(wrench_stamped_msg)
                
                if self.ispos == 1:
                    # import pdb; pdb.set_trace()
                    depthImg = getDepth(self.model_pos, rectImg)
                    imgDepth_rgb = cv2.cvtColor(depthImg, cv2.COLOR_GRAY2RGB)
                    
                    # convert the depth image into mm
                    depthImg = depthImg.astype(np.float32) / 256 * (self.maxrad - self.minrad) + self.minrad
                    
                    # here, we do the filtering based on mask
                    # I create image_idx to keep track of the coordinate in the original image, idx = y * WIDTH + x
                    image_idx = np.arange(0, 768*1024)
                    img_noncrop = np.zeros((768, 1024))
                    img_noncrop[self.cen_y-self.imgsize:self.cen_y+self.imgsize, self.cen_x-self.imgsize:self.cen_x+self.imgsize] = depthImg
                    img_vec = img_noncrop.reshape(-1)
                    img_vec = img_vec[self.imgidx][self.radidx]
                    image_idx = image_idx[self.imgidx][self.radidx]

                    # ray vec is the 3d point of the gel.
                    ray_vec = self.rayvector[self.radidx,:]
                    # to ensure the same coordinate, let's use the same atan ftn from convertsperical_np ftn
                    # just for angle!
                    ray_vec_spherical = convertSpherical_np_cam(ray_vec)
                    ray_vec_spherical[:, 0] = img_vec       # TODO check this line for exp.
                    ray_vec_spherical[:, 1] = ray_vec[:,2]

                    # radius filtering
                    radfiltering_upper, radfiltering_lower = 14.5, 12.2
                    radius_filtering_index = np.where(
                        (ray_vec_spherical[:,0] < radfiltering_upper) & 
                        (ray_vec_spherical[:,0] > radfiltering_lower))
                    ray_vec_spherical = ray_vec_spherical[radius_filtering_index]
                    image_idx = image_idx[radius_filtering_index]

                    zfiltering = 2.95
                    ray_xyz = convertxyz_np(ray_vec_spherical)
                    filtering_idx = np.where(ray_xyz[:, 2] > zfiltering)
                    xyz_reduced = ray_xyz[filtering_idx]
                    img_idx = image_idx[filtering_idx]

                    # take the z map
                    img_height = img_idx // 1024
                    img_width = img_idx % 1024
                    filtered_depthImg = np.zeros((768, 1024))
                    filtered_depthImg[img_height, img_width] = xyz_reduced[:,2]
                    
                    # take the center crop
                    filtered_depthImg = filtered_depthImg[self.cen_y-self.imgsize:self.cen_y+self.imgsize, self.cen_x-self.imgsize:self.cen_x+self.imgsize]
                    # note the unit is in mm before x1000
                    filtered_depthImg *= 1000
                    filtered_depthImg = filtered_depthImg.astype(np.uint16)

                    msg_depth = self.br.cv2_to_imgmsg(filtered_depthImg, "16UC1")
                    msg_depth.header.stamp = rospy.get_rostime()
                    msg_depth.header.frame_id = "touch"
                    self.img_pub_depth.publish(msg_depth)

                    msg_depthshow = self.br.cv2_to_imgmsg(imgDepth_rgb, "rgb8")
                    msg_depthshow.header.stamp = rospy.get_rostime()
                    msg_depthshow.header.frame_id = "touch"
                    self.img_pub_depth_show.publish(msg_depthshow)
            else:
                print('please make netuse True')

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    start = timer()
	# in new laptop, the path has changed into ../catkin_ws/src/tactile_camera... need to change into relative path
    # os.chdir('../catkin_ws/src/dtv2_tactile_camera/src/')
    
    camopen = True
    netuse = True

    ################ Bring ROS param and match the camera idx #############
    rospy.init_node("Pythonnode")

    try:
        rospy.get_param_names()
    except ROSException:
        print("could not get param name")

    param_name = rospy.search_param('camname1')
    PortNum1 = rospy.get_param(param_name)
    rospy.loginfo("OpenCV Version: {}".format(cv2.__version__))

    # first value: video # for dtv2 camera (ex: n = 4 if dtv2 cam is dev/video4)
    # 2nd value: sensor number 
    # 3rd value: netuse = True if you want to use network option(shape / force estimation)
    # 4th value: PSNR checking mode
    rospy.loginfo("Video Port: {}".format(PortNum1))
    sennum = 102
    cSerial = RunCamera(PortNum1, sennum, netuse, camopen)

    # rospy shutdown in handled in the thread   
    try: 
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cSerial.should_exit = True
    
    
