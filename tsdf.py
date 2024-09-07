import open3d as o3d
import cv2
import json as js
import numpy as np

JSON_PATH = 'data/2024-09-06-03-17-00/transforms.json'
ROOT = 'data/2024-09-06-03-17-00/'
if __name__ == "__main__":
    # read json file
    with open(JSON_PATH) as f:
        data = js.load(f)


    frames = data["frames"]
    # create TSDF Volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    cam_w = data["w"]
    cam_h = data["h"]
    focal_x = data["fl_x"]
    focal_y = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]

    # create camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(cam_w, cam_h, focal_x, focal_y, cx, cy)

    idx = 0
    for idx in range(len(frames)):
        print(idx)
        frame =  frames[idx]
        color = cv2.imread(f'{ROOT}/{frame["file_path"]}')
        depth = cv2.imread(f'{ROOT}/{frame["depth_file_path"]}', cv2.IMREAD_UNCHANGED)
        depth_meter = depth.astype(np.float32) / 1000.0
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        # create camera pose
        c2w = np.array(frame["transform_matrix"])
        extrinsic = np.linalg.inv(c2w)

        # create rgbd image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth_meter),
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False)
        
        # integrate rgbd image into TSDF volume
        volume.integrate(rgbd, intrinsic, extrinsic)
        
        
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()  # Compute normals for better visualization
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])
    
    point_cloud = volume.extract_point_cloud()
    
    o3d.visualization.draw_geometries([point_cloud])