import open3d as o3d
import json as js
import numpy as np

file = "/home/arm/NBV_Boshu/gs_data/touch/touch.json"

with open(file) as f:
    data = js.load(f)

coords = []
for frame in data["frames"]:
    transform = np.array(frame["transformation"])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    coord.transform(transform)
    coords.append(coord)

o3d.visualization.draw_geometries(coords)