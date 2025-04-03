from GeoL_net.dataset_gen.RGBD2PC import backproject, visualize_points
import cv2
import numpy as np
import open3d as o3d
color_image = ".tmp/wild_color.png"
depth_image = ".tmp/wild_depth.png"
depth_image2 = ".tmp/wild_depth_m3d.png"
intrinsics = np.loadtxt(".tmp/wild_intr.txt")

color_image = cv2.imread(color_image)
depth_image = cv2.imread(depth_image, cv2.IMREAD_ANYDEPTH)
depth_image2 = cv2.imread(depth_image2, cv2.IMREAD_ANYDEPTH)

pc, _ = backproject(depth_image, intrinsics, depth_image > 0)
pc2, _ = backproject(depth_image2, intrinsics, depth_image2 > 0)
pcd1 = visualize_points(pc)
pcd2 = visualize_points(pc2)
pcd1.paint_uniform_color([1, 0, 0])
pcd2.paint_uniform_color([0, 1, 0])
o3d.visualization.draw([pcd1, pcd2])
