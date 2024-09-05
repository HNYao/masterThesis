import open3d as o3d
import numpy as np
import PIL.Image as Image
pcd = o3d.io.read_point_cloud("outputs/point_cloud_4cls.ply")
R = pcd.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])

pcd.rotate(R, center=(0, 0, 0))
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)

vis.add_geometry(pcd)

vis.poll_events()
vis.update_renderer()
#vis.capture_screen_image("outputs/point_cloud_4cls.png", do_render=True)

img = vis.capture_screen_float_buffer(do_render=True)
img = np.asarray(img) * 255
img = img.astype(np.int8)
print(img.shape)
vis.destroy_window()
img = np.asarray(img) * 255
img = img.astype(np.uint8)
print(img.shape)
image_pil = Image.fromarray(img)
