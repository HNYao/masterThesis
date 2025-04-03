from realsense.realsense_helper import get_profiles
import pyrealsense2 as rs
from pynput import keyboard
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from Geo_comb.full_pipeline import predict_depth

def process_frames(frames, depth_sensor, K, model_depth=None):
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    aligned_frames = align.process(frames)
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())[..., [2, 1, 0]]

    if model_depth is not None:
        depth_image, _ = predict_depth(model_depth, color_image, K)
        depth_image = (depth_image * 1000).astype(np.uint16)
        
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    # depth image is 1 channel, color is 3 channels
    depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
    bg_removed = np.where(
        (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
        grey_color,
        color_image,
    )

    # Render images
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET
    )
    vis_image = np.hstack((bg_removed, depth_colormap))


    return color_image, depth_image, vis_image


def capture_rgbd(args, model_depth=None):
    global capture_rgbd
    
    # Initialize the realsense recorder
    pipeline = rs.pipeline()
    config = rs.config()
    color_profiles, depth_profiles = get_profiles()
    color_profile_id, depth_profile_id = 26, 0  # D455i
    color_profile_id, depth_profile_id = 18, 0  # D435i

    # note: using 640 x 480 depth resolution produces smooth depth boundaries
    #       using rs.format.bgr8 for color image format for OpenCV based image visualization
    print(
        "Using the default profiles: \n  color:{}, depth:{}".format(
            color_profiles[color_profile_id], depth_profiles[depth_profile_id]
        )
    )
    w, h, fps, fmt = depth_profiles[depth_profile_id]
    config.enable_stream(rs.stream.depth, w, h, fmt, fps)
    w, h, fps, fmt = color_profiles[color_profile_id]
    config.enable_stream(rs.stream.color, w, h, fmt, fps)

    # Start streaming and get the profile of the camera
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 1)

    # run pipeline
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.eye(3)
        K[0, 0] = intrinsics.fx
        K[1, 1] = intrinsics.fy
        K[0, 2] = intrinsics.ppx
        K[1, 2] = intrinsics.ppy
        K = K.astype(np.float32)
        color_image, depth_image, vis_image = process_frames(
            frames,
            depth_sensor,
            K,
            model_depth,
        )
        
        # Display using matplotlib
        plt.clf()  # Clear the current figure
        plt.imshow(vis_image[:, :, ::-1])
        plt.axis('off')  # Hide axes
        plt.draw()
        plt.pause(0.001)  # Small pause to allow the plot to update

        if capture_rgbd:
            color_path = ".tmp/wild_color.png"
            depth_path = ".tmp/wild_depth.png"
            intr_path = ".tmp/wild_intr.txt"
            cv2.imwrite(color_path, color_image.copy().astype(np.uint8))
            cv2.imwrite(depth_path, (depth_image).astype(np.uint16))
            np.savetxt(intr_path, K)
            print("Save images to {} and {}, intr to {}".format(color_path, depth_path, intr_path))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_m3d", action="store_true")
    args = parser.parse_args()
    if args.use_m3d:
        model_depth = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
        model_depth = model_depth.to("cuda")
        model_depth.eval()
    else:
        model_depth = None

    t1 = threading.Thread(target=capture_rgbd, args=(args, model_depth))
    t1.start()

    capture_rgbd = False

    def on_press(key):
        global capture_rgbd
        try:
            # Save image if we press 'blank space'
            if key == keyboard.Key.space:
                print("Saving images")
                capture_rgbd = True

        except AttributeError:
            # Handles special keys like 'Shift', 'Ctrl', etc.
            pass

    def on_release(key):
        global capture_rgbd
        capture_rgbd = False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
        t1.join()
