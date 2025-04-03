# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/sensors/realsense_recorder.py

# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
from os import makedirs
from os.path import exists, join, abspath
import shutil
import json
from enum import IntEnum

import sys, os
from pynput import keyboard
import threading

sys.path.append(abspath(__file__))
from realsense_helper import get_profiles

# try:
#     # Python 2 compatible
#     input = raw_input
# except NameError:
#     pass


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, "w") as outfile:
        obj = json.dump(
            {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "intrinsic_matrix": [
                    intrinsics.fx,
                    0,
                    0,
                    0,
                    intrinsics.fy,
                    0,
                    intrinsics.ppx,
                    intrinsics.ppy,
                    1,
                ],
            },
            outfile,
            indent=4,
        )


def process_frames(frames, depth_sensor):
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


def main(args):
    global save_rgbd

    # Initialize the save folder
    path_depth = join(args.output_folder, "depth")
    path_color = join(args.output_folder, "color")
    os.makedirs(path_depth, exist_ok=True)
    os.makedirs(path_color, exist_ok=True)

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
    depth_sensor.set_option(rs.option.visual_preset, Preset.Default)
    frame_count = len(os.listdir(path_depth))

    # run pipeline
    while True:
        frames = pipeline.wait_for_frames()
        if frame_count == 0:
            color_frame = frames.get_color_frame()
            save_intrinsic_as_json(
                join(args.output_folder, "camera_intrinsic.json"), color_frame
            )

        color_image, depth_image, vis_image = process_frames(
            frames,
            depth_sensor,
        )
        # save the images if we press s
        path_depth = join(args.output_folder, "depth")
        path_color = join(args.output_folder, "color")
        if save_rgbd:
            cv2.imwrite(join(path_depth, f"{frame_count:06d}.png"), depth_image)
            cv2.imwrite(join(path_color, f"{frame_count:06d}.png"), color_image)
            frame_count += 1
            print("Save frame to {}".format(join(path_color, f"{frame_count:06d}.png")))

        cv2.imshow("RealSense", vis_image)
        cv2.waitKey(2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Realsense Recorder. Please select one of the optional arguments"
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default="./dataset/wild_collide/",
        help="set output folder",
    )

    args = parser.parse_args()
    t1 = threading.Thread(target=main, args=(args,))
    t1.start()

    save_rgbd = False

    def on_press(key):
        global save_rgbd
        try:
            # Save image if we press 'blank space'
            if key == keyboard.Key.space:
                print("Saving images")
                save_rgbd = True

        except AttributeError:
            # Handles special keys like 'Shift', 'Ctrl', etc.
            pass

    def on_release(key):
        global save_rgbd
        save_rgbd = False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
        t1.join()
