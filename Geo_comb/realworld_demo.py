from pred_one_case_completed import *

def realworld_demo(rgb_image, depth_image, intrinsics, affordance_model, diffusion_model, use_chatgpt, is_visulization):
    """
    Perform real-world demo on a given RGB-D image.

    Parameters:
        rgb_image (np.ndarray): The input RGB image.
        depth_image (np.ndarray): The input depth image.
        intrinsics (np.ndarray): The camera intrinsics.
        affordance_model (torch.nn.Module): The affordance prediction model.
        diffusion_model (torch.nn.Module): The diffusion model.
        use_chatgpt (bool): Whether to use ChatGPT for generating text descriptions.
        is_visulization (bool): Whether to visualize the results.

    Returns:
        str: The generated text description.
    """

    # generate the prompts
    if use_chatgpt:
        print("====> Using ChatGPT to generate prompts...")
        target_name, direction_text = chatgpt_condition(rgb_image_file_path, "object_placement")
        print("====> Predicting Affordance...")
    else:
        print("====> Please Set the prompts manually...")
        target_name = input("Please input the target object name: ")
        direction_text = input("Please input the direction: ")
    
    # use GroundingDino to detect the target object
    annotated_frame = rgb_obj_dect(rgb_image, depth_image, intrinsics, target_name)
    color_no_obj = np.array(annotated_frame) / 255
    depth = cv2.imread(depth_image_file_path, cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32)

    intr = INTRINSICS
    points_no_obj_scene, scene_no_obj_idx = backproject(
        depth,
        intr,
        np.logical_and(depth > 0, depth < 2500),
        NOCS_convention=False,
    )
    colors_no_obj_scene = color_no_obj[scene_no_obj_idx[0], scene_no_obj_idx[1]]
    pcd_no_obj_scene = visualize_points(points_no_obj_scene, colors_no_obj_scene)

    # model init
    affordance_model.eval()
    diffusion_model.eval()

    dataset = pred_one_case_dataset(pcd_no_obj_scene, rgb)
