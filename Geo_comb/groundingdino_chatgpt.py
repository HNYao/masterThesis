from thirdpart.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from GeoL_net.gpt.gpt import chatgpt_select_id
import torch

def rgb_obj_dect_use_gpt_select(
    image_path,
    text_prompt,
    out_dir=None,
    model_path="GroundingDINO/weights/groundingdino_swint_ogc.pth",
    text_prompt_all = "cup, bottle, monitor, phone, bowl, plate, laptop, keyboard, mouse, box, pen",

):

    model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", model_path
    )
    IMAGE_PATH = image_path
    TEXT_PROMPT = text_prompt_all
    BOX_TRESHOLD = 0.1 # 0.35
    TEXT_TRESHOLD = 0.2 # 0.25

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )

    phrases = [f"id{id}" for id in range(len(phrases))]


    h, w, _ = image_source.shape
    ori_boxes = boxes * torch.Tensor([w, h, w, h])
    ori_boxes = torch.round(ori_boxes)

    id_xy_dict = {}
    for i, box in enumerate(ori_boxes):
        id_xy_dict[phrases[i]] = (int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item()))

    if len(ori_boxes) == 0:
        print("No object detected")
        anchor_x = 100
        anchor_y = 100
        anchor_w = 10
        anchor_h = 10
    else:
        anchor_x = int(ori_boxes[0][0].item())
        anchor_y = int(ori_boxes[0][1].item())
        anchor_w = int(ori_boxes[0][2].item())
        anchor_h = int(ori_boxes[0][3].item())


    if out_dir is not None:
        # print("orignal boxes cxcy:", ori_boxes, ori_boxes[0][0], ori_boxes[0][1])
        annotated_frame = annotate(
            image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
        )

        write_box_path = "Geo_comb/annotated_box.jpg"  
        cv2.imwrite(write_box_path, annotated_frame)      

    
    response = chatgpt_select_id(write_box_path, text_prompt)
    #print("response:", response)

    # use the chatgpt to select the object

    id_bbox = response.replace(": ", "")
    if id_bbox not in id_xy_dict:
        print("The object is not in the image")
        id_bbox = "id0" # default to the first object
    center_x, center_y, w, h = id_xy_dict[id_bbox]

    
    anchor_x = center_x
    anchor_y = center_y

    print("box id:", id_bbox)
    annotated_frame[:] = 0
    cv2.circle(annotated_frame, (anchor_x, anchor_y), 5, (255, 0, 0), -1)
    write_path = "Geo_comb/annotated.jpg"
    cv2.imwrite(write_path, annotated_frame)

    return annotated_frame


if __name__ == "__main__":
    rgb_img_path = "dataset/realworld_2103/color/000037.png"
    anchor_obj_name = "laptop"
    annotated_frame = rgb_obj_dect_use_gpt_select(
        image_path=rgb_img_path,
        text_prompt=anchor_obj_name,
        out_dir="outputs"
    )
