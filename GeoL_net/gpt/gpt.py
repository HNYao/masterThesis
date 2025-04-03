import os
import cv2
from typing import Optional
import openai
import base64
import requests
import numpy as np
import ast
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def chatgpt_condition(image_path: str, mode="object_placement"):
    """
    ChatGPT condition for object placement or scene understanding

    Params:
    image_path: image path or image
    mode: object_placement or scene_understanding

    """

    base64_image = encode_image(image_path)

    api_key = os.getenv("CHATGPT_API_KEY")
  

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    

    # object placement
    if mode == "object_placement":
      object_placement = input("Object need to be placed: ")
      extra_info = input("Extra information: ")
      payload = {
        "model": "gpt-4o",
        "messages": [
          {
            "role": "system", 
            "content": 
            "You are an assistant that helps people place objects on a table. \
            You will be given an image of the tabletop and a target object that needs to be placed. \
            Your task is to identify one or more anchor objects already on the table and describe them clearly. \
            Then, determine the best placement direction(s) for the target object relative to the anchor object(s). \
            Use the following directional terms for placement: Left Front, Right Front, Left Behind, Right Behind, Left, Right, Front, or Behind. \
            If multiple anchor objects are necessary, specify each anchor and its corresponding direction, separating them with commas. Ensure that the placement avoids clutter and maintains logical accessibility based on the scene." 
                    },
          {
          "role": "assistant",
          "content": """
              Here are the examples:
              Assume the given image contains: monitor, cup, hone, red can, black bottle, green book.
                Please note that anchors should be split by ",".
                1. Mouse. I am a right-handed. Please answer:
                    anchor: monitor, black bottle
                    direction: Right Front, Left Front
                2. Mouse. I am a left-handed. Please answer:
                    anchor: monitor 
                    direction: Left Front
                3. bottle. Please answer:
                    anchor: black bottle
                    direction: Left Front
                4. phone. I like playing mobile games and drinking coffee. Please answer:
                    anchor: blue cup
                    direction: Right Front
          """
            },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Base on the image, where should I put {object_placement} reasonably without collision and overlap with other objects? Please attention: {extra_info} Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\n "
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
        ],
        "max_tokens": 20
      }
      print("object need to be placed: ", object_placement)

    elif mode == "scene_understanding":
    # scene understanding
      payload =  {
        "model": "gpt-4o-mini",
        "messages": [
          {
            "role": "system", 
            "content": "You are an assistant that helps people place objects on the table.\
                  You are given a image of the tabletop and an target object to be placed. \
                    You should determine the anchor object with color description and in which direction the target object should be placed relative to the anchor object."
                    },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "What is on the table? in schema: <color> <object>, <color> <object>, ..."
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
        ],
        "max_tokens": 300
      }


    
    reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(reponse.json()['choices'][0]['message']['content'])

    # refine the response
    while True:
        user_input = input("User: ")

        if user_input.lower() == 'okie':
            break
        
        payload['messages'].append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": reponse.json()['choices'][0]['message']['content']
                }
            ]
        })

        payload['messages'].append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        })

        reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(reponse.json()['choices'][0]['message']['content'])

    anchor, direction = extract_response(reponse.json()['choices'][0]['message']['content'])
    return anchor, direction

def chatgpt_selected_plan(image_path: str):
    
    """
    ChatGPT condition for object placement or scene understanding

    Params:
    image_path: image path or image
    mode: object_placement or scene_understanding

    """

    base64_image = encode_image(image_path)

    api_key = os.getenv("CHATGPT_API_KEY")
  

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    

    # object placement

    object_placement = input("Object need to be placed: ")
    extra_info = input("Extra information: ")
    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "system", 
          "content":
            "You are an assistant that helps people place objects on a table. \
            You are given a image of the tabletop with bounding box and the ID provided for each object in the scene, and a target object's category name to be placed. \
            Your task is to select one or more anchor objects with bounding boxes and IDs in the image, and provide the name of each object within the bounding box.\
            Only the object with a bounding box and an ID can be selected as the anchor object. Note the ID is in the top left corner of each bounding box.\
            Then, determine the best placement direction(s) for the target object relative to these anchor object(s). \
            Use the following directional terms for placement: Left Front, Right Front, Left Behind, Right Behind, Left, Right, Front, or Behind. \
            If multiple anchor objects are necessary, specify each anchor and its corresponding direction, separating them with commas. \
            Ensure that the placement follows human common sense and maintains logical accessibility based on the scene.\
            The format should be as follows:\
            anchor: <object name 1, object name 2, ...>\n \
            direction: <direction of target object relative to object name 1, direction of target object relative to object name 2, ...>\n \
            bbox id: <int(ID of object name 1),  int(ID of object name 2), ...>\n "
                  },
        {
        "role": "assistant",
        "content": """
            Here are the examples:
            Assume the given image contains objects and their bboxes: monitor(bbox id2), blue cup(bbox id0), blue phone(bbox id3), red can(no bbox), black bottle(bbox id 1), green book(no box).
              Please note that anchors should be split by ",".
              1. Mouse. I am a right-handed. Please answer:
                  anchor: monitor, black bottle
                  direction: Right Front, Left Front
                  bbox id: 2, 1
              2. Mouse. I am a left-handed. Please answer:
                  anchor: monitor 
                  direction: Left Front
                  bbox id: 2
              3. bottle. Please answer:
                  anchor: black bottle
                  direction: Left Front
                  bbox id: 1
              4. phone. I like playing mobile games and drinking coffee. Please answer:
                  anchor: blue cup
                  direction: Right Front
                  bbox id: 0
        """
          },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Base on the image, where should I put {object_placement} reasonably without collision and overlap with other objects? Please attention: {extra_info} Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\n id: <id of the bounding box>\n "
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 40
    }
    print("object need to be placed: ", object_placement)

    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json()['choices'][0]['message']['content'])

    # refine the response
    while True:
        user_input = input("User: ")

        if user_input.lower() == 'okie':
            break
        
        payload['messages'].append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.json()['choices'][0]['message']['content']
                }
            ]
        })

        payload['messages'].append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_input
                }
            ]
        })

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json()['choices'][0]['message']['content'])
    content = response.json()['choices'][0]['message']['content']
    content_split = content.split("\n")
    anchor_reponse = content_split[0].split(":")[1].strip()
    direction_response = content_split[1].split(":")[1].strip()
    bbox_id_response = content_split[2].split(":")[1].strip()

    anchor_reponse = [anchor.strip() for anchor in anchor_reponse.split(", ")]
    direction_response = [direction.strip() for direction in direction_response.split(", ")]
    bbox_id_response = [bbox_id.strip() for bbox_id in bbox_id_response.split(", ")]

    return anchor_reponse, direction_response, bbox_id_response

def extract_response(response: str) -> str:
    """
    Extract the anchor and direction from the chatgpt response

    Params:
    response: chatgpt response

    Return:
    anchor, direction
    """
    response = response.split("\n")
    anchor_reponse = response[0].split(":")[1].strip()
    direction_response = response[1].split(":")[1].strip()

    anchors = [anchor.strip() for anchor in anchor_reponse.split(",")]
    directions = [direction.strip() for direction in direction_response.split(",")]

    return anchors, directions

def chatgpt_object_detection_bbox(image_path: str, object_name: str):
    """
    ChatGPT condition for object detection and bounding box

    Params:
    image_path: image path or image
    object_name: object name to be detected

    Return:
    List of bounding box coordinates, [[min x, min y , max, x, max y], ...]
    """

    base64_image = encode_image(image_path)

    api_key = os.getenv("CHATGPT_API_KEY")

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
          {
            "role": "system", 
            "content": "You are an assistant that helps people detect object on the table.\
                  You are given a image of the tabletop and an target object to be detected. \
                  Please respond, in text, with bounding box coordinates of the object position.\
                  The bounding box coordinates should be of the form [min x, min y, max x, max y] in descending  order of confidence\
                  where x y are 0.00-1.00 correspond to fraction of the image along the width and height of the image with the top left of the image as the origin. \
                  If there are no locations in the image where \
                  a <object_type> could be placed, respond only with the empty list '[]'.\
                  do not include any other text in your response."
                    },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Please find the bounding box of {object_name} in the following image"
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
        ],
        "max_tokens": 300
      }

    reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(reponse.json()['choices'][0]['message']['content'])
    bbox_list = extract_bbox_list_from_response(reponse.json()['choices'][0]['message']['content'])
    return bbox_list

def extract_bbox_list_from_response(response: str):
    """
    Extract the bounding box coordinates from the chatgpt response

    Params:
    response: chatgpt response

    Return:
    List of bounding box coordinates, [[min x, min y , max, x, max y], ...]
    """
    try:
        return ast.literal_eval(response)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing the input string: {e}")
        return None
    

def chatgpt_object_placement_bbox(gpt_version:str, image_path: str, prompts_obj_place: str, prompts_direction: str, prompts_anchor_obj: str):
    """
    ChatGPT condition for object detection and bounding box

    Params:
    image_path: image path or image
    object_name: object name to be detected

    Return:
    List of bounding box coordinates, [[min x, min y , max, x, max y], ...]
    """

    base64_image = encode_image(image_path)

    api_key = os.getenv("CHATGPT_API_KEY")

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # gpt-4o-mini
    # gpt-4o

    payload = {
        "model": gpt_version, 
        "messages": [
          {
            "role": "system", 
            "content": "You are an assistant that helps people place object on the table.\
                  Plese avoid collision and overlap with other objects.\
                  You are given a image of the tabletop and an target object to be placed. \
                  Please respond, in text, with bounding box coordinates of potential locations to place the object.\
                  The bounding box coordinates should be of the form [min x, min y, max x, max y], only containing one [], no more []\
                  where x y are 0.00-1.00 correspond to fraction of the image along the width and height of the image with the top left of the image as the origin. \
                  If there are no locations in the image where \
                  a <object_type> could be placed, respond only with [0, 0, 1, 1].\
                  do not include any other text in your response."
                    },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Please provide the bounding box to placing a new {prompts_obj_place}. The {prompts_obj_place} should be placed to  {prompts_direction} the {prompts_anchor_obj}"
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
        ],
        "max_tokens": 300
      }

    reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(reponse.json()['choices'][0]['message']['content'])
    bbox_list = extract_bbox_list_from_response(reponse.json()['choices'][0]['message']['content'])
    return bbox_list


def chatgpt_object_placement_bbox_o1(image_path: str, prompts_obj: str, prompts_direction: str):
    """
    ChatGPT condition for object detection and bounding box

    Params:
    image_path: image path or image
    object_name: object name to be detected

    Return:
    List of bounding box coordinates, [[min x, min y , max, x, max y], ...]
    """

    base64_image = encode_image(image_path)

    api_key = os.getenv("CHATGPT_API_KEY")

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # gpt-4o-mini
    model = "gpt-4o-mini"
    #model = "gpt-4o"
    #model = 'o1-preview'
    payload = {
        "model": model, 
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Please provide the bounding box to placing a new {prompts_obj}. The {prompts_obj} should be placed to {prompts_direction}\
                  You are an assistant that helps people place object on the table.\
                  Plese avoid collision and overlap with other objects.\
                  You are given a image of the tabletop and an target object to be placed. \
                  Please respond, in text, with bounding box coordinates of potential locations to place the object.\
                  The bounding box coordinates should be of the form [min x, min y, max x, max y] in descending  order of confidence\
                  where x y are 0.00-1.00 correspond to fraction of the image along the width and height of the image with the top left of the image as the origin. \
                  If there are no locations in the image where \
                  a <object_type> could be placed, respond only with the empty list '[]'.\
                  do not include any other text in your response."
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
        ],
        "max_completion_tokens": 300
      }

    reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(reponse.json()['choices'][0]['message']['content'])
    bbox_list = extract_bbox_list_from_response(reponse.json()['choices'][0]['message']['content'])
    return bbox_list


def visualize_bbox_set(image_path, bbox_list, is_mask=False):
    """
    Visualize the bounding box on the image

    Params:
    image: image
    bbox_list: list of bounding box coordinates, [[min x, min y , max, x, max y], ...]

    Return:
    Image with bounding box
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    assert len(bbox_list) > 0, "No object detected in the image"

    for bbox in bbox_list:
        [min_x, min_y, max_x, max_y] = bbox
        min_x = int(min_x * width)
        min_y = int(min_y * height)
        max_x = int(max_x * width)
        max_y = int(max_y * height)
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    
    
    cv2.imwrite("outputs/img_output/o1.jpg", image)
    

def chatgpt_select_id(image_path: str, text_prmpt, mode="object_placement"):
    """
    Use chatgpt to select an id 

    Params:
    image_path: image path or image
    mode: object_placement or scene_understanding

    Return:
    final_response
    """

    base64_image = encode_image(image_path)

    api_key = os.getenv("CHATGPT_API_KEY")


    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    

    # object placement
    if mode == "object_placement":
      payload = {
        "model": "gpt-4o",
        "messages": [
          {
            "role": "system", 
            "content": "You are an assistant that helps people place objects on the table.\
                  You are given a image of the tabletop and an target object to be found. \
                  Please select a reference object among the objects with bounding boxes and labels.\
                    You should determine the id of bounding box of target object\
                    "
                    },
          {
          "role": "assistant",
          "content": """
              Here are the examples:
              Assume the given image contains: white monitor, mouse, blue cup, blue phone, red can, black bottle, green book.

                1. Monitor . Please answer:
                    id: 2
                2. bottle. Please answer:
                    id: 1
                3. phone.  Please answer:
                    id: 0

          """
            },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Base on the image, where bbox id is {text_prmpt}?  Answer should be in the following format without any explanations: id: <id of the bounding box>\n"
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
        ],
        "max_tokens": 20
      }


    
    reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        context = reponse.json()['choices'][0]['message']['content']
        context = context.split(": ")[1].strip()
    except KeyError:
        #context = "id: 0"
        context=0
    return context
    # print(reponse.json()['choices'][0]['message']['content'])

    # # refine the response

        
    # payload['messages'].append({
    #     "role": "assistant",
    #     "content": [
    #         {
    #             "type": "text",
    #             "text": reponse.json()['choices'][0]['message']['content']
    #         }
    #     ]
    # })

    # # payload['messages'].append({
    # #     "role": "user",
    # #     "content": [
    # #         {
    # #             "type": "text",
    # #             "text": user_input
    # #         }
    # #     ]
    # # })

    # reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    #print(reponse.json()['choices'][0]['message']['content'])
    # anchor= extract_response()
    # import re
    # numbers = re.findall(r'\d+', anchor)v2_kinect_cfg/id18/cup_0004_white/with_obj/test_pbr/000000/rgb/000000.jpg"
    # object_name = "a pair of eye glasses"
    # prompts_obj = "cup"
    # prompts_direction = 'the left of the eye glasses'
    # bbox_list = chatgpt_object_placement_bbox_o1(image_path, prompts_obj, prompts_direction)
    # visualize_bbox_set(image_path, bbox_list)