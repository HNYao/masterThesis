import os
import cv2
from typing import Optional
import openai #1.45.1
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
    base64_image_example = encode_image("./GeoL_net/gpt/example_case.jpg")

    api_key = os.getenv("CHATGPT_API_KEY")
  

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    

    # object placement
    model = "gpt-4o"
    role = "system" if model == "gpt-4o" else "user"
    object_placement = input("Object need to be placed: ")
    extra_info = input("Extra information: ")
    payload = {
      "model":  f"{model}",
      "messages": [
        {
          "role": f"{role}", 
          "content":
            "You are an AI assistant that helps place objects for people.\
            **Input**: You are provided with an image of a tabletop scene. The image contains multiple objects, each enclosed within a bounding box with a unique ID displayed in the top-left corner.\
            You are also given the category name of a **target object** that needs to be placed. \n\
            Task: Your goal is to determine the best placement for the target object using existing objects in the scene as **anchor objects** (i.e., reference objects). Think step by Step! \
            **Guidelines**:\
            1. Selecting Anchor Objects: Choose one or more anchor objects from the image. Only objects with a bounding box and an assigned ID can be selected as anchors. For each anchor, provide its name and ID. \n \
            2. Determining Placement Direction: Specify the best placement direction(s) for the target object relative to each chosen anchor. Use the following directional terms: Left Front, Right Front, Left Behind, On \n \
              Right Behind, Left, Right, Front, Behind. If multiple anchor objects are needed, list each anchor alongside its corresponding placement direction. \n \
            3.  Ensuring Logical Placement: Placement should follow human common sense and maintain accessibility in the scene, also notice the physical plausibility. \n \
            **Output Format**:\
            anchor: <object name 1, object name 2, ...>\n \
            direction: <direction of target object relative to object name 1, direction of target object relative to object name 2, ...>\n \
            bbox id: <int(ID of object name 1),  int(ID of object name 2), ...>\n "
                  },
        {"role": "user",
         "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_example}"
                }
            },
            {
                "type": "text",
                "text": f"Base on the image, where should I put a mouse reasonably without collision and overlap with other objects? Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\nbbox id: <id of the bounding box>\n "
            }

          ]
            
        },
        {
          "role": "assistant",
          "content":"""There is already a monitor, a keyboard, a laptop, a white cup and a power strip on the table. \
            The monitor has a bounding box with ID 0, so it can be sleected as a potential anchor object. \
            The keyboard has a bounding box with ID 1, so it can be sleected as a potential anchor object. \
            The power strip has a bounding box with ID 2, so it can be sleected as a potential anchor object. \
            The cup has a bounding box with ID 3, so it can be sleected as a potential anchor object. \
            The laptop does not have a bounding box, so it can not be selected as anchor object forever. \
            The user want to put a mouse on the table. \
            The mouse is always used with the minitor and keyboard, so the keyboard and monitor are the anchor objects. \
            The mouse is usually used with the keyboard.\
            The user does not provide any extra information, so we can assumen user is right-handed. \
            The mouse should be placed on the Right of the keyboard. \
            so the final response is: anchor: keyboard\ndirection: Right\nbbox id: 1
          """
        },

        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Base on the image, where should I put a bottle reasonably without collision and overlap with other objects?  Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\nbbox id: <id of the bounding box>\n "
            }
          ]
        },
        {
          "role": "assistant",
          "content":"""There is already a monitor, a keyboard, a laptop, a white cup and a power strip on the table. \
            The monitor has a bounding box with ID 0, so it can be sleected as a potential anchor object. \
            The keyboard has a bounding box with ID 1, so it can be sleected as a potential anchor object. \
            The power strip has a bounding box with ID 2, so it can be sleected as a potential anchor object. \
            The cup has a bounding box with ID 3, so it can be sleected as a potential anchor object. \
            The laptop does not have a bounding box, so it can not be selected as anchor object forever. \
            The user want to put a bottle on the table. \
            The bottle is usually used with the cup.\
            The user does not provide any extra information.\
            The bottle should be placed on the Left Front of the cup. \
            so the final response is: anchor: cup\ndirection: Left Front\nbbox id: 3
          """
        },
        
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Base on the image, where should I put {object_placement} reasonably without collision with other objects? {extra_info}\
                    Attention: Placement should try to satisfy the user preference but also make sure to not collide with the other scene objects. \
                    Attention: We prefer to use 1~3 anchors objects to describe the placement location!\
                    Attention: Answer should be in the following format, don't give extra contents: \
                    ```\n\
                      anchor: <anchor object 1, anchor object 2, ...>\n\
                      direction: <direction 1, direction 2, ...>\n\
                      id: <id of anchor object 1, id of anchor object 2, ...>\n \
                      reason: <reason why these anchor objects are chosen>\n\
                    ```",
                    
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
      # "max_tokens": 60
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
    content = content.strip("```").strip("```plaintext").strip("```python").strip("```json").strip("```yaml").strip("```text").strip("```txt").strip("```python3").strip("```plaintext")
    try:
        content_split = content.split("\n")
        if "" in content_split or " " in content_split:
            content_split.remove("")
        anchor_reponse = content_split[0].split(":")[1].strip()
        direction_response = content_split[1].split(":")[1].strip()
        bbox_id_response = content_split[2].split(":")[1].strip()
    except:
        import pdb; pdb.set_trace()
    anchor_reponse = [anchor.strip() for anchor in anchor_reponse.split(", ")]
    direction_response = [direction.strip() for direction in direction_response.split(", ")]
    bbox_id_response = [bbox_id.strip() for bbox_id in bbox_id_response.split(", ")]

    return anchor_reponse, direction_response, bbox_id_response



def chatgpt_selected_plan_given_image(image_path: str, image_given_case: str = "GeoL_net/gpt/tabletop_scene_yao.JPG"):
    
    """
    ChatGPT condition for object placement or scene understanding

    Params:
    image_path: image path or image
    mode: object_placement or scene_understanding

    """

    base64_image = encode_image(image_path)
    base64_image_example = encode_image("./GeoL_net/gpt/example_case.jpg") # for teaching the gpt model how to use visual prompting
    base64_image_given_case = encode_image(image_given_case) # for asking the gpt model to follow the given case to arrange objects

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
            "You are an AI assistant that helps place objects for people.\
            **Input**: You are provided with an image of a tabletop scene. The image contains multiple objects, each enclosed within a bounding box with a unique ID displayed in the top-left corner.\
            You are also given the category name of a **target object** that needs to be placed. \n\
            You are also gieven an example image which guidances you where to place the object. You should place the object as the example image.\
            Task: Your goal is to determine the best placement for the target object using existing objects in the scene as **anchor objects** (i.e., reference objects). Please try to use more than\
              one anchor objects to describe one location. Think step by Step! \
            **Guidelines**:\
            1. Selecting Anchor Objects: Choose one or more anchor objects from the image. Only objects with a bounding box and an assigned ID can be selected as anchors. For each anchor, provide its name and ID. \n \
            2. Determining Placement Direction: Specify the best placement direction(s) for the target object relative to each chosen anchor. Use the following directional terms: Left Front, Right Front, Left Behind, On \n \
              Right Behind, Left, Right, Front, Behind. If multiple anchor objects are needed, list each anchor alongside its corresponding placement direction. \n \
            3.  Ensuring Logical Placement: Placement should follow human common sense and maintain accessibility in the scene, also notice the physical plausibility. \n \
            **Output Format**:\
            anchor: <object name 1, object name 2, ...>\n \
            direction: <direction of target object relative to object name 1, direction of target object relative to object name 2, ...>\n \
            bbox id: <int(ID of object name 1),  int(ID of object name 2), ...>\n "
        },
        {"role": "user",
         "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image_example}"
                }
            },
            {
                "type": "text",
                "text": f"Base on the image, where should I put a mouse reasonably without collision and overlap with other objects? Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\nbbox id: <id of the bounding box>\n "
            }

          ]    
        },
        {
          "role": "assistant",
          "content":"""There is already a monitor, a keyboard, a laptop, a white cup and a power strip on the table. \
            The monitor has a bounding box with ID 0, so it can be sleected as a potential anchor object. \
            The keyboard has a bounding box with ID 1, so it can be sleected as a potential anchor object. \
            The power strip has a bounding box with ID 2, so it can be sleected as a potential anchor object. \
            The cup has a bounding box with ID 3, so it can be sleected as a potential anchor object. \
            The laptop does not have a bounding box, so it can not be selected as anchor object forever. \
            The user want to put a mouse on the table. \
            The mouse is always used with the minitor and keyboard, so the keyboard and monitor are the anchor objects. \
            The mouse is usually used with the keyboard.\
            The user does not provide any extra information, so we can assumen user is right-handed. \
            The mouse should be placed on the Right of the keyboard. \
            so the final response is: anchor: keyboard\ndirection: Right\nbbox id: 1
          """
        },

        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Base on the image, where should I put a bottle reasonably without collision and overlap with other objects?  Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\nbbox id: <id of the bounding box>\n "
            }
          ]
        },
        {
          "role": "assistant",
          "content":"""There is already a monitor, a keyboard, a laptop, a white cup and a power strip on the table. \
            The monitor has a bounding box with ID 0, so it can be sleected as a potential anchor object. \
            The keyboard has a bounding box with ID 1, so it can be sleected as a potential anchor object. \
            The power strip has a bounding box with ID 2, so it can be sleected as a potential anchor object. \
            The cup has a bounding box with ID 3, so it can be sleected as a potential anchor object. \
            The laptop does not have a bounding box, so it can not be selected as anchor object forever. \
            The user want to put a bottle on the table. \
            The bottle is usually used with the cup.\
            The user does not provide any extra information.\
            The bottle should be placed on the Left Front of the cup. \
            so the final response is: anchor: cup\ndirection: Left Front\nbbox id: 3
          """
        },
        
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Base on the image, where should I put {object_placement} reasonably without collision and overlap with other objects? Please attention: {extra_info}. Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\n id: <id of the bounding box>\n "
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image_given_case}"
                }
            },
            {
                "type": "text",
                "text": f"Please follow the example case when suggesting object placements. For instance, if the long-handled soup spoon is placed on the pot in the example, you should also suggest placing the long-handled soup spoon on the pot in similar cases. Base your recommendations closely on the placements in the example"
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"Base on the image, where should I put {object_placement} reasonably without collision and overlap with other objects? Please attention: follow the expample case to place the object. \
                Use more than one anchor objects to describe one location. \
                Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\n id: <id of the bounding box>\n "
            },
          ]
        },

      ],
      "max_tokens": 60
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
    content.strip("```").strip("plaintext")

    try:
        content_split = content.split("\n")
        anchor_reponse = content_split[0].split(":")[1].strip()
        direction_response = content_split[1].split(":")[1].strip()
        bbox_id_response = content_split[2].split(":")[1].strip()
    except:
        import pdb; pdb.set_trace()
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
        "model": "gpt-4o-mini",
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

  


def get_sentence_probability(sentence):
    api_key = os.getenv("CHATGPT_API_KEY")
    openai.api_key = api_key
    # 使用gpt-3.5-turbo模型进行聊天模式请求
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # 使用gpt-3.5-turbo模型
        messages=[
            {"role": "system", "content": "Repeat the given sentenece."},
            {"role": "user", "content": sentence}
        ],
        logprobs=True,  # 使用布尔值True来启用logprobs
        max_tokens=10
    )


    # sum the log probabilities
    log_prob_sum = 0.0
    for i in range(len(response['choices'][0].get('logprobs', {})['content'])):
        log_prob_i = response['choices'][0].get('logprobs', {})['content'][i]["logprob"]
        log_prob_sum += log_prob_i
    return log_prob_sum          



def convert_form_fix2free(list_anchor:list, list_direction:list, object_to_place:str):
  """
  use chatgpt to convert the fixed form instruction to a free form instruction
  e.g. Left-Monitor -> I wanna put the object on the left of the monitor
  """

  api_key = os.getenv("CHATGPT_API_KEY")
  headers = {
      "content-type": "application/json",
      "Authorization": f"Bearer {api_key}"
  }
  assert len(list_anchor) == len(list_direction), "The length of anchor and direction should be the same"

  fixed_form = ""
  for i in range(len(list_anchor)):
      if list_direction[i] == "On":
        fixed_form += f"Put the {object} On the {list_anchor[i]}.\n"
      if list_direction[i] in ["Behind", "Left Behind", "Right Behind", "Left Front", "Right Front"]:
        fixed_form += f"Put the {object_to_place} {list_direction[i]} the {list_anchor[i]}.\n"
      if list_direction[i] in ["Front"]:
        fixed_form += f"Put the {object_to_place} in {list_direction[i]} the {list_anchor[i]}.\n"
      if list_direction[i] in ["Left", "Right"]:
        fixed_form += f"Put the {object_to_place} to the {list_direction[i]} of the {list_anchor[i]}.\n"

  

  # object placement
  payload = {
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "system", 
        "content": "You are tasked with rephrasing location instructions into a more free-form, natural-sounding English sentence.\
                    The original instruction follows the fixed format: Put the <object> to <direction> of the <anchor object>.\
                    When rephrasing: You must not change the anchor object or the direction. You can freely vary sentence structure, add natural phrasing, \
                    and slightly expand the sentence to sound more human-like. Maintain the original meaning exactly. But do not use a question. Avoid rigid or robotic phrasing. \
                      The output should only contain the rephrased instruction, without any additional text or explanations. Each instruction one line."
                },
      {
          "role": "assistant",
          "content": """
              Here are the example:\n
              Assume the given fixed-form instrction is: Put the cup on the Right of the bootle. \n
              The output can be: Help me place the cup on the Right of the bootle, I want to drink water. \n

              Another example:\n
              Assume the given fixed-form instrction is: Put the keyboard in Front of the monitor. \n
              The output can be: Place the keyboard in Front of the monitor, I want to type with it.\n

              Now input the new fixed-form instruction, I will help you to convert it to free-form instruction.\n
          """
            },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": f"{fixed_form}"
          }
        ]
      }
    ],
    "max_tokens": 200
  }


  
  reponse = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  context = reponse.json()['choices'][0]['message']['content']
  context_list= context.split("\n")

  return context



if __name__ == "__main__":
  from sklearn.metrics.pairwise import cosine_similarity
  sentences = [
      "A monitor is on the working desk.",
      "A keyboard is on the working desk.",
      "A laptop is on the working desk.",
      "A plate is on the working desk.",
      "A bowl is on the working desk.",
      "A cup is on the working desk.",
      "A mouse is on the working desk.",
      "A power strip is on the working desk.",
      "A bottle is on the working desk.",
      "A can is on the working desk.",
      "A book is on the working desk.",
      "A pen is on the working desk.",
      "A paper is on the working desk.",
  ]

  # 计算每个句子的log概率并排序
  log_probs = {sentence: get_sentence_probability(sentence) for sentence in sentences}
  sorted_log_probs = sorted(log_probs.items(), key=lambda x: x[1])

  # # 输出按可能性排序的句子
  # for sentence, score in sorted_log_probs:
  #     print(f"Sentence: '{sentence}', Probability: {score:.6f}")

  # 按照从高到底排序
  sorted_log_probs.reverse()
  # 输出按可能性排序的句子
  for sentence, score in sorted_log_probs:
      print(f"Sentence: '{sentence}', Probability: {score:.6f}")
  