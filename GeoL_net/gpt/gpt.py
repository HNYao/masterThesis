import os
import sys
import time
from typing import Optional
import openai
import base64
import requests

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def chatgpt_condition(image_path: str, mode="object_placement"):
    """
    ChatGPT condition for object placement or scene understanding

    Params:
    image_path: image path
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
      object_placement = input("Object need to be placed: ")
      extra_info = input("Extra information: ")
      payload = {
        "model": "gpt-4o-mini",
        "messages": [
          {
            "role": "system", 
            "content": "You are an assistant that helps people place objects on the table.\
                  You are given a image of the tabletop and an target object to be placed. \
                    You should determine the anchor object with color description and in which direction the target object should be placed relative to the anchor object."
                    },
          {
          "role": "assistant",
          "content": """
              Here are the examples:
              Assume the given image contains: white monitor, blue cup, blue phone, red can, black bottle, green book.
                Please note that anchors should be split by ",".
                1. Mouse. I am a right-handed. Please answer:
                    anchor: white monitor
                    direction: right front
                2. Mouse. I am a left-handed. Please answer:
                    anchor: white monitor 
                    direction: left front
                3. bottle. Please answer:
                    anchor: black bottle
                    direction: front
                4. phone. I like playing mobile games and drinking coffee. Please answer:
                    anchor: blue cup
                    direction: right
          """,
            },
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": f"Base on the image, where should I put {object_placement} reasonably without collision and overlap with other objects? Please attention: {extra_info} Answer should be in the following format without any explanations: anchor: <target object>\ndirection: <direction>\n"
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

        if user_input.lower() == 'exit':
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
    anchor, diraction = extract_response(reponse.json()['choices'][0]['message']['content'])
    return anchor, diraction

def extract_response(response: str) -> str:
    """
    Extract the anchor and direction from the chatgpt response

    Params:
    response: chatgpt response

    Return:
    anchor, direction
    """
    response = response.split("\n")
    anchor = response[0].split(":")[1].strip()
    direction = response[1].split(":")[1].strip()
    return anchor, direction

if __name__ == "__main__":
    image_path = "dataset/scene_RGBD_mask/id695_2/keyboard_0004_normal/no_obj/test_pbr/000000/rgb/000000.jpg"
    #mode = "object_placement"
    mode = "scene_understanding"
    anchor, direction = chatgpt_condition(image_path, mode)
    print(anchor, direction)