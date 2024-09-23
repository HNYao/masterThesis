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
image_path = "dataset/scene_RGBD_mask/id747_2/bottle_0003_cola/with_obj/test_pbr/000000/rgb/000000.jpg"

base64_image = encode_image(image_path)

api_key = os.getenv("OPENAI_API_KEY")


headers = {
    "content-type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


payload = {
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Where should I put a monitor?Give me a anchor object and a direction."
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

'''
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4"

messages = [
                {
                    "role": "system", 
                    "content": "You are an assistant that helps people place objects on the table.\
                          You are given a image of the tabletop and an target object to be placed. \
                            You should determine the anchor object and in which direction the target object should be placed relative to the anchor object."
                            },
                {
                    "role": "assistant",
                    "content": """
                        Here are the examples:
                        Assume the given image contains: Monitor, cup, phone, can, bottle, book.
                        Please note that anchors should be split by ",".
                        1. Keyboard. Please answer:
                            target: monitor
                            anchor: front
                        2. Mouse, I am a right-handed. Please answer:
                            target: Monitor
                            anchor: right
                        3. Mouse, I am a left-handed. Please answer:
                            target: monitor 
                            anchor: left
                        4. bottle. Please answer:
                            target: bottle
                            anchor: behind
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                        The room contains: {object_string}. The text description is: {text}. 
                        Please make sure that the target object and anchor object are in the room. If you cannot find the answer, just make a guess.
                        Answer should be in the following format without any explanations: target: <target object>\nanchor: <anchor object>\n
                    """,
                }
            ]
'''