import os
from typing import Optional
import openai #1.45.1
import base64
import requests
import numpy as np
import ast
from GeoL_net.gpt.gpt import convert_form_fix2free
from metrics.dataset_factory import BlendprocDesktopDataset_incompleted_sparse, BlendprocDesktopDataset_incompleted_mult_cond, realworld_dataset

from torch.utils.data import DataLoader
import json
import torch


if __name__ == "__main__":

    
    # # genrate the free form instruction and save as json 

    #dataset = BlendprocDesktopDataset_incompleted_mult_cond()
    dataset = realworld_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    scene_instrction = {}

    for data_batch in dataloader:
        object_to_place = data_batch["obj_to_place"][0]
        anchor_obj_name = data_batch["ref_objects"]
        direction = data_batch["directions"]
        file_path = data_batch["json_file"][0]
        list_anchor = [item[0] for item in anchor_obj_name]
        list_direction = [item[0] for item in direction]
        print(direction)
        free_form_instruction = convert_form_fix2free(list_anchor, list_direction, object_to_place) 
        print(free_form_instruction)
        print(object_to_place)
        scene_instrction[file_path] = [free_form_instruction]
    
    # # save as json
    with open('metrics/scene_instruction_realworld.json', 'w') as f:
         json.dump(scene_instrction, f)