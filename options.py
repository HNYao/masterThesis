import numpy as np
import os,sys,time
import torch
import random
import string
import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

def parse_arguments(args):
    """
    Parse arguments from command line.
    Syntax: --key1.key2.key3=value --> value
            --key1.key2.key3=      --> None
            --key1.key2.key3       --> True
            --key1.key2.key3!      --> False
    """
    opt_cmd = {}
    for arg in args:
        assert(arg.startswith("--"))
        if "=" not in arg[2:]:
            key_str,value = (arg[2:-1],"false") if arg[-1]=="!" else (arg[2:],"true")
        else:
            key_str,value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub: opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub,keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd

def set(opt_cmd={}, safe_check=True):
    # load config from yaml file
    assert("config" in opt_cmd)
    fname = opt_cmd.config
    opt_base = load_options(fname)
    # override with command line arguments
    opt = override_options(opt_base,opt_cmd,key_stack=[],safe_check=safe_check)
    process_options(opt)
    return opt

def load_options(fname):
    opt = OmegaConf.load(fname)
    opt = OmegaConf.to_container(opt)
    opt = edict(opt)
    print("loading {}...".format(fname))
    return opt

def override_options(opt,opt_over,key_stack=None,safe_check=False):
    for key,value in opt_over.items():
        if key == "config": continue
        if isinstance(value,dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key,dict()),value,key_stack=key_stack+[key],safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y","n"]:
                    key_str = ".".join(key_stack+[key])
                    add_new = input("\"{}\" not found in original opt, add? (y/n) ".format(key_str))
                if add_new=="n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt

def process_options(opt):
    pass

def to_dict(D,dict_type=dict):
    D = dict_type(D)
    for k,v in D.items():
        if isinstance(v,dict):
            D[k] = to_dict(v,dict_type)
    return D

def to_cmd(opt,entry=''):
    cmd = ""
    for key,value in to_dict(opt).items():
        if isinstance(value,dict):
            cmd += to_cmd(value, entry=entry+key+'.')
        else:
            cmd += " --{}={} ".format(entry+key,value)

    return cmd

def save_options_file(opt):
    output_path = os.path.join(opt.log_dir, opt.name)
    os.makedirs(output_path,exist_ok=True)
    opt_fname = "{}/config.yaml".format(output_path)
    to_save = to_dict(opt)
    if os.path.isfile(opt_fname):
        with open(opt_fname) as file:
            opt_old = yaml.safe_load(file)
        if opt!=opt_old:
            # prompt if options are not identical
            opt_new_fname = "{}/config_temp.yaml".format(output_path)
            with open(opt_new_fname,"w") as file:
                yaml.safe_dump(to_save,file,default_flow_style=False,indent=4)
            print("existing options file found (different from current one)...")
            os.system("diff {} {}".format(opt_fname,opt_new_fname))
            os.system("rm {}".format(opt_new_fname))
            override = None
            while override not in ["y","n"]:
                override = input("override? (y/n) ")
            if override=="n":
                print("safe exiting...")
                exit()
        else: print("existing options file found (identical)")
    else: print("(creating new options file...)")
    with open(opt_fname,"w") as file:
        yaml.safe_dump(to_save,file,default_flow_style=False,indent=4)

def print_options(opt,level=0):
    for key,value in sorted(opt.items()):
        if isinstance(value,(dict,edict)):
            print("   "*level+"* "+key+":")
            print_options(value,level+1)
        else:
            print("   "*level+"* "+key+":",value)