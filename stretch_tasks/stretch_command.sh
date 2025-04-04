hostname -I
scp hello-robot@192.168.1.2:/home/hello-robot/stretch_manip/config_base_cam.json /home/cvai/hanzhi_ws/HephaisBot/masterThesis/stretch_config
ssh hello-robot@192.168.1.2
cd stretch_manip && conda activate hellorobot
export PYTHONPATH="${PYTHONPATH}:$PWD" && export PYTHONPATH="${PYTHONPATH}:/usr/lib/python3/dist-packages"   
python robot_server/start_server.py