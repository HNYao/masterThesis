import subprocess
import time

# BlenderProc 运行命令
command = [
    "blenderproc", "run",
    "/home/stud/zhoy/MasterThesis_zhoy/GeoL_net/dataset_gen/mesh_scene_gen_mask_bproc_v2_multi_view.py"
]

def run_blenderproc():
    """ 运行 BlenderProc 并在检测到 'RESTART' 时重新启动 """
    while True:
        print("启动 BlenderProc...")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        restart_flag = False  # 标记是否需要重启
        for line in iter(process.stdout.readline, ''):
            print(line, end="")  # 实时打印输出
            if "RESTART" in line:  # 检测到 'RESTART' 信号
                print("检测到 'RESTART' 信号，重新运行 BlenderProc...")
                restart_flag = True
                process.kill()  # 终止进程
                break  # 跳出循环，准备重启
        
        process.wait()  # 确保进程正确结束
        if not restart_flag:
            print(f"BlenderProc 进程结束，退出代码: {process.returncode}")

        # 2 秒冷却后重启
        print("重新启动 BlenderProc...")
        time.sleep(2)

if __name__ == "__main__":
    run_blenderproc()