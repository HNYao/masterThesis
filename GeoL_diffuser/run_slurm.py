import os, sys
import pathlib
import time
import options
from options import to_cmd


def main(opt_cmd):
    job_name = "GeoL_diffusion"
    slurm_log_dir = pathlib.Path(
        f"/home/stud/zhoy/MasterThesis_zhoy/outputs/slurm/logs/{job_name}"
    ).expanduser()
    slurm_log_dir.mkdir(exist_ok=True, parents=True) 
    file_name = time.strftime("%Y-%b-%d-%H-%M-%S")
    job_file: pathlib.Path = slurm_log_dir / f"{file_name}.job"
    out_file: pathlib.Path = slurm_log_dir / f"{file_name}.out"
    error_file: pathlib.Path = slurm_log_dir / f"{file_name}.out"
    script_cmd = ""
    opt_cmd.update({"log_dir": "/home/stud/zhoy/MasterThesis_zhoy/outputs/logs/GeoL_diffusion/"})

    ckpt_dir = pathlib.Path(opt_cmd["log_dir"]) #/ pathlib.Path(opt_cmd["name"])
    if ckpt_dir.exists():
        print(
            f"====> ckpt dir {ckpt_dir} exists, delete it first, or change a new name for your experiment!!"
        )
        return

    script_cmd = to_cmd(opt_cmd)
    print(f"... Running command: python scripts/train.py {script_cmd}")
    print(f"... Checking logs using: \n tail -f {out_file}  ")
    print("==========================================================")

    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={job_name}\n")
        fh.writelines(f"#SBATCH --output={out_file.as_posix()}\n")
        fh.writelines(f"#SBATCH --error={error_file.as_posix()}\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --mem=12G\n")
        fh.writelines("#SBATCH --gres=gpu:1,VRAM:10G\n")
        fh.writelines("#SBATCH --cpus-per-task=4\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --time=72:00:00\n")
        # fh.writelines("#SBATCH --partition=DEADLINEBIG\n")
        # fh.writelines("""#SBATCH --comment="RAL" \n""")
        #fh.writelines("conda activate dift\n")
        # fh.writelines(
        #     "export PYTHONPATH=${PYTHONPATH}:/home/wiss/chenh/hand_object_percept/egoprior-diffuser/ \n"
        # )
        fh.writelines(f"srun /home/stud/zhoy/anaconda3/envs/o2o/bin/python /home/stud/zhoy/MasterThesis_zhoy/GeoL_diffuser/run.py \n")

    os.system("sbatch %s" % job_file)


if __name__ == "__main__":
    opt_cmd = options.parse_arguments(sys.argv[1:])
    main(opt_cmd)