# python doe_run.py --plan experiment_plan.csv --name doe1 --n-threads 10 --no-cuda
# python doe_run.py --plan experiment_plan.csv --name doe1 --n-threads 2 --no-cuda --no-run

import pandas as pd
import subprocess
import os
import time
import argparse


def prepare_command(config):
    cmd = 'python main_zero_rand.py '
    for key, value in config.items():
        cmd += f" --{key} {value}"
    return cmd


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experiment runner')
    parser.add_argument('--name', type=str, default="doe1", metavar='S',
                        help='DOE name. Used to construct model/log folders')
    parser.add_argument('--plan', type=str, default="experiment_plan.csv", metavar='S',
                        help='DOE plan .csv file')

    # Execution options
    parser.add_argument('--no-run', action='store_true', default=False,
                        help='Do not run the experiments, just print the commands')
    parser.add_argument('--n-threads', type=int, default=2,
                        help='number of CPU threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='dry run')

    args = parser.parse_args()

    doe_name = args.name
    df_plan = pd.read_csv(args.plan)
    df_plan.set_index("experiment_id", inplace=True)
    log_dir = os.path.join("logs", doe_name)
    model_dir = os.path.join('models', doe_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    project_root = os.path.abspath(os.path.join("..", ".."))  # 2 levels up
    env = os.environ
    env["PYTHONPATH"] = project_root  # including the right PYTHONPATH

    n_exp = df_plan.shape[0]
    cnt = 0
    for row in df_plan.itertuples(index=True):
        cnt += 1
        row = row._asdict()  # dict(row) seems to fail...
        row[df_plan.index.name] = row["Index"]  # row["experiment_id"]
        del row["Index"]
        exp_id = row[df_plan.index.name]

        log_name = os.path.join(log_dir, f"log_{exp_id}")
        if os.path.exists(log_name):
            print(f"Skipping simulation {cnt } of {n_exp}...")
            continue  # simulation already there, skip...

        cmd_in = prepare_command(row)
        cmd_in += f" --save-folder {model_dir}"
        cmd_in += f" --n-threads {args.n_threads}"
        if args.no_cuda:
            cmd_in += f" --no-cuda"
        if args.dry_run:
            cmd_in += f" --dry-run"
        cmd_in += f" --no-figures"
        if args.no_run:
            print(cmd_in)
            continue  # do not execute experiment, just print the command!
        print(f"Running simulation {cnt } of {n_exp}...")
        print(cmd_in)
        # run example
        time_start = time.time()
        cmd_out = subprocess.check_output(cmd_in.split(), env=env).decode()
        run_time = time.time() - time_start
        # save text log
        log_msg = cmd_in + "\n" + cmd_out + f"\nRun time: {run_time}"
        with open(log_name, "w") as f:
            f.write(log_msg)
