# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import uuid
from pathlib import Path

import tools.train_net
import submitit

def parse_args():
    default_parser = tools.train_net.default_argument_parser()
    parser = argparse.ArgumentParser("Submitit for ConvNeXt", parents=[default_parser], add_help=False)
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=120, type=int, help="Duration of the job, in hours")
    parser.add_argument("--job_name", default="convnext", type=str, help="Job name")
    parser.add_argument("--job_dir", default="", type=str, help="Job directory; leave empty for default")
    parser.add_argument("--partition", default="learnai", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/fsx/wufeim/checkpoint/").is_dir():
        p = Path(f"/fsx/wufeim/checkpoint/omni3d")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from detectron2.engine import launch
        import tools.train_net

        self.args.dist_url = get_init_file().as_uri()
        job_env = submitit.JobEnvironment()

        self._setup_gpu_args()
        # tools.train_net.main(self.args)
        launch(
            tools.train_net.main,
            self.args.num_gpus,
            num_machines=self.args.nodes,
            machine_rank=job_env.global_rank,
            dist_url=get_init_file().as_uri(),
            args=(self.args,))

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        self.args.auto_resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(self.args.job_dir)
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()

    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout * 60

    partition = args.partition
    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.job_name)

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)

if __name__ == "__main__":
    main()
