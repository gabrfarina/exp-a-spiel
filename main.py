import dh3
import time
import logging
import numpy as np
import itertools
import argparse
import submitit
from pathlib import Path

logging.basicConfig(format="[%(levelname)s][%(name)s][%(asctime)s] %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)



def train(
    avg:dh3.AveragingStrategy,
    alternation:bool,
    dcfr:bool,
    rmplus:bool,
    pcfrp:bool,
    game,
    N = 1000,
):
    t = dh3.CornerDhTraverser()
    cfr = dh3.CfrSolver(
        t,
        avg=dh3.AveragingStrategy.LINEAR,
        alternation=True,
        dcfr=True,
        rmplus=True,
        pcfrp=True,
    )
    logger.info("Starting CFR")
    # x1, y1 = t.construct_uniform_strategies()
    x2, y2 = cfr.avg_bh()
    # logger.info("delta: %s, %s" % (np.max(np.abs(x1 - x2)), np.max(np.abs(y1 - y2))))
    # logger.info("uniform expo %s" % t.ev_and_exploitability(*t.construct_uniform_strategies()))
    # logger.info("cfr init expo %s" % t.ev_and_exploitability(*cfr.avg_bh()))
    expos = []
    for i in range(1, N + 1):
        cfr.step()
        logger.info(f"done step {i}")
        x1, y1 = cfr.avg_bh()
        if i % 10 == 0 or i == N:
            expo = t.ev_and_exploitability(*cfr.avg_bh())
            expos.append(expo)
            logger.info("expo %s" % expo)
    return expos

if __name__ == "__main__":
    project_dir = Path(__file__).parent.resolve()
    (project_dir / "exps").mkdir(exist_ok=True)
    
    executor = submitit.SlurmExecutor(folder=project_dir / "exps")

    executor.update_parameters(
        exclusive=True,
        mem=0,
        time=48*60,
    )

    with executor.batch():
        for (avg, alter, dcfr, rmplus, pcfrp, game) in itertools.product(
            [dh3.AveragingStrategy.UNIFORM, dh3.AveragingStrategy.LINEAR, dh3.AveragingStrategy.QUADRATIC], 
            [True, False], # alter
            [True, False], # dcfr
            [True, False], # rmplus
            [True, False], # pcfrp
            [dh3.CornerDhTraverser, dh3.DhTraverser, dh3.AbruptDhTraverser, dh3.PtttTraverser, dh3.AbruptPtttTraverser]
        ):
            executor = submitit.AutoExecutor(folder="submitit_jobs")
            executor.update_parameters(
                timeout_min=60 * 24,
                slurm_partition="dev",
                cpus_per_task=1,
                gpus_per_node=1,
                nodes=1,
                slurm_mem="32G",
                dry_run=True,
            )
            job = executor.submit(train, avg, alter, dcfr, rmplus, pcfrp, game)
            print(job.job_id)

            print(job)