import dh3
import time
import logging
import numpy as np
import itertools
import argparse
import submitit
from pathlib import Path


def train(game, N=1000, **cfr_config):
    logging.basicConfig(format="[%(levelname)s][%(name)s][%(asctime)s] %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    print(flush=True)
    logger.info(f"Training {game.__name__} with %s", cfr_config)
    time.sleep(1)
    print(flush=True)
    t = game()
    cfr = dh3.CfrSolver(t, **cfr_config)
    logger.info("Starting CFR")

    x2, y2 = cfr.avg_bh()
    expos = []
    for i in range(1, N + 1):
        cfr.step()
        logger.info(f"done step {i}")
        x1, y1 = cfr.avg_bh()
        if i % 10 == 0 or i == N:
            expo = t.ev_and_exploitability(*cfr.avg_bh())
            expos.append(expo)
            logger.info("expo %s" % expo)
            print(flush=True)
    return expos


if __name__ == "__main__":
    project_dir = Path(__file__).parent.resolve()
    (project_dir / "exps").mkdir(exist_ok=True)

    executor = submitit.SlurmExecutor(folder=project_dir / "exps")

    executor.update_parameters(
        exclusive=True,
        nodes=1,
        mem=0,
        time=48 * 60,
        additional_parameters={"partition": "cpu"},
    )

    with executor.batch():
        for avg, alter, dcfr, rmplus, pcfrp, game in itertools.product(
            [
                dh3.AveragingStrategy.UNIFORM,
                dh3.AveragingStrategy.LINEAR,
                dh3.AveragingStrategy.QUADRATIC,
            ],
            [True, False],  # alter
            [True, False],  # dcfr
            [True, False],  # rmplus
            [True, False],  # pcfrp
            [
                dh3.CornerDhTraverser,
                dh3.DhTraverser,
                dh3.AbruptDhTraverser,
                dh3.PtttTraverser,
                dh3.AbruptPtttTraverser,
            ],
        ):
            train(
                game,
                avg=avg,
                alternation=alter,
                dcfr=dcfr,
                rmplus=rmplus,
                pcfrp=pcfrp,
            )
            # job = executor.submit(
            #     train,
            #     game,
            #     avg=avg,
            #     alter=alter,
            #     dcfr=dcfr,
            #     rmplus=rmplus,
            #     pcfrp=pcfrp,
            # )
