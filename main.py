import eas
import time
import logging
import itertools
import submitit
from pathlib import Path


def train(game, N=1000, cfr_config=None):
    logging.basicConfig(format="[%(levelname)s][%(name)s][%(asctime)s] %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    print(flush=True)
    logger.info(f"Training {game.__name__} with %s", cfr_config)
    time.sleep(1)
    print(flush=True)
    t = game()
    cfr = eas.CfrSolver(t, cfr_config)
    logger.info("Starting CFR")

    x2, y2 = cfr.avg_bh()
    expos = []
    for i in range(1, N + 1):
        cfr.step()
        logger.info(f"done step {i}")
        x1, y1 = cfr.avg_bh()
        if i % 10 == 0 or i == N:
            expo = t.ev_and_exploitability(*cfr.avg_bh())
            expos.append({"expo": expo.expl, "ev0": expo.ev0})
            logger.info("expo %s" % expo)
            print(flush=True)
    return expos


if __name__ == "__main__":
    project_dir = Path(__file__).parent.resolve()
    (project_dir / "exps").mkdir(exist_ok=True)

    executor = submitit.SlurmExecutor(folder=project_dir / "exps_new")

    executor.update_parameters(
        exclusive=True,
        nodes=1,
        mem=0,
        time=60 * 24 * 7,
        additional_parameters={"partition": "cpu"},
    )

    with executor.batch():
        for game, cfr in itertools.product(
            [
                eas.CornerDhTraverser,
                eas.DhTraverser,
                eas.AbruptDhTraverser,
                eas.PtttTraverser,
                eas.AbruptPtttTraverser,
            ],
            [
                eas.CfrConf.PCFRP,
                eas.CfrConf.DCFR,
            ]
        ):
            job = executor.submit(
                train,
                game,
                N=1000,
                cfr_config=cfr)
