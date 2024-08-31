import dh3
import time
import logging
import numpy as np

logging.basicConfig(format="[%(levelname)s @ %(name)s %(asctime)s] %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
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
for i in range(10000):
    cfr.step()
    logger.info(f"done step {i}")
    x1, y1 = cfr.avg_bh()
    logger.info("delta: %s, %s" % (np.max(np.abs(x1 - x2)), np.max(np.abs(y1 - y2))))
    logger.info("expo %s" % t.ev_and_exploitability(*cfr.avg_bh()))
