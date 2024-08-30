import dh3
import time
import logging
logging.basicConfig(format='[%(levelname)s @ %(name)s %(asctime)s] %(message)s')
logger = logging.getLogger(__file__)
t = dh3.DhTraverser()
cfr = t.make_cfr_solver(
    dh3.CfrConf(
        avg=dh3.AveragingStrategy.LINEAR,
        alternation=True,
        dcfr=True,
        rmplus=True,
        pcfrp=True,
    ),
)
logger.info("Starting CFR")
for i in range(10):
    cfr.step()
    res = t.ev_and_exploitability(cfr.avg_bh())
    print(res)