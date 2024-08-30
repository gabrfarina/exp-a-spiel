import dh3
import time

t = dh3.DhTraverser()
cfr = t.make_cfr_solver(
    dh3.CfrConf(
        avg=dh3.AVERAGING_STRATEGY.LINEAR,
        alternation=True,
        dcfr=True,
        rmplus=True,
        pcfrp=True,
    ),
)

for i in range(10):
    cfr.step()
    res = t.ev_and_exploitability(cfr.avg_bh())
    print(res)