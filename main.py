import dh3
import time

cfr_conf = dh3.CfrConf()
cfr_conf.linear = True

t = dh3.DhTraverser()
cfr_buffer = t.init_cfr()
for i in range(10):
    t.update_cfr(cfr_conf, i % 2, cfr_buffer)  
    print(f"i: {i}, ev1: {cfr_buffer.ev[0]}, ev2: {cfr_buffer.ev[1]}")
ret = t.ev_and_exploitability(*cfr_buffer.bh) 
print(f"ev: {ret.ev}, exploitability: {ret.exploitability}")
