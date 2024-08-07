# DarkHex3

High-performance implementation of DarkHex3 / Abrupt DarkHex3, with support for
exploitability and best response computation.

## Python interface

### Game state

```
import pydh3 as dh3

s = dh3.State() # Constructs a new initial state
assert(s.winner() == None)
assert(str(s) ==
r"""** It is Player 1's turn
** Player 1's board:
                _____
               /     \
         _____/   6   \_____
        /     \       /     \
  _____/   3   \_____/   7   \_____
 /     \       /     \       /     \
/   0   \_____/   4   \_____/   8   \
\       /     \       /     \       /
 \_____/   1   \_____/   5   \_____/
       \       /     \       /
        \_____/   2   \_____/
              \       /
               \_____/

** Player 2's board:
                _____
               /     \
         _____/   6   \_____
        /     \       /     \
  _____/   3   \_____/   7   \_____
 /     \       /     \       /     \
/   0   \_____/   4   \_____/   8   \
\       /     \       /     \       /
 \_____/   1   \_____/   5   \_____/
       \       /     \       /
        \_____/   2   \_____/
              \       /
               \_____/
""")
# The numbers denote the ID of each cell. No pieces has been placed yet.

assert(s.player() == 0) # 0 is player 1, 1 is player 2
s.next(0)               # Player 1 plays in cell 0
assert(s.player() == 1) # The turn has passed to player 1 now
s.next(0)               # Player 2 plays in cell 0 (which is occupied)...
assert(s.player() == 1) # ... so the turn does not pass to the opponent
s.next(2)               # Player 2 plays in cell 2
assert(s.winner() == None)
assert(s.player() == 0)
s.next(1)               # Player 1 plays in cell 1
assert(s.player() == 1)
# Player 2 has already probed cells 0 and 2, so placing there
# again is an illegal move
assert(s.action_mask() == [False, True, False, True, True, True, True, True, True])
assert(str(s) ==
r"""** It is Player 2's turn
** Player 1's board:
                _____
               /     \
         _____/   6   \_____
        /     \       /     \
  _____/   3   \_____/   7   \_____
 /XXXXX\       /     \       /     \
/X  0  X\_____/   4   \_____/   8   \
\X t=1 X/XXXXX\       /     \       /
 \XXXXX/X  1  X\_____/   5   \_____/
       \X t=2 X/     \       /
        \XXXXX/   2   \_____/
              \       /
               \_____/

** Player 2's board:
                _____
               /     \
         _____/   6   \_____
        /     \       /     \
  _____/   3   \_____/   7   \_____
 /     \       /     \       /     \
/   0   \_____/   4   \_____/   8   \
\  t=1  /     \       /     \       /
 \_____/   1   \_____/   5   \_____/
       \       /OOOOO\       /
        \_____/O  2  O\_____/
              \O t=2 O/
               \OOOOO/
""")
# Under each cell ID, a timestamp of the form "t=X" denotes the time (from
# the point of view of the player) at which that cell was probed or filled.
```

You can use `next_abrupt` instead of `next` to use the abrupt DH rule.

### Strategy representation

In order to compute exploitability and expected values, the library expects the
input strategies to be in a specific tensor format. The library supports the numpy
representation, which can be extracted from torch using the `.numpy()` method.

The strategy tensor for player 1 must have shape `(dh3.NUM_INFOS_PL1, 9)`, and for Player 2 it 
must have shape `(dh3.NUM_INFOS_PL2, 9)`. For reference, `NUM_INFOS_PL1 = 3720850` and `NUM_INFOS_PL2 = 2352067`.

Each row of the tensor contains the strategy for each of the possible infosets of the game. It is mandatory that the probability of illegal actions be `0.0`.

TODO: determine a good format to explain the correspondence between row and infoset---what is a convenient way to export the observations that define the infoset for you to query the net?

```
import pydh3 as dh3

t = dh3.Traverser()  # This takes roughly 55s on my machine.
(x, y) = t.construct_uniform_strategies()
assert(x.shape == (3720850, 9))
ret = t.ev_and_exploitability(x, y)   # This takes roughly 75s on my machine.
# Sample output:
#
# [1723044420.499|>INFO] [traverser.cpp:353] begin exploitability computation...
# [1723044420.499|>INFO] [traverser.cpp:299] begin gradient computation (num threads: 16)...
# [1723044433.797|>INFO] [traverser.cpp:335]   > 10/81 threads returned
# [1723044444.868|>INFO] [traverser.cpp:335]   > 20/81 threads returned
# [1723044449.731|>INFO] [traverser.cpp:335]   > 30/81 threads returned
# [1723044458.575|>INFO] [traverser.cpp:335]   > 40/81 threads returned
# [1723044465.895|>INFO] [traverser.cpp:335]   > 50/81 threads returned
# [1723044475.045|>INFO] [traverser.cpp:335]   > 60/81 threads returned
# [1723044483.914|>INFO] [traverser.cpp:335]   > 70/81 threads returned
# [1723044490.975|>INFO] [traverser.cpp:335]   > 80/81 threads returned
# [1723044495.729|>INFO] [traverser.cpp:340] ... aggregating thread buffers...
# [1723044495.907|>INFO] [traverser.cpp:348] ... all done.
# [1723044495.907|>INFO] [traverser.cpp:356] computing expected value...
# [1723044495.919|>INFO] [traverser.cpp:377] computing exploitabilities...
# [1723044495.952|>INFO] [traverser.cpp:383] ... all done. (ev = 0.333684, expl = 0.666318, 1.166488)
print(ret.ev, ret.expl)
```