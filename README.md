# DarkHex3 & PhantomTTT

High-performance implementation of DarkHex3 / Abrupt DarkHex3 / PhantomTTT / Abrupt PhantomTTT, with support for
exploitability and best response computation.

## Building

This project is packaged with [pixi](https://prefix.dev/), which you can think of as a "local" conda (and in fact it pulls packages from the conda repository).

To create the pixi environment, you can do
```
> $ pixi install
✔ The default environment has been installed.
```

To activate the environment, you can then do
```
> $ pixi shell
```

To make sure everything works well you can do
```
> $ python
>>> import dh3
```

To re-build the environment (eg. after having modified the C++ code), run
```
> $ pixi clean
```
then follow the steps above to re-create and activate the environment.

Alternatively, you can install the environment with 
```
> $ pip install -e . -Ceditable.rebuild=True -Cbuild-dir=pypy_build
```
where `pypy_build` is any folder and the project will recompile if needed on import.
Note that this is terrible if the there are multiple processes using an NFS and that 
pixi has a tendency to reinstall the project w/o these additional options when ran with
the `install` command.

## Python interface

### Game state

There are four state objects defined for now:
- `DhState`
- `AbruptDhState`
- `PtttState`
- `AbruptPtttState`

Here is an example of API, which is common to all games.

```
import dh3

s = dh3.DhState() # Constructs a new initial state
assert(s.is_terminal() == False)
assert(str(s) ==
r"""** It is Player 1's turn
** Player 1's board:
                _____
               /     \
         _____/   2   \_____
        /     \       /     \
  _____/   1   \_____/   5   \_____
 /     \       /     \       /     \
/   0   \_____/   4   \_____/   8   \
\       /     \       /     \       /
 \_____/   3   \_____/   7   \_____/
       \       /     \       /
        \_____/   6   \_____/
              \       /
               \_____/

** Player 2's board:
                _____
               /     \
         _____/   2   \_____
        /     \       /     \
  _____/   1   \_____/   5   \_____
 /     \       /     \       /     \
/   0   \_____/   4   \_____/   8   \
\       /     \       /     \       /
 \_____/   3   \_____/   7   \_____/
       \       /     \       /
        \_____/   6   \_____/
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
assert(s.is_terminal() == False)
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
         _____/   2   \_____
        /XXXXX\       /     \
  _____/X  1  X\_____/   5   \_____
 /XXXXX\X t=2 X/     \       /     \
/X  0  X\XXXXX/   4   \_____/   8   \
\X t=1 X/     \       /     \       /
 \XXXXX/   3   \_____/   7   \_____/
       \       /     \       /
        \_____/   6   \_____/
              \       /
               \_____/

** Player 2's board:
                _____
               /OOOOO\
         _____/O  2  O\_____
        /     \O t=2 O/     \
  _____/   1   \OOOOO/   5   \_____
 /     \       /     \       /     \
/   0   \_____/   4   \_____/   8   \
\  t=1  /     \       /     \       /
 \_____/   3   \_____/   7   \_____/
       \       /     \       /
        \_____/   6   \_____/
              \       /
               \_____/
""")
# Under each cell ID, a timestamp of the form "t=X" denotes the time (from
# the point of view of the player) at which that cell was probed or filled.
```

Once the game is over, `s.winner()` contains the winner. It can be `0`, `1` or `None` in case of a tie (only applicable to PTTT).

### Player goals

In DH and Abrupt DH:
- Player 1 wants to connect down-right (cells {0,1,2} with {6,7,8}).
- Player 2 wants to connect up-right (cells {0,3,6} with {2,5,8}).

In PTTT and Abrupt PTTT, each player wants to put three of their symbols in a line as usual.

### Traversers and strategy representation

Anything related to manipulating the game tree, computing exploitability, et cetera goes through a "Traverser", which is able to quickly expand the game tree. There are four traverser objects implemented:
- `DhTraverser`
- `AbruptDhTraverser`
- `PtttTraverser`
- `AbruptPtttTraverser`

In order to compute exploitability and expected values, the library expects the
input strategies to be in a specific tensor format. The library supports the numpy
representation, which can be extracted from torch using the `.numpy()` method.

The strategy tensor for player 1 must have shape `(traverser.NUM_INFOS_PL1, 9)`, and for Player 2 it 
must have shape `(traverser.NUM_INFOS_PL2, 9)`. For reference, `NUM_INFOS_PL1 = 3720850` and `NUM_INFOS_PL2 = 2352067`
for (regular, non-abrupt) DarkHex3.
.

Each row of the tensor contains the strategy for each of the possible infosets of the game. It is mandatory that the probability of illegal actions be `0.0`.

```
import dh3

t = dh3.DhTraverser()  # This takes roughly 55s on my machine.
(x, y) = t.construct_uniform_strategies()
assert(x.shape == (t.NUM_INFOS_PL1, 9))
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
# [1723044495.952|>INFO] [traverser.cpp:383] ... all done. (ev0 = 0.333684, expl = 1.166488, 0.666318)
print(ret.ev0, ret.expl)
```

The correspondence between rows and information set can be recovered by using the function `traverser.infoset_desc(player, row_number)`. For example, `traverser.infoset_desc(0, 12345)` returns `'5*8*4.0.'`. This means that row `12345` of the strategy tensor of Player `0` corresponds to the strategy used by that player when its their turn assuming the observations they made was: they placed a piece on cell `5`, and it went through (`*`); then they played on cell `8` and it went through; then they played on cell `4`, but it was found occupied (`.`); then they played on `0` and it was occupied; and now it is their turn.
