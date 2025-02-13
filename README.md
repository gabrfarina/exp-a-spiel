# exp-a-spiel

This package (C++ with python binding) implement fast gradient, exploitability and tabular solvers for
- dark Hex 3,
- abrupt Dark Hex,
- pantom tic-tac-toe, and
- abrupt phatom tic-tac-toe.

This package is compatible with [OpenSpiel](https://github.com/google-deepmind/open_spiel).


This package was part of TODO


# Hight level concepts:

## Games

[`Traverser`](#traverser), [`State`](#state), and [`CfrSolver`](#cfr) are defined per game, we use the prefix
- `Dh` for dark hex 3,
- `AbruptDh` for abrupt dark hex 3,
- `Pttt` for phantom tic-tac-toe, and
- `AbruptPttt` for abrupt phantom tic-tac-toe.

Lastly, we provide `CornerDh`, a game of DarkHex where the first two moves are determined for debugging.

## `Traverser`
Every game defines a `Traverser` class. The traverser can:

Return a tensor of the infostates of `player` where each row is one infostate.
```python
def compute_openspiel_infostates(self, player: int) -> np.ndarray
```
Creats a uniform strategy for both players.
```python
def construct_uniform_strategies(self) -> tuple[np.ndarray, np.ndarray]
```
Calculate the expected value and exploitability of both policies. See [`EvExpl`](#evexpl)
```python
def ev_and_exploitability(self, strat0: np.ndarray, strat1: np.ndarray) -> EvExpl
```
Return a visual representation of the infoset `row` of `player`.
```python
def infoset_desc(self, player: int, row: int) -> str
```
Invert the output of `infoset_desc`
```python
def row_for_infoset(self, player: int, infoset_desc: str) -> int
```
Create an [`Averager`](#averager) for `player`. See [`Averager`](#averager) for a description of averaging strategies.
```python
def new_averager(self, player: int, avg_strategy: AveragingStrategy) -> Averager
```
Returns a row and an action. The row is the row of the parent of `row` and the action is the action taken at the parent to reach `row`. Note that multiple rows may have the same `parent_index_and_action` as the outcome of an action may depend on the opponent or chance.
```python
def parent_index_and_action(self, player: int, row: int) -> tuple[int, int]
```
Returns a new cfr solver. See [CFR](#cfr)
```python
def cfr_solver(self, conf: CfrConf)
```

Furtheremore, a traverser defines the following constant attributes:

Number of infostates for each of the player. The strategy arrays are of this size.
```python
NUM_INFOS_PL0: int
NUM_INFOS_PL1: int
```

Note that all games have 9 actions and that strategies are arrays of size `(NUM_INFOS_PL0, 9)` and `(NUM_INFOS_PL1, 9)`.

## `EvExpl`

This helper class defines the following attribtues:

the best reponse strategies
```python
best_reponse: tuple[np.ndarray, np.ndarray]`
```
the best response values
```python
expl: tuple[float ,float]
```
head to head expected value of player 0. the head-to-head value of player 1 is just the inverse.
```python
ev0
```
the gradients. See [Sequence form strategies](#sequence-form-strategies)
```python
gradient: tuple[np.ndarray, np.ndarray]`
```


## `Averager`

In general, sound algorithms for 2p0s games require calculating the [sequence form average](#sequence-form-strategies) of the behavioral policies. The averager converts the behavioral policy to a sequene form policy and take the average. The contribution of iterate $t$ to the final policy is detemined by the `AveragingStrategy`. The final result is proportional to
- $1$ for `UNIFORM`,
- $t$ for `LINEAR`,
- $t^2$ for `QUADRATIC`, and
- ?? for `EXPERIMENTAL`.
Furthermore, the `AveragingStrategy` can be last to always select the last iteration or `CUSTOM` if the user provides the weights.

The `Averager` defines two methods:
```python
def push(self, start: np.ndarray, weight: float | none) -> None
```
`strat` is a behavior strategy and `weight` only need to be defined if the averaging strategy is `CUSTOM`.

```python
def running_avg(self) -> np.ndarray
```
Returns the current average

```python
def clear(self) -> None
```
resets the averager.


Note that you need one averager per player and that the player is set during the creation of the averager by the [`Traverser`](#traverser).


## `CFR`

This codebase implements many CFR variants. You can configure them with a `CfrConf` object.
It defines:
a constructor
```python
def __init__(self, avg: AveragingStrategy, alternation: bool, dcfr: bool, rmplus: bool, predictive: bool) -> CfrConf
```
- `avg` is the strategy used for averaging, [see `Averager`](#averager)
- `alternation` make CFR update only one policy per gradient
- `dcfr` will discount the regrets, the default parameter of $\alpha=1.5$ and $\beta=0$ are used. See [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040)
- `rmplus` will set the regrets to zero if they are negative. If used in conjuncation with `dcfr`, it will be applied after discounting.
- `predictive` will use optimistic regret minimizers. See [Stable-Predictive Optimistic Counterfactual Regret Minimization](https://arxiv.org/abs/1902.04982).

We provide the following static variants:
- `PCFRP`: predictive cfr+, and
- `DCFR`: discounted cfr.

The CFR object has tow methods
```python
def step(self) -> None: ...
def avg_bh(self) -> tuple[np.ndarray, np.ndarray]: ...
```
They run one step of the algorithm and return the average strategy respectively.


## State

Each game define a State object, it defines 

```python
def clone(self) -> Self
```
Return a clone.


```python
def player(self) -> int
```
Returns the active player
```python
def next(self, action: int) -> None
```
Updates (inplace) the state by playing `action`
```python
def is_terminal(self) -> bool
```
Returns true if the game has ended
```python
def winner(self) -> int | None
```
Returns the winning player or `None` in case of a tie.
```python
def action_mask(self) -> np.array
```
Returns the valid action mask. The probabilities of invalid actions must be 0.
```python
def infoset_desc(self) -> str
```
Returns a human readable description of the infoset
```python
def compute_openspiel_infostate(self) -> np.ndrarray
```
Returns the OpenSpiel infostate of the active player.


## Sequence form strategies

Please refer to the paper for the strategy




## C++ interface

The python interface is a clone of the C++ interface except that C++ uses valarrays instead of `np.array`s.



# An example
```
import eas
 # Constructs a new initial state
s = eas.DhState()
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

## Player goals

In DH and Abrupt DH:
- Player 1 wants to connect down-right (cells {0,1,2} with {6,7,8}).
- Player 2 wants to connect up-right (cells {0,3,6} with {2,5,8}).

In PTTT and Abrupt PTTT, each player wants to put three of their symbols in a line as usual.

## Traversers and strategy representation

<!-- Anything related to manipulating the game tree, computing exploitability, et cetera goes through a "Traverser", which is able to quickly expand the game tree. There are four traverser objects implemented:
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
. -->

Each row of the tensor contains the strategy for each of the possible infosets of the game. It is mandatory that the probability of illegal actions be `0.0`.

```
import eas

t = eas.DhTraverser()  # This takes roughly 55s on my machine.
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

## Building

This project is packaged with [pixi](https://prefix.dev/), a conda replacement.

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
>>> import eas
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
