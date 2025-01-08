from __future__ import annotations
import dh3
import numpy
import pybind11_stubgen.typing_ext
import typing
__all__ = ['AbruptDhCfrSolver', 'AbruptDhState', 'AbruptDhTraverser', 'AbruptPtttCfrSolver', 'AbruptPtttState', 'AbruptPtttTraverser', 'Averager', 'AveragingStrategy', 'CfrConf', 'CfrSolver', 'CornerDhCfrSolver', 'CornerDhState', 'CornerDhTraverser', 'DhCfrSolver', 'DhState', 'DhTraverser', 'EvExpl', 'PtttCfrSolver', 'PtttState', 'PtttTraverser']
class AbruptDhCfrSolver:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, traverser: AbruptDhTraverser, cfr_conf: CfrConf) -> None:
        ...
    def avg_bh(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Returns the average behavioral strategies
        """
    def step(self) -> None:
        """
        Performs a single CFR step
        """
class AbruptDhState:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 162
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def action_mask(self) -> typing.Annotated[list[bool], pybind11_stubgen.typing_ext.FixedSize(9)]:
        """
        Returns a mask of legal actions
        """
    def clone(self) -> AbruptDhState:
        ...
    def compute_openspiel_infostate(self) -> numpy.ndarray[bool]:
        """
        Returns the OpenSpiel infoset
        """
    def infoset_desc(self) -> str:
        """
        Returns a description of the current infoset
        """
    def is_terminal(self) -> bool:
        """
        Returns True if the game is over
        """
    def next(self, cell: int) -> None:
        """
        Play the given cell
        """
    def player(self) -> int | None:
        """
        Returns the current player, or None if the game is over
        """
    def winner(self) -> int | None:
        """
        Returns the winner (0, 1, or None if it's a tie)
        """
class AbruptDhTraverser:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 162
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def compute_openspiel_infostate(self, arg0: int, arg1: int) -> numpy.ndarray[bool]:
        ...
    def compute_openspiel_infostates(self, player: int) -> numpy.ndarray[bool]:
        ...
    def construct_uniform_strategies(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Constructs uniform (behavioral) strategies for both players
        """
    def ev_and_exploitability(self, strat0: numpy.ndarray[numpy.float64], strat1: numpy.ndarray[numpy.float64]) -> EvExpl:
        ...
    def infoset_desc(self, player: int, row: int) -> str:
        ...
    def is_valid_strategy(self, arg0: int, arg1: numpy.ndarray[numpy.float64]) -> bool:
        """
        Checks if the given strategy is valid, (has the right shape, is nonnegative and sums to 1)
        """
    def new_averager(self, player: int, avg_strategy: AveragingStrategy) -> Averager:
        """
        Returns a new Averager object
        """
    def parent_index_and_action(self, player: int, row: int) -> tuple[int, int]:
        ...
    def row_for_infoset(self, player: int, infoset_desc: str) -> int:
        ...
    @property
    def NUM_INFOS_PL0(self) -> int:
        """
        Number of infosets for player 0
        """
    @property
    def NUM_INFOS_PL1(self) -> int:
        """
        Number of infosets for player 1
        """
class AbruptPtttCfrSolver:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, traverser: AbruptPtttTraverser, cfr_conf: CfrConf) -> None:
        ...
    def avg_bh(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Returns the average behavioral strategies
        """
    def step(self) -> None:
        """
        Performs a single CFR step
        """
class AbruptPtttState:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 108
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def action_mask(self) -> typing.Annotated[list[bool], pybind11_stubgen.typing_ext.FixedSize(9)]:
        """
        Returns a mask of legal actions
        """
    def clone(self) -> AbruptPtttState:
        ...
    def compute_openspiel_infostate(self) -> numpy.ndarray[bool]:
        """
        Returns the OpenSpiel infoset
        """
    def infoset_desc(self) -> str:
        """
        Returns a description of the current infoset
        """
    def is_terminal(self) -> bool:
        """
        Returns True if the game is over
        """
    def next(self, cell: int) -> None:
        """
        Play the given cell
        """
    def player(self) -> int | None:
        """
        Returns the current player, or None if the game is over
        """
    def winner(self) -> int | None:
        """
        Returns the winner (0, 1, or None if it's a tie)
        """
class AbruptPtttTraverser:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 108
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def compute_openspiel_infostate(self, arg0: int, arg1: int) -> numpy.ndarray[bool]:
        ...
    def compute_openspiel_infostates(self, player: int) -> numpy.ndarray[bool]:
        ...
    def construct_uniform_strategies(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Constructs uniform (behavioral) strategies for both players
        """
    def ev_and_exploitability(self, strat0: numpy.ndarray[numpy.float64], strat1: numpy.ndarray[numpy.float64]) -> EvExpl:
        ...
    def infoset_desc(self, player: int, row: int) -> str:
        ...
    def is_valid_strategy(self, arg0: int, arg1: numpy.ndarray[numpy.float64]) -> bool:
        """
        Checks if the given strategy is valid, (has the right shape, is nonnegative and sums to 1)
        """
    def new_averager(self, player: int, avg_strategy: AveragingStrategy) -> Averager:
        """
        Returns a new Averager object
        """
    def parent_index_and_action(self, player: int, row: int) -> tuple[int, int]:
        ...
    def row_for_infoset(self, player: int, infoset_desc: str) -> int:
        ...
    @property
    def NUM_INFOS_PL0(self) -> int:
        """
        Number of infosets for player 0
        """
    @property
    def NUM_INFOS_PL1(self) -> int:
        """
        Number of infosets for player 1
        """
class Averager:
    """
    Utility class for averaging strategies
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def clear(self) -> None:
        """
        Clears the running average
        """
    def push(self, strategy: numpy.ndarray[numpy.float64], weight: float | None = None) -> None:
        """
        Pushes a strategy, use weight argument for custom averaging
        """
    def running_avg(self) -> numpy.ndarray[numpy.float64]:
        ...
class AveragingStrategy:
    """
    Members:
    
      UNIFORM : All strategies have the same weight.
    
      LINEAR : Strategy t has weight proportional to t
    
      QUADRATIC : Strategy t has weight proportional to $t^2$
    
      EXPERIMENTAL : Experimental averating
    
      LAST : The last strategy has weight 1
    
      CUSTOM : The user will provide the weights via the `weigh` argument
    """
    CUSTOM: typing.ClassVar[AveragingStrategy]  # value = <AveragingStrategy.CUSTOM: 5>
    EXPERIMENTAL: typing.ClassVar[AveragingStrategy]  # value = <AveragingStrategy.EXPERIMENTAL: 3>
    LAST: typing.ClassVar[AveragingStrategy]  # value = <AveragingStrategy.LAST: 4>
    LINEAR: typing.ClassVar[AveragingStrategy]  # value = <AveragingStrategy.LINEAR: 1>
    QUADRATIC: typing.ClassVar[AveragingStrategy]  # value = <AveragingStrategy.QUADRATIC: 2>
    UNIFORM: typing.ClassVar[AveragingStrategy]  # value = <AveragingStrategy.UNIFORM: 0>
    __members__: typing.ClassVar[dict[str, AveragingStrategy]]  # value = {'UNIFORM': <AveragingStrategy.UNIFORM: 0>, 'LINEAR': <AveragingStrategy.LINEAR: 1>, 'QUADRATIC': <AveragingStrategy.QUADRATIC: 2>, 'EXPERIMENTAL': <AveragingStrategy.EXPERIMENTAL: 3>, 'LAST': <AveragingStrategy.LAST: 4>, 'CUSTOM': <AveragingStrategy.CUSTOM: 5>}
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CfrConf:
    """
    
            Configuration for CFR
            - avg: Averaging strategy,
            - alternation: Use alternation,
            - dcfr: Use discounted CFR,
            - predictive: Use predictive CFR
            - rmplus: Use RM+, can be mixed with DCFR, 
    
            The following configurations are available:
            - PCFRP: Predictive CFR+,
            - DCFR: Discounted CFR,
    """
    DCFR: typing.ClassVar[CfrConf]  # value = CfrConf(avg=quadratic, alternation=true, dcfr=true, rmplus=false, predictive=false)
    PCFRP: typing.ClassVar[CfrConf]  # value = CfrConf(avg=quadratic, alternation=true, dcfr=false, rmplus=true, predictive=true)
    alternation: bool
    avg: AveragingStrategy
    dcfr: bool
    predictive: bool
    rmplus: bool
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, *, avg: AveragingStrategy = dh3.AveragingStrategy.QUADRATIC, alternation: bool = True, dcfr: bool = True, rmplus: bool = False, predictive: bool = False) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
class CornerDhCfrSolver:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, traverser: CornerDhTraverser, cfr_conf: CfrConf) -> None:
        ...
    def avg_bh(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Returns the average behavioral strategies
        """
    def step(self) -> None:
        """
        Performs a single CFR step
        """
class CornerDhState:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 162
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def action_mask(self) -> typing.Annotated[list[bool], pybind11_stubgen.typing_ext.FixedSize(9)]:
        """
        Returns a mask of legal actions
        """
    def clone(self) -> CornerDhState:
        ...
    def compute_openspiel_infostate(self) -> numpy.ndarray[bool]:
        """
        Returns the OpenSpiel infoset
        """
    def infoset_desc(self) -> str:
        """
        Returns a description of the current infoset
        """
    def is_terminal(self) -> bool:
        """
        Returns True if the game is over
        """
    def next(self, cell: int) -> None:
        """
        Play the given cell
        """
    def player(self) -> int | None:
        """
        Returns the current player, or None if the game is over
        """
    def winner(self) -> int | None:
        """
        Returns the winner (0, 1, or None if it's a tie)
        """
class CornerDhTraverser:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 162
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def compute_openspiel_infostate(self, arg0: int, arg1: int) -> numpy.ndarray[bool]:
        ...
    def compute_openspiel_infostates(self, player: int) -> numpy.ndarray[bool]:
        ...
    def construct_uniform_strategies(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Constructs uniform (behavioral) strategies for both players
        """
    def ev_and_exploitability(self, strat0: numpy.ndarray[numpy.float64], strat1: numpy.ndarray[numpy.float64]) -> EvExpl:
        ...
    def infoset_desc(self, player: int, row: int) -> str:
        ...
    def is_valid_strategy(self, arg0: int, arg1: numpy.ndarray[numpy.float64]) -> bool:
        """
        Checks if the given strategy is valid, (has the right shape, is nonnegative and sums to 1)
        """
    def new_averager(self, player: int, avg_strategy: AveragingStrategy) -> Averager:
        """
        Returns a new Averager object
        """
    def parent_index_and_action(self, player: int, row: int) -> tuple[int, int]:
        ...
    def row_for_infoset(self, player: int, infoset_desc: str) -> int:
        ...
    @property
    def NUM_INFOS_PL0(self) -> int:
        """
        Number of infosets for player 0
        """
    @property
    def NUM_INFOS_PL1(self) -> int:
        """
        Number of infosets for player 1
        """
class DhCfrSolver:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, traverser: DhTraverser, cfr_conf: CfrConf) -> None:
        ...
    def avg_bh(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Returns the average behavioral strategies
        """
    def step(self) -> None:
        """
        Performs a single CFR step
        """
class DhState:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 162
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def action_mask(self) -> typing.Annotated[list[bool], pybind11_stubgen.typing_ext.FixedSize(9)]:
        """
        Returns a mask of legal actions
        """
    def clone(self) -> DhState:
        ...
    def compute_openspiel_infostate(self) -> numpy.ndarray[bool]:
        """
        Returns the OpenSpiel infoset
        """
    def infoset_desc(self) -> str:
        """
        Returns a description of the current infoset
        """
    def is_terminal(self) -> bool:
        """
        Returns True if the game is over
        """
    def next(self, cell: int) -> None:
        """
        Play the given cell
        """
    def player(self) -> int | None:
        """
        Returns the current player, or None if the game is over
        """
    def winner(self) -> int | None:
        """
        Returns the winner (0, 1, or None if it's a tie)
        """
class DhTraverser:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 162
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def compute_openspiel_infostate(self, arg0: int, arg1: int) -> numpy.ndarray[bool]:
        ...
    def compute_openspiel_infostates(self, player: int) -> numpy.ndarray[bool]:
        ...
    def construct_uniform_strategies(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Constructs uniform (behavioral) strategies for both players
        """
    def ev_and_exploitability(self, strat0: numpy.ndarray[numpy.float64], strat1: numpy.ndarray[numpy.float64]) -> EvExpl:
        ...
    def infoset_desc(self, player: int, row: int) -> str:
        ...
    def is_valid_strategy(self, arg0: int, arg1: numpy.ndarray[numpy.float64]) -> bool:
        """
        Checks if the given strategy is valid, (has the right shape, is nonnegative and sums to 1)
        """
    def new_averager(self, player: int, avg_strategy: AveragingStrategy) -> Averager:
        """
        Returns a new Averager object
        """
    def parent_index_and_action(self, player: int, row: int) -> tuple[int, int]:
        ...
    def row_for_infoset(self, player: int, infoset_desc: str) -> int:
        ...
    @property
    def NUM_INFOS_PL0(self) -> int:
        """
        Number of infosets for player 0
        """
    @property
    def NUM_INFOS_PL1(self) -> int:
        """
        Number of infosets for player 1
        """
class EvExpl:
    """
    Utility class for holding EV and exploitability results
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __repr__(self) -> str:
        ...
    @property
    def best_response(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Best response policies
        """
    @property
    def ev0(self) -> float:
        """
        EV0
        """
    @property
    def expl(self) -> tuple[float, float]:
        """
        Exploitabilities
        """
    @property
    def gradient(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Gradients
        """
class PtttCfrSolver:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, traverser: PtttTraverser, cfr_conf: CfrConf) -> None:
        ...
    def avg_bh(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Returns the average behavioral strategies
        """
    def step(self) -> None:
        """
        Performs a single CFR step
        """
class PtttState:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 108
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def action_mask(self) -> typing.Annotated[list[bool], pybind11_stubgen.typing_ext.FixedSize(9)]:
        """
        Returns a mask of legal actions
        """
    def clone(self) -> PtttState:
        ...
    def compute_openspiel_infostate(self) -> numpy.ndarray[bool]:
        """
        Returns the OpenSpiel infoset
        """
    def infoset_desc(self) -> str:
        """
        Returns a description of the current infoset
        """
    def is_terminal(self) -> bool:
        """
        Returns True if the game is over
        """
    def next(self, cell: int) -> None:
        """
        Play the given cell
        """
    def player(self) -> int | None:
        """
        Returns the current player, or None if the game is over
        """
    def winner(self) -> int | None:
        """
        Returns the winner (0, 1, or None if it's a tie)
        """
class PtttTraverser:
    OPENSPIEL_INFOSTATE_SIZE: typing.ClassVar[int] = 108
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def compute_openspiel_infostate(self, arg0: int, arg1: int) -> numpy.ndarray[bool]:
        ...
    def compute_openspiel_infostates(self, player: int) -> numpy.ndarray[bool]:
        ...
    def construct_uniform_strategies(self) -> tuple[numpy.ndarray[numpy.float64], numpy.ndarray[numpy.float64]]:
        """
        Constructs uniform (behavioral) strategies for both players
        """
    def ev_and_exploitability(self, strat0: numpy.ndarray[numpy.float64], strat1: numpy.ndarray[numpy.float64]) -> EvExpl:
        ...
    def infoset_desc(self, player: int, row: int) -> str:
        ...
    def is_valid_strategy(self, arg0: int, arg1: numpy.ndarray[numpy.float64]) -> bool:
        """
        Checks if the given strategy is valid, (has the right shape, is nonnegative and sums to 1)
        """
    def new_averager(self, player: int, avg_strategy: AveragingStrategy) -> Averager:
        """
        Returns a new Averager object
        """
    def parent_index_and_action(self, player: int, row: int) -> tuple[int, int]:
        ...
    def row_for_infoset(self, player: int, infoset_desc: str) -> int:
        ...
    @property
    def NUM_INFOS_PL0(self) -> int:
        """
        Number of infosets for player 0
        """
    @property
    def NUM_INFOS_PL1(self) -> int:
        """
        Number of infosets for player 1
        """
@typing.overload
def CfrSolver(traverser: DhTraverser, cfr_conf: CfrConf) -> DhCfrSolver:
    """
    Constructs a new CfrSolver
    """
@typing.overload
def CfrSolver(traverser: AbruptDhTraverser, cfr_conf: CfrConf) -> AbruptDhCfrSolver:
    """
    Constructs a new CfrSolver
    """
@typing.overload
def CfrSolver(traverser: CornerDhTraverser, cfr_conf: CfrConf) -> CornerDhCfrSolver:
    """
    Constructs a new CfrSolver
    """
@typing.overload
def CfrSolver(traverser: PtttTraverser, cfr_conf: CfrConf) -> PtttCfrSolver:
    """
    Constructs a new CfrSolver
    """
@typing.overload
def CfrSolver(traverser: AbruptPtttTraverser, cfr_conf: CfrConf) -> AbruptPtttCfrSolver:
    """
    Constructs a new CfrSolver
    """
