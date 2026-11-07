"""Budget-constrained agentic planning for multi-stage Text-to-SQL."""
import os as _os

# The world model is many small trees rather than one big one, and small fits
# are several times faster on one thread than on eight, so cap the BLAS and
# OpenMP pools unless the caller has already chosen.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

from .config import BUDGETS, DIMS, PRICES, PlannerConfig, vec
from .grammar import Grammar, State, Workflow, reference_library
from .planner import BudgetSQL, Meter, PlanningAgent
from .trace import SimulatedTrace, TraceCorpus
from .worldmodel import CrossFitStacker, StageWorldModel

__version__ = "0.1.0"

__all__ = ["BUDGETS", "DIMS", "PRICES", "PlannerConfig", "vec", "Grammar",
           "State", "Workflow", "reference_library", "BudgetSQL", "Meter",
           "PlanningAgent", "SimulatedTrace", "TraceCorpus", "CrossFitStacker",
           "StageWorldModel", "__version__"]
