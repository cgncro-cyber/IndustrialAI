"""Skogestad Column A — dynamic distillation twin.

Clean-room Python re-implementation of the canonical Column A
distillation benchmark defined in:

- Skogestad, S. & Morari, M. (1988). Understanding the Dynamic Behavior
  of Distillation Columns. *Ind. Eng. Chem. Res.* 27(10), 1848–1862.
- Skogestad, S. & Postlethwaite, I. (1996). *Multivariable Feedback
  Control: Analysis and Design*. Wiley.
- Skogestad, S. (1997). Dynamics and Control of Distillation Columns —
  A Tutorial Introduction. *Trans IChemE* 75(A), 539–562.

The Python port is derived from the published equations, not from
line-by-line translation of the MATLAB source at
https://skoge.folk.ntnu.no/book/1st_edition/matlab_m/cola/. The MATLAB
files serve as semantic reference only; their numerical outputs
(``cola_init.mat`` in particular) serve as validation anchors for the
pytest regression suite.
"""

from industrial_ai.twin.column_a.integrator import (
    IntegrationResult,
    integrate_open_loop,
)
from industrial_ai.twin.column_a.model import column_a_rhs
from industrial_ai.twin.column_a.parameters import (
    DEFAULT_PARAMETERS,
    ColumnAParameters,
)
from industrial_ai.twin.column_a.steady_state import (
    SteadyStateResult,
    compute_steady_state_by_integration,
    compute_steady_state_by_newton,
)

__all__ = [
    "DEFAULT_PARAMETERS",
    "ColumnAParameters",
    "IntegrationResult",
    "SteadyStateResult",
    "column_a_rhs",
    "compute_steady_state_by_integration",
    "compute_steady_state_by_newton",
    "integrate_open_loop",
]
