"""Column A control configurations (LV, DV, L/D-V/B).

Each configuration closes the two level loops (condenser holdup and
reboiler holdup) and exposes the remaining two inputs as the
manipulated-variable pair for the supervisory layer. The configurations
themselves are stateless callables of the form ``(t, X, disturbances)
-> U`` ready to be plugged into ``integrate_open_loop`` via its
``inputs_fn`` argument.

Phase 1 ships ``lv``, ``dv``, and ``ldvb`` (the canonical Skogestad
trio).
"""

from industrial_ai.twin.column_a.configurations.dv import (
    DVConfiguration,
    assemble_inputs_dv,
)
from industrial_ai.twin.column_a.configurations.ldvb import (
    LDVBConfiguration,
    assemble_inputs_ldvb,
    nominal_ratios,
)
from industrial_ai.twin.column_a.configurations.lv import (
    LVConfiguration,
    assemble_inputs_lv,
)

__all__ = [
    "DVConfiguration",
    "LDVBConfiguration",
    "LVConfiguration",
    "assemble_inputs_dv",
    "assemble_inputs_ldvb",
    "assemble_inputs_lv",
    "nominal_ratios",
]
