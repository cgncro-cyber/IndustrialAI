"""Simplified steady-state decoupler for the LV composition pair.

The LV configuration of Skogestad's Column A has a notoriously large
relative-gain-array element ``lambda_11 ~ 36`` at the canonical
operating point (``G^LV(0)`` from
:func:`industrial_ai.twin.column_a.linearize.linearize_lv`). This
means the top and bottom composition loops compete strongly: a SISO
PID pair on this plant is structurally handicapped and an
agent-vs-SISO-PID comparison can be charged with comparing
"sighted MIMO" (agent) against "blind MIMO" (PID).

The Phase-2 baseline therefore includes an option to install a
**simplified static decoupler** between the two PIDs and the LV plant:

    D = [[1, -G12/G11],
         [-G21/G22, 1]]

so that ``G(0) * D`` is diagonal at the operating point. The
decoupler operates on PID *deviations* from the operating-point bias
to keep the closed-loop coordinate system the same. The exact
formulation appears as the "simplified decoupling" recipe in
Garrido et al. and is the variant Skogestad & Postlethwaite (1996),
§10.8 recommend over the inverse-based form ``D = G^-1 * diag(G)``
for high-RGA plants — the inverse-based form requires unphysically
large gains for ``lambda_11 ~ 36``.

After decoupling, the effective per-loop plant gain shrinks by the
RGA factor: ``g_ii_eff = g_ii / lambda_ii``. Any subsequent tuning
must be rerun against this new effective plant.

References.

- Garrido, J., Vázquez, F. and Morilla, F. (2011). *An extended
  approach of inverted decoupling.* Journal of Process Control 21(1),
  55-68.
- Skogestad, S. and Postlethwaite, I. (1996). *Multivariable
  Feedback Control: Analysis and Design.* Wiley, §10.8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from industrial_ai.twin.column_a.linearize import (
    LinearizedLVModel,
    steady_state_gain,
)

__all__ = [
    "DecouplerSpec",
    "identity_decoupler",
    "rga",
    "simplified_decoupler",
]


@dataclass(frozen=True, slots=True)
class DecouplerSpec:
    """Static 2x2 decoupler applied to LV-loop MV deviations.

    Attributes
    ----------
    matrix : numpy.ndarray of shape (2, 2)
        Decoupler matrix ``D``. The simulator applies it as
        ``[LT, VB]_actual = bias + D @ ([LT, VB]_pid - bias)`` so the
        no-op decoupler is ``np.eye(2)``.
    rga_11 : float
        Diagonal RGA element of the underlying ``G(0)``. Recorded so
        downstream code can surface the structural rationale.
    g_effective_diag : numpy.ndarray of shape (2,)
        Diagonal elements of ``G(0) @ D`` — the *effective* per-loop
        plant gain visible to the SISO PIDs after decoupling. For the
        Skogestad LV at nominal: roughly ``g_ii / lambda_ii``, an
        order of magnitude smaller than ``g_ii``.
    """

    matrix: npt.NDArray[np.float64]
    rga_11: float
    g_effective_diag: npt.NDArray[np.float64]


def identity_decoupler() -> DecouplerSpec:
    """No-op decoupler. Use to keep the simulator signature uniform across variants."""
    return DecouplerSpec(
        matrix=np.eye(2, dtype=np.float64),
        rga_11=1.0,
        g_effective_diag=np.array([np.nan, np.nan], dtype=np.float64),
    )


def rga(G: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the Relative Gain Array ``G * (G^-1).T`` for a 2x2 ``G``."""
    if G.shape != (2, 2):
        raise ValueError(f"RGA helper here handles 2x2 only; got {G.shape}")
    return np.asarray(G * np.linalg.inv(G).T, dtype=np.float64)


def simplified_decoupler(
    model: LinearizedLVModel,
) -> DecouplerSpec:
    """Build the simplified static decoupler from an LV linearization.

    Parameters
    ----------
    model : LinearizedLVModel
        Linearized plant whose ``G^LV(0)`` defines the decoupler.

    Returns
    -------
    DecouplerSpec
    """
    G0 = steady_state_gain(model)[:, :2]  # (2, 2) y_D/x_B vs L/V block
    g11, g12 = float(G0[0, 0]), float(G0[0, 1])
    g21, g22 = float(G0[1, 0]), float(G0[1, 1])
    if g11 == 0.0 or g22 == 0.0:
        raise ValueError("simplified decoupler requires non-zero diagonal elements in G(0)")
    D = np.array(
        [
            [1.0, -g12 / g11],
            [-g21 / g22, 1.0],
        ],
        dtype=np.float64,
    )
    G_eff = G0 @ D
    lambda_11 = float(rga(G0)[0, 0])
    return DecouplerSpec(
        matrix=D,
        rga_11=lambda_11,
        g_effective_diag=np.array([G_eff[0, 0], G_eff[1, 1]], dtype=np.float64),
    )
