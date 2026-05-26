"""Setpoint ingress with rate-limiter / ramping.

Per ADR 006, every supervisory configuration (C0 / C1 / C2 / C3) talks
to the regulatory layer via a single setpoint interface that applies a
configurable rate limit. This pre-empts a class of failure modes where
a supervisor — especially the LLM-based one — proposes an unrealistic
step change that would push the column off-spec or cause integrator
divergence.

The interface is stateful (it remembers the previous setpoint to
enforce the rate limit) but free of any control logic. It is the
narrow boundary between the supervisor and the plant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RateLimiter", "SetpointInterface"]


@dataclass(slots=True)
class RateLimiter:
    """First-order rate limiter on a single setpoint channel.

    The output ramps toward the requested setpoint at a configurable
    maximum slew rate. Any abrupt requested-setpoint jump is therefore
    smoothed to a linear ramp of at most ``max_rate`` per minute.

    Attributes
    ----------
    max_rate : float
        Maximum absolute change of the output per minute. Must be
        strictly positive.
    current : float
        Currently held output (carried between :meth:`update` calls).
    """

    max_rate: float
    current: float = 0.0

    def update(self, *, requested: float, dt: float) -> float:
        """Move the held output toward ``requested`` over ``dt`` minutes.

        Parameters
        ----------
        requested : float
            New requested setpoint.
        dt : float
            Time elapsed since the previous update (min). Must be
            strictly positive.

        Returns
        -------
        float
            Rate-limited setpoint to apply this cycle.

        Raises
        ------
        ValueError
            If ``max_rate`` is non-positive or ``dt`` is non-positive.
        """
        if self.max_rate <= 0.0:
            raise ValueError(f"max_rate must be strictly positive, got {self.max_rate}")
        if dt <= 0.0:
            raise ValueError(f"dt must be strictly positive, got {dt}")

        delta = requested - self.current
        max_delta = self.max_rate * dt
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        self.current = self.current + delta
        return self.current

    def seed(self, value: float) -> None:
        """Reset the held output to ``value`` without ramping."""
        self.current = value


@dataclass(slots=True)
class SetpointInterface:
    """Uniform setpoint ingress for a Column A supervisory configuration.

    The interface owns one :class:`RateLimiter` per supervisory channel
    and applies them in lock-step. Designed for the LV configuration
    where the supervisor sets ``y_D`` and ``x_B`` setpoints (the
    regulatory layer then translates these to ``LT`` and ``VB``
    commands), but the same pattern extends to any supervisory variable
    set.

    Attributes
    ----------
    limiters : dict of str to RateLimiter
        Channel name → rate limiter. Channels are accessed by name
        (e.g., ``"yD_setpoint"``, ``"xB_setpoint"``).
    """

    limiters: dict[str, RateLimiter] = field(default_factory=dict)

    def register(self, *, name: str, max_rate: float, initial: float) -> None:
        """Add a new setpoint channel.

        Parameters
        ----------
        name : str
            Channel name (used as the key in :meth:`apply`).
        max_rate : float
            Maximum absolute slew rate (per minute).
        initial : float
            Initial held setpoint (the controller starts at this value).

        Raises
        ------
        ValueError
            If the channel already exists.
        """
        if name in self.limiters:
            raise ValueError(f"channel {name!r} is already registered")
        self.limiters[name] = RateLimiter(max_rate=max_rate, current=initial)

    def apply(
        self,
        *,
        requested: dict[str, float],
        dt: float,
    ) -> dict[str, float]:
        """Rate-limit a batch of requested setpoints.

        Parameters
        ----------
        requested : dict of str to float
            New requested setpoints, keyed by channel name. Every key
            must correspond to a registered channel.
        dt : float
            Time elapsed since the previous :meth:`apply` call (min).

        Returns
        -------
        dict of str to float
            Rate-limited setpoints to apply this cycle.

        Raises
        ------
        KeyError
            If ``requested`` references an unregistered channel.
        """
        unknown = set(requested) - set(self.limiters)
        if unknown:
            raise KeyError(f"unregistered setpoint channels: {sorted(unknown)}")
        return {
            name: self.limiters[name].update(requested=value, dt=dt)
            for name, value in requested.items()
        }
