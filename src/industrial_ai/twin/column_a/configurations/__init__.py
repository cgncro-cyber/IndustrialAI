"""Column A control configurations (LV, DV, L/D-V/B).

Each configuration closes the two level loops (condenser holdup and
reboiler holdup) and exposes the remaining two inputs as the
manipulated-variable pair for the supervisory layer. The configurations
themselves are stateless callables of the form ``(t, X, disturbances)
-> U`` ready to be plugged into ``integrate_open_loop`` via its
``inputs_fn`` argument.

Phase 1 ships ``lv`` first; ``dv`` and ``ldvb`` follow.
"""
