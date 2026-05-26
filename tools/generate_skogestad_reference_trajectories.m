% Generate Skogestad Column A reference trajectories for Phase 1 mini-gate.
%
% Runs the published cola_lv.m MATLAB code (level loops closed,
% L and V as inputs) under three canonical disturbance scenarios and
% dumps the resulting (time, state) trajectories as CSV files. A
% Python helper converts the CSVs to one JSON file under
% data/reference/.
%
% This script is the Option-A engineering cross-check per the Phase 1
% mini-gate plan: validates that the Python port reproduces the
% trajectory of Skogestad's own MATLAB implementation, evaluated
% against the same nominal Xinit from cola_init.mat. The Python port's
% correctness w.r.t. the *published equations* is established
% separately via the Skogestad-1997 Eq. (31) and Section 4.4 scalar
% checks (Option C) in the pytest mini-gate.
%
% Required inputs:
%   /tmp/cola_matlab_ref/colamod.m
%   /tmp/cola_matlab_ref/cola_init.mat
% Output:
%   data/reference/_octave_trajectory_*.csv (one per scenario)

addpath('/tmp/cola_matlab_ref');
load('/tmp/cola_matlab_ref/cola_init.mat');  % loads Xinit, Uinit
NT = 41;

% Nominal operating point (from cola_init.mat).
LT_nom = 2.70629;
VB_nom = 3.20629;
F_nom  = 1.0;
zF_nom = 0.5;
qF_nom = 1.0;

% Output directory.
out_dir = fullfile(pwd, 'data', 'reference');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Inline cola_lv with parameterized inputs so we can step each one.
function xprime = cola_lv_param(t, X, LT, VB, F, zF, qF)
    NT = 41;
    KcB = 10; KcD = 10;
    MDs = 0.5; MBs = 0.5;
    Ds  = 0.5; Bs  = 0.5;
    MB = X(NT + 1);
    MD = X(2 * NT);
    D = Ds + (MD - MDs) * KcD;
    B = Bs + (MB - MBs) * KcB;
    U = [LT VB D B F zF qF];
    xprime = colamod(t, X, U);
endfunction

% Common integration settings — relative tolerance 1e-10 to keep the
% reference numerically tight against the published continuous-time
% solution, matching the integrator-limit tolerance discussed in the
% mini-gate scope.
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);

% Time grid for output: 0..500 min, 501 evenly spaced points.
tspan = linspace(0, 500, 501);

% --- Scenario 1: +1% step in reflux L at t = 0 ---
LT_step = LT_nom * 1.01;
[t1, x1] = ode15s(@(t, X) cola_lv_param(t, X, LT_step, VB_nom, F_nom, zF_nom, qF_nom), ...
                  tspan, Xinit, opts);
csvwrite(fullfile(out_dir, '_octave_trajectory_L_plus_1pct.csv'), [t1, x1]);

% --- Scenario 2: -10% step in feed composition zF at t = 0 ---
zF_step = zF_nom * 0.90;
[t2, x2] = ode15s(@(t, X) cola_lv_param(t, X, LT_nom, VB_nom, F_nom, zF_step, qF_nom), ...
                  tspan, Xinit, opts);
csvwrite(fullfile(out_dir, '_octave_trajectory_zF_minus_10pct.csv'), [t2, x2]);

% --- Scenario 3: +10% step in feed flow F at t = 0 ---
F_step = F_nom * 1.10;
[t3, x3] = ode15s(@(t, X) cola_lv_param(t, X, LT_nom, VB_nom, F_step, zF_nom, qF_nom), ...
                  tspan, Xinit, opts);
csvwrite(fullfile(out_dir, '_octave_trajectory_F_plus_10pct.csv'), [t3, x3]);

printf('Wrote 3 trajectory CSVs to %s\n', out_dir);
printf('  L  +1%%:  %d rows\n', size(x1, 1));
printf('  zF -10%%: %d rows\n', size(x2, 1));
printf('  F  +10%%: %d rows\n', size(x3, 1));
