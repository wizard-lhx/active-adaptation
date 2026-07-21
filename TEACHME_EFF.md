# Effective Impedance Diagnostics

This branch adds play-time effective impedance calculation and optional damping
intervention to `ppo_symaug`. It does not add any training loss, optimizer term,
or training-time logging.

## Files

| File | Purpose |
| --- | --- |
| `active_adaptation/learning/diagnostics/eff_config.py` | Impedance and clamp configuration |
| `active_adaptation/learning/diagnostics/eff_impedance.py` | Torch Jacobian, matrices, spectra, and condition numbers |
| `active_adaptation/learning/diagnostics/eff_clamp.py` | Physics-substep damping correction |
| `active_adaptation/learning/diagnostics/eff_record.py` | Periodic NPZ recording |
| `active_adaptation/learning/ppo/ppo_symaug_eff.py` | Environment-derived slices, gains, scaling, and Hydra registration |
| `scripts/play_eff.py` | Effective impedance playback and recording |
| `scripts/play_interactive.py` | `play_eff` with sling and mouse interaction |
| `scripts/visualize_eff_impedance.py` | Interactive video and matrix replay |
| `scripts/analyze_clamp.py` | Clamp energy, saturation, and dose analysis |

The normal `scripts/play.py` is the main-branch playback script and has no
effective impedance dependency.

## Calculation

Let the deterministic policy mean be `mu(o)`. The policy input columns occupied
by controlled joint position and velocity are inferred from the environment:

```text
Jq  = d mu / d q
Jqd = d mu / d qd
```

For the current position-target action space:

```text
Keff = diag(Kp) (I - diag(alpha) Jq)
Deff = diag(Kd) - diag(Kp) diag(alpha) Jqd
```

`alpha` is the action-to-position scale from the action manager. `Kp` and `Kd`
are the actual gains of the recorded environment. Keff, Deff, their symmetric
eigenvalues, and their symmetric condition numbers are calculated with Torch on
the policy device.

This is a static local approximation. The formula does not include the action
manager's randomized delay or action low-pass filter.

## Clamp

The controller forms:

```text
S       = 0.5 (Deff + Deff.T)
S       = V diag(lambda) V.T
DeltaD  = V diag(relu(d_min - lambda)) V.T
tau     = -DeltaD qd
```

`tau` is evaluated from the current simulator joint velocity and applied at
every physics substep. The existing per-joint `tau_limit` clipping is retained.

Modes:

| Configuration | Applied matrix |
| --- | --- |
| Clamp disabled | No additional effort |
| `baseline_mode=true` | Zero matrix after the complete calculation |
| Active clamp | Projected `DeltaD` |
| `override_diag_c>0` | `override_diag_c * I` |

`p_neg_substep` is computed from the negative spectrum of `S`. In diagonal
override mode it remains a theoretical diagnostic and is not expected to equal
the applied correction power.

## Recording Flow

Each row follows one control interval:

```text
obs_t and video frame_t
  -> policy action_t
  -> compute impedance / update clamp
  -> physics substeps: qd_t and tau_t
  -> record row t
```

The first row is saved immediately. The complete accumulated NPZ is then
atomically replaced every ten rows. At 50 Hz, an abrupt process exit loses at
most about 0.2 seconds. Saving does not depend on a `finally` block and there is
no exit-time supplemental save.

Torch tensors are converted to CPU NumPy arrays only inside `Recorder.record()`.

## Commands

Record matrices without intervention:

```bash
uv run --project venv/isaac51 python scripts/play_eff.py \
  task=B2/B2Z1Loco \
  backend=isaac \
  checkpoint_path=<checkpoint> \
  record_video=true
```

Active clamp:

```bash
uv run --project venv/isaac51 python scripts/play_eff.py \
  task=B2/B2Z1Loco \
  backend=isaac \
  checkpoint_path=<checkpoint> \
  record_video=true \
  algo.eff_impedance.clamp.enabled=true \
  algo.eff_impedance.clamp.d_min=0.0 \
  algo.eff_impedance.clamp.tau_limit=50.0
```

Baseline uses the same command with:

```text
algo.eff_impedance.clamp.baseline_mode=true
```

Diagonal override example:

```text
algo.eff_impedance.clamp.enabled=true
algo.eff_impedance.clamp.override_diag_c=5.0
algo.eff_impedance.clamp.tau_limit=150.0
```

Interactive sling experiment:

```bash
uv run --project venv/isaac51 python scripts/play_interactive.py \
  task=B2/B2Z1Loco \
  checkpoint_path=<checkpoint> \
  record_video=true \
  algo.eff_impedance.clamp.enabled=true
```

`play_interactive.py` selects Isaac, CPU physics, GUI, one environment,
`InteractiveTwist`, and teleoperation. Press `H` to enable the sling and hold
`UP` or `DOWN` to change its target height.

## Output

Hydra places the files under one playback run:

```text
outputs_play/<date>/<run>/videos/<task>-<time>.mp4
outputs_play/<date>/<run>/eff_impedance/eff_impedance_timeseries.npz
```

Every NPZ contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `steps` | `(T,)` | Control-step index, starting at zero |
| `Keff`, `Deff` | `(T,n,n)` | Effective matrices |
| `Keff_eigvals`, `Deff_eigvals` | `(T,n)` | Sorted eigenvalues of symmetric parts |
| `Keff_cond`, `Deff_cond` | `(T,)` | Symmetric matrix condition numbers |
| `kp`, `kd` | `(T,n)` | Actual environment gains |
| `joint_names` | `(n,)` | Action and matrix joint order |
| `physics_dt`, `control_dt`, `decimation` | scalar | Timing metadata |

Clamp recordings additionally contain:

| Key | Shape | Meaning |
| --- | --- | --- |
| `tau_corr` | `(T,decimation,n)` | Applied correction torque |
| `joint_vel_substep` | `(T,decimation,n)` | Velocity used to compute torque |
| `p_neg_substep` | `(T,decimation)` | Power associated with negative S modes |
| `s_eigvals` | `(T,n)` | Original symmetric Deff spectrum |
| `delta_d_eigvals` | `(T,n)` | PSD correction spectrum |
| `clamp_applied` | `(T,)` | False for baseline, true for active/override |
| `d_min`, `tau_limit`, `override_diag_c` | scalar | Clamp parameters |

The format intentionally has no compatibility path for earlier NPZ files.

## Visualization

```bash
uv run --project venv/isaac51 python scripts/visualize_eff_impedance.py \
  --video <run>/videos/<video>.mp4 \
  --npz <run>/eff_impedance/eff_impedance_timeseries.npz
```

Controls:

```text
SPACE        play / pause
LEFT/RIGHT   step backward / forward; hold for continuous movement
V            show / hide matrix and eigenvalue numbers
```

The viewer reads the stored spectra. It only selects the minimum eigenvalue and
counts negative entries; it does not run an eigensolver.

## Clamp Analysis

Single rollout:

```bash
uv run --project venv/isaac51 python scripts/analyze_clamp.py \
  <run>/eff_impedance/eff_impedance_timeseries.npz \
  --trim-start-s 2.0
```

Dose response:

```bash
uv run --project venv/isaac51 python scripts/analyze_clamp.py \
  --dose <c0.npz> <c2.npz> <c5.npz> <c10.npz> \
  --trim-start-s 2.0 \
  --out-dir <output-dir>
```

The analyzer obtains joint count, substep count, names, timing, Kd, and torque
limit directly from the NPZ. `lam_min` is the minimum original `s_eigvals`, not
the nonnegative correction spectrum.
