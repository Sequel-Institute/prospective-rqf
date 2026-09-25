# Selective Pure RQF

For the selective copying task, model configuration, and training commands,
see [selective copying](selective_copying.md).

`model=pure_rqf_selective` inherits `pure_rqf_paper`. Selectivity is also exposed
as `PureRQFEncoder(..., selective=SelectiveRQFConfig(...))` or a dictionary.
`selective=None` preserves the static model's modules, parameters and checkpoint
keys. The encoder's `forward(x)` interface is unchanged.

Selective models require complex weights, one forward oscillator per channel,
ZOH, log-space parameters, fixed positive tau, `clamp_gamma=false`, and split
ReLU. The final activation can still be disabled for a raw complex readout.

## Controllers

| Option | Default | Meaning |
| --- | --- | --- |
| `gamma_control` | `true` | Enable input-dependent damping. |
| `gamma_link` | `sigmoid` | `sigmoid` or multiplicative `log_shift`. |
| `gamma_features` | `real_imag` | Signed planes, or `modulus` for gamma only. |
| `gamma_max_log_scale` | `log(1.25)` | Log-shift gamma bound. |
| `gamma_edge` | `interior_clamp` | Interior resting gamma, or historical `st_clamp`. |
| `gamma_edge_margin` | `0.001` | Normalized resting-gamma margin at both edges; in `(0, 0.5)`. |
| `write_control` | `false` | Enable input-dependent write gain. |
| `write_link` | `sigmoid` | Bounded positive `sigmoid`, or historical signed `affine`. |
| `write_max` | `2.0` | Sigmoid gain ceiling; finite and greater than one. |
| `frequency_warp` | `none` | `none`, `global`, or `grouped`. |
| `warp_bound` | `band` | Bound warped frequency to the band and Nyquist; `none` restores historical warping. |
| `num_groups` | `4` | Group count for grouped frequency warps. |
| `max_log_scale` | `log(1.25)` | Frequency log-scale bound. |
| `input_source` | `layer_output` | Raw branch, or `layer_input` for the actual drive input. |

For selector source `s`, let `v = [Re(s); Im(s)]`. Gamma's features `f` are `v`
or `abs(s)`; magnitude features have zero subgradient at the origin.

- Sigmoid gamma: `gamma = min + span * sigmoid(b + W_gamma @ f)`, where `b`
  is the logit of normalized resting `exp(log_gamma)`. The default projects
  resting gamma into `[min + margin*span, max - margin*span]` at initialization
  and after updates. The logit uses ordinary autograd, without a straight-through
  clamp. At margin `0.001`, the derivative amplification relative to resting
  gamma is bounded by `1/(4*margin*(1-margin))`, approximately 250.25.
  `gamma_edge=st_clamp` preserves the historical machine-epsilon interior
  logit and straight-through gradient for checkpoint reproduction.
- Log-shift gamma: `gamma = exp(log_gamma + c * tanh(W_gamma @ f))`. The
  resting ceiling is lowered to `max_gamma * exp(-c)` at construction and after
  updates. This can change initial damping even with a zero head. Effective
  damping remains positive but can lie below the resting lower bound.
- Bounded write (default): `gate = write_max * sigmoid(W_write @ v - log(write_max - 1))`.
  The gain lies in `(0, write_max)` and equals one at a zero head. Its logit
  derivative is `gate * (1 - gate / write_max)`. Floating-point saturation can
  reach either endpoint. The fixed offset adds no trainable parameter.
- Affine write (`write_link=affine`): `gate = 1 + W_write @ v`, signed and
  unbounded. `write_max` is inert for this link; both options are inert when
  write control is disabled.
- Frequency: `delta = max_log_scale * tanh(W_delta @ v)`, then
  `theta = exp(log_theta) * exp(delta[group])`. With `warp_bound=band`, clamp
  this effective theta to `[2*pi*min_freq*tau, min(2*pi*max_freq*tau, pi*tau/dt)]`.
  Resting theta uses the same bounds; an empty or nonpositive allowed band is
  rejected. The conservative ceiling guarantees `(1-gamma)*theta*dt/tau <= pi`.
  Gradients to both the resting frequency and warp head are zero outside the
  cap, with the same boundary convention in PyTorch and CUDA. `warp_bound=none`
  retains the original unconstrained effective frequency.

All heads are bias-free and start at zero without consuming random numbers.
Enabled heads are trainable independently of their resting parameters. A frozen
resting gamma or frequency does not freeze its controller. Explicitly freezing
the selector or complete model still disables its gradients.
Gamma and frequency heads use the dynamics learning rate and zero weight decay;
the write head follows `W`. Group membership is `initial_rank * G // F`, using
stable initial `log_theta` ranks, and is saved as a fixed buffer. Gamma, write,
and frequency controls can be enabled independently; an all-disabled block is
rejected in favor of `selective=None`.

## Source and residual semantics

The first block reads the existing input projection. Later blocks normally use
`x = split_relu(y) + skip` for their drive and `s = y + skip` for raw selectors.
With `input_source=layer_input`, selectors use `x` too. The same shape-based skip
eligibility and final-activation setting as the static encoder apply. This
preserves the encoder's activation-then-add ordering. Selectors are current-time,
previous-layer functions, with no additional recurrent state.

## Recurrence and learning

Writing `eta = dt/tau` and `a = eta*(-gamma + i*(1-gamma)*theta)`:

```text
A = exp(a)
B = gamma * eta * expm1(a)/a                 # stable series near zero
p = B + ratio * gamma * A
n = -ratio * gamma * A
y[t] = A[t]*y[t-1] + gate[t]*(p[t]*W*x[t] + n[t]*W*x[t-1])
```

The current interval's gate multiplies both taps. Its lagged drive is ungated.
`ratio=1` for `prospective_mode=tau`, `1/gamma[t]` for `gamma`, and
`prospective_tau/tau` when an explicit horizon is supplied. Non-prospective
input has `p=B, n=0`.

Full BPTT differentiates the complete trajectory. TBPTT(1) detaches `y[t-1]`
and the source `x[t-1]`, while keeping `W*x[t-1].detach()` differentiable into
the current layer's weight. All current selectors remain differentiable,
including their dependence on raw previous-layer outputs.
Selective prospective TBPTT(1) supports only the `strict` delayed-input graph.
Requesting `tbptt_delayed_input_mode=literal` with a nonzero delayed tap raises
an error at configuration/model validation or dispatch. The setting is inert
for full BPTT, non-prospective input, and an explicitly zero horizon.

The CUDA implementation keeps complex `(F,B,T)` tensors and computes controller
links and coefficients in FP32 registers. Backward replays each chunk from its
entry state, gathers neighbors without dividing by transitions, and routes the
lag cotangent only to `W` in TBPTT(1). Raw-output gradients bypass the ReLU mask.
Grouped warp and resting-parameter reductions are deterministic.

Selector settings are saved in module extra state as well as the training
config; loading a checkpoint into different selector settings raises an error.
Checkpoints without `write_link` metadata retain affine semantics; explicitly
select `write_link=affine` when loading these into a model with write control.
New training checkpoints also save Hydra overrides so `evaluate_checkpoint.py`
can reconstruct the selective architecture and controller options.
Evaluation restores the saved write link and ceiling even if preset defaults
have since changed; saved configurations without the link use affine gain.
Missing `gamma_edge` and `warp_bound` settings restore `st_clamp` and `none`,
respectively, before the model is built. This also preserves checkpoints with
bounded write gain that predate the new gamma and frequency policies. Loading
these weights into a model with different active boundary policies is rejected.
Legacy static checkpoints continue to load strictly. Cross-repository checkpoint
conversion and other prospective-horizon families are not included.

## Validation commands

Using this repo's virtual environment:

```bash
# CPU suite, including float64 gradchecks. Unavailable GPU tests report skips.
.venv/bin/python -m unittest discover -s tests -v

# Run on the Mac host with access to Metal; no CPU fallback is permitted.
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=tests \
  .venv/bin/python -m unittest test_selective_rqf.SelectiveMPSTest -v

# Native CUDA/Triton: fused outputs and every gradient, both learning modes.
PYTHONPATH=tests .venv/bin/python -m unittest test_selective_cuda -v

# Additional arithmetic checks where the Triton CPU interpreter is supported.
TRITON_INTERPRET=1 PYTHONPATH=tests \
  .venv/bin/python -m unittest test_selective_cuda -v

# CUDA timing and peak memory. Repeat with --grad-mode tbptt1 and --non-prospective.
.venv/bin/python benchmarks/bench_selective_rqf.py --time 4096 --batch 16
```

MPS runs the PyTorch reference in complex64/float32; float64 tests run on CPU.
Training computes complex gradient norms through real/imaginary views on MPS,
where PyTorch 2.7's native complex norm is unsupported.
MPS parity and interpreter checks do not validate CUDA compilation, execution,
or performance. The native CUDA suite and benchmark must run on NVIDIA hardware
before reporting those results. This repo's `device=auto` retains its existing
CUDA-or-CPU selection; request `device=mps` explicitly for Mac training.
