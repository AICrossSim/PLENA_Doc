# Analytic Model

The Analytic Model is PLENA's fast, closed-form cost estimator. It provides performance, area, and power numbers directly from a hardware configuration without running the transactional emulator or RTL synthesis, so the co-design search loop can sweep thousands of candidate configurations cheaply.

## Role in the Toolchain

In the co-design loop, the Analytic Model provides the **area** and a fast-path **latency** estimate:

| Objective | Description | Source |
|-----------|-------------|--------|
| Accuracy | Model quality under the chosen quantization config | Accuracy Evaluator |
| Latency (fast) | Closed-form performance estimate | Analytic Model |
| Latency (detailed) | Cycle-approximate, memory-accurate | Transactional Emulator |
| **Area / Power** | Hardware resource and power cost | Analytic Model |

## Performance Analytic Model

A closed-form performance model that estimates end-to-end latency for a given configuration and workload.

- **Much faster than the transactional emulator** — returns a cost in milliseconds instead of running an event-driven simulation, which lets the search loop explore broadly before committing to detailed evaluation.
- **Less accurate than the transactional emulator** — does not model bank-level DRAM timing, instruction scheduling conflicts, or the fine-grained memory–compute interactions that the transactional emulator captures via Ramulator / DRAMSys.
- **Intended use** — coarse ranking and pruning of the design space; promising candidates are then re-evaluated with the Transactional Emulator for high-fidelity latency numbers.

## Area and Power Model

Analytical cost functions for on-chip area and power, calibrated against hardware synthesis results.

- **Built from synthesis samples** — a set of representative hardware configurations is pushed through the RTL synthesis flow, and the reported area and power numbers are collected.
- **Curve fitting over the design space** — the sampled points are fit to parameterised cost functions of the PLENA configuration knobs (tile sizes, SRAM depths, precision bit widths, etc.), so the model can interpolate between sampled configurations without re-running synthesis.
- **Output** — per-configuration estimates of total silicon area and average power, suitable as optimization objectives or hard constraints.

The underlying RTL and cost-modeling assets live in the `PLENA_RTL/` submodule:

- **`src/definitions/configuration.svh`** — Hardware configuration knobs that parameterise the fitted models.
- **`src/definitions/precision.svh`** — Precision settings (MXFP / MXINT / FP bit widths) consumed by the area/power fit.
- **`tools/cost_model/`** — The fitted area, power, and performance models together with the sampling scripts used to calibrate them.

## Interface

The co-design layer queries the analytic model through:

- **`get_area()`** in `co_design/interface/interface.py` — Returns the fitted area (and power) estimate for the active configuration.
- **`get_latency()`** (fast path) — Returns the closed-form performance estimate when the search loop is pruning; the transactional emulator is used instead when a high-fidelity number is required.

Precision settings are parsed from the active TOML configuration via `parse_precision_config()` and converted to bit widths with `get_bit_width()` in `co_design/interface/utils.py`.

## Co-Design Search Integration

Because the analytic model is cheap to evaluate, it is the innermost cost query inside the Optuna search loop in `co_design/search/search.py`:

```
┌─────────┐   ┌─────────────┐   ┌──────────────────────┐
│ Sample  │──▶│  Validate   │──▶│  Analytic Model      │
│ Params  │   │ Constraints │   │  (perf / area / pwr) │
└─────────┘   └─────────────┘   └──────────┬───────────┘
     ▲                                     │
     │                                     ▼
     │                          ┌──────────────────────┐
     │                          │  Promising configs → │
     │                          │  Transactional Emu + │
     │                          │  Accuracy Evaluator  │
     │                          └──────────┬───────────┘
     │                                     │
     └──────── Update Pareto Front ◀───────┘
```

`check_constraints()` in `co_design/search/utils.py` rejects configurations that violate the hardware constraints documented in the Hardware Configuration page before any cost query is issued.

### Supported Samplers

| Sampler | Algorithm | Best For |
|---------|-----------|----------|
| `botorch` | Bayesian Optimization | Sample-efficient exploration |
| `tpe` | Tree-structured Parzen Estimator | High-dimensional spaces |
| `nsga2` | NSGA-II | Multi-objective evolutionary search |
| `random` | Random Search | Baseline comparison |
