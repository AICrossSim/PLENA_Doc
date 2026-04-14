# Accuracy Evaluator

The Accuracy Evaluator measures how PLENA's quantization choices affect model quality on real workloads. Given a target model and a precision configuration (weight / activation / KV-cache formats, bit widths, scale granularity), it runs the model on a specified workload and reports task accuracy, so the co-design loop can trade precision for area and latency without silently degrading model quality.

It lives in the [`PLENA_Software`](https://github.com/AICrossSim/PLENA_Software) repository.

## What It Evaluates

- **Quantization fidelity** — Simulates PLENA's on-device numerics (MXFP / MXINT / FP formats with configurable mantissa, exponent, and shared-scale granularity) applied to weights, activations, and the KV cache.
- **Asymmetric mixed precision** — Supports independent precision settings per tensor class, matching the full PLENA design space.
- **Workload-driven scoring** — Runs the quantized model on the workload specified by the user (e.g. long-context LLM inference tasks) and reports the corresponding task metric.

## Interface

The co-design layer queries the evaluator through:

- **`get_accuracy()`** in `co_design/interface/interface.py` — Accepts a precision configuration, invokes `PLENA_Software` to run the workload under that configuration, and returns the resulting accuracy metric for use as an optimization objective.

Precision settings are parsed from the active TOML configuration via `parse_precision_config()` and `get_bit_width()` in `co_design/interface/utils.py`, then passed through to `PLENA_Software` before each evaluation.
