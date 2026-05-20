# Toolchain Overview

The PLENA toolchain is a full-stack framework that takes a Hugging Face model description and pre-trained weights through software optimization, simulation, and hardware evaluation. It is organized into three layers connected by a Co-Design Design Space Exploration (DSE) engine that drives two optimization loops: a **Software Optimization Loop** that tunes data type and precision settings against accuracy, and a **Hardware Optimization Loop** that tunes the accelerator configuration against Power, Performance, and Area (PPA).

<div style="text-align: center;">
  <img src="figs/PLENA_Sys.png" alt="PLENA System Architecture" style="max-width: 90%;">
</div>

---

## Software Optimization Layer

This layer ingests the model description and weights and produces both quantized weights and machine code for the accelerator.

- **Compiler** — Lowers a Hugging Face model onto the PLENA stack. It contains:
    - *Parser* — Converts the model graph into the compiler's internal representation.
    - *Scheduling* — Orders and tiles operators to match the accelerator's pipeline and memory hierarchy.
    - *Code Gen* — Emits instruction sequences targeting the PLENA ISA.
    - *Assembler* — Produces the final machine code consumed by the simulator and the hardware.
- **Training-Free Quantization Optimization Flow** — Applies post-training quantization without fine-tuning, using:
    - *W-Quan* — Weight quantization.
    - *KV-Rotation* — Rotation applied to the KV cache to suppress outliers prior to quantization.
    - *A-Rotation* — Rotation applied to activations for the same purpose.
    - *Quan* — The activation/KV quantization step that consumes the rotated tensors.
- **Accuracy Evaluator** — Measures the model accuracy under the chosen quantization settings, providing the feedback signal that closes the Software Optimization Loop.

---

## Simulation Layer

This layer consumes the generated machine code and the hardware configuration and provides fast, model-based performance estimates.

- **Transactional Simulator (HBM Enabled)** — A cycle-approximate simulator that executes the generated machine code with an HBM memory model, capturing the behaviour relevant to long-context inference.
- **Regression-based Simulation** — Lightweight analytical models used inside the DSE inner loop for rapid evaluation:
    - *Latency Model* — Predicts end-to-end runtime.
    - *Area Model* — Predicts silicon area.
    - *Power Model* — Predicts power consumption.

---

## Hardware Layer

This layer realizes a configuration as actual RTL and provides ground-truth PPA numbers.

- **PLENA Accelerator** — The configurable RTL implementation parameterized by the hardware configuration supplied by the DSE.
- **Hardware Evaluation** — Performs the slow, high-fidelity evaluation path:
    - *Synthesis* — Produces area, timing, and power estimates from the synthesized design.
    - *RTL Simulation* — Verifies functional correctness and gathers cycle-accurate performance data.

---

## Co-Design DSE

The **Co-Design DSE** sits at the centre of the toolchain and explores the joint software/hardware design space. It drives two coupled loops:

- **Software Optimization Loop** (green) — The DSE proposes *Data Type and Precision Settings*; the Software Optimization Layer applies them and reports back model accuracy from the Accuracy Evaluator.
- **Hardware Optimization Loop** (blue) — The DSE proposes a *Hardware Config*; PPA is obtained in two modes:
    - *PPA Fast Mode* — Uses the regression-based latency/area/power models for rapid inner-loop search.
    - *PPA Slow Mode* — Uses synthesis and RTL simulation on the PLENA Accelerator for high-fidelity validation of promising candidates.

Together these loops let PLENA jointly optimize quantization choices and hardware configuration for long-context LLM inference workloads.
