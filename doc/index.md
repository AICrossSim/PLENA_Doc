# PLENA: A Programmable Long-context Efficient Neural Accelerator

**PLENA** is an open-source hardware-software co-designed system optimized for long-context LLM inference in agentic applications. It addresses the memory bandwidth and capacity walls that limit compute utilization during long-context workloads such as tool-use agents, web agents, and command-line agents.

<div style="text-align: center;">
  <img src="PLENA_Sys.png" alt="PLENA System Architecture" style="max-width: 90%;">
</div>

---

## Key Features

- **Asymmetric Quantization Scheme** -- Efficient mixed-precision representation that minimizes memory footprint while preserving model accuracy
- **Flattened Systolic Array** -- A novel dataflow architecture with native FlashAttention support, achieving high compute utilization on attention-heavy workloads
- **Complete Software Stack** -- Custom ISA, compiler, cycle-emulated simulator, and automated design space exploration (DSE)
- **Multi-Objective Co-Design** -- Bayesian optimization over hardware architecture, precision settings, and inference performance jointly

## Performance Highlights

| Metric | Value |
|--------|-------|
| Compute utilization vs. existing accelerators | **8.5x** higher |
| Throughput vs. NVIDIA A100 | **2.24x** higher |
| Throughput vs. Google TPU v6e | **3.85x** higher |

*All comparisons use equivalent multiplier count and memory configurations.*

## System Components

| Component | Description |
|-----------|-------------|
| **Co-Design Engine** | Multi-objective Bayesian optimization (BoTorch, TPE, NSGA-II) over hardware and precision parameters |
| **PLENA RTL** | Synthesizable SystemVerilog hardware design with configurable datapath |
| **PLENA Simulator** | Cycle-emulated simulator for accuracy and latency evaluation |
| **PLENA Software** | Compiler and software stack for LLM inference on the PLENA accelerator |

## Project Structure

```
PLENA/
├── co_design/              # Co-design optimization engine
│   ├── interface/          # Hardware/simulator abstraction layer
│   ├── search/             # Multi-objective optimization (Optuna)
│   └── configs/            # TOML configuration files
├── PLENA_Software/         # Compiler and software stack
├── PLENA_RTL/              # SystemVerilog hardware design
├── PLENA_Simulator/        # Cycle-emulated hardware simulator
└── PLENA_Doc/              # This documentation
```

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/AICrossSim/PLENA.git

# Install dependencies
pip install -e .

# Run optimization search
python -m co_design.search.search --config co_design/configs/config.toml
```

## Publication

If you use PLENA in your research, please cite:

```bibtex
@misc{wu2025combatingmemorywallsoptimization,
    title={Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference},
    author={Haoran Wu and Can Xiao and Jiayi Nie and Xuan Guo and Binglei Lou and Jeffrey T. H. Wong and Zhiwen Mo and Cheng Zhang and Przemyslaw Forys and Wayne Luk and Hongxiang Fan and Jianyi Cheng and Timothy M. Jones and Rika Antonova and Robert Mullins and Aaron Zhao},
    year={2025},
    eprint={2509.09505},
    archivePrefix={arXiv},
    primaryClass={cs.AR},
    url={https://arxiv.org/abs/2509.09505},
}
```

[Paper Link](https://arxiv.org/abs/2509.09505)
