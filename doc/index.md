# Welcome to PLENA

**PLENA** ([GitHub](https://github.com/AICrossSim/PLENA)) is an open-source hardware-software co-designed system optimized for long-context LLM inference in agentic applications. It addresses the memory bandwidth and capacity walls that limit compute utilization during long-context workloads such as tool-use agents, web agents, and command-line agents.

The project is led by Dr. Yiren Zhao (Imperial College London) and Prof. Robert Mullins (University of Cambridge), and is funded by the Advanced Research + Invention Agency (ARIA).

<div style="text-align: center;">
  <img src="figs/PLENA_Sys.png" alt="PLENA System Architecture" style="max-width: 90%;">
</div>

---

## Project Overview

PLENA is structured around a full-stack approach to long-context agentic LLM inference:

- **Analytic modelling** of memory bandwidth and capacity bottlenecks across different hardware configurations.
- **Hardware-software co-design** combining ISA extensions, a custom compiler, and a transactional emulator to explore the design space.
- **Accuracy evaluation** to ensure optimizations (e.g., quantization, sparsity) preserve model quality on agentic benchmarks.

---

## Current Status

- ✅ Analytic model for long-context inference bottleneck analysis
- ✅ ISA specification for PLENA accelerator
- ✅ Compiler pipeline for mapping LLM operators to PLENA hardware
- ✅ Transactional emulator for cycle-approximate simulation
- ✅ Hardware-software co-design exploration framework
- ✅ Accuracy evaluator with quantization and sparsity support
- ⏹️ Full FPGA prototype integration
- ⏹️ End-to-end agentic workload benchmarking on hardware

---

## Publication

If you use PLENA in your research, please cite the following paper (to appear in ISCA 2026):

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
