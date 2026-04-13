# Getting Started

## PLENA Simulator

### Quick Start

#### Prerequisites

Install the following tools before setting up the simulator:

- [**nix**](https://nixos.org/download) — package manager used to provision the toolchain
- [**direnv**](https://direnv.net/) — automatic environment loading
- [**plena_compiler**](https://github.com/AICrossSim/PLENA_Compiler) — compiler stack targeting the PLENA ISA

Enable the `direnv` hook in your shell:

```bash
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

#### Installation

```bash
# Allow direnv to load the environment
direnv allow

# Enter the development environment
nix develop

# Update git submodules
git submodule update --remote --merge
```

#### Running Tests

Use the following commands to run simulator tasks.

Standard mode:

```bash
just build-emulator [task]
# Example:
just build-behave-sim linear
```

Debug mode:

```bash
just build-emulator-debug [task]
# Example:
just build-behave-sim-debug linear
```