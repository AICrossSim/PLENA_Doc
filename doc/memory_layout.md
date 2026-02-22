# On-chip Memory Layout Convention

There are four on-chip SRAM in total.
- Matrix SRAM: Used to store the weight matrix from HBM only, cannot be written by matrix sram. And only the matrix machine can read weight datafrom this memory.
- Vector SRAM: Used as the scatchpad for activations and intermediate outputs (Not Weight Data), it has two ports:
 - Port A (RW): Used to prefetch and write back data from HBM or Used to load data to vector machine for V_* ops
 - Port B (RW): Used to load data to vector machine for V_* ops and matrix machine for M_* ops, and write back the computed result from matrix and vector sram.
- Scalar INT SRAM: Used as the extended register file for S_*_INT ops
- Scalar FP SRAM: Used as the extended register file for S_*_FP ops, we have preloaded file to this memory, it will store some FP constant for the computation.

All the on-chip address are specified in the gp register, and the address are in the unit of data elements (not related to the data type).

---

## Critical Scaling Constraints

Each `H_PREFETCH_V` loads `BLEN × VLEN` elements. When workload data exceeds this capacity, additional complexity is required:

- **Single-prefetch workloads** (≤ BLEN × VLEN elements): Can be fully unrolled without loops or offset tracking.
- **Multi-prefetch workloads** (> BLEN × VLEN elements): Require careful management of VRAM destination addresses, loop ordering, and tile accumulation.

### Multi-Tile VRAM Management

When multiple prefetches are needed, each tile must be stored at a **distinct VRAM address**. Prefetching multiple tiles to the same VRAM location will overwrite previous data, leaving only the final tile in memory.

The VRAM destination pointer must **accumulate** across tiles rather than reset. Track a running VRAM offset that advances after each prefetch.

### Stride-Mode VRAM Layout Ordering

When the output comparison uses stride-mode layout, the tile processing order matters. The expected layout groups data by **column chunk first, then batch blocks**:
- Process all batch blocks for column chunk 0
- Then all batch blocks for column chunk 1
- And so on...

Incorrect ordering (batch-first instead of column-first) produces correct values in wrong positions, causing comparison failures despite correct computation.

### Register Pressure at Scale

Larger workloads require more registers for loop counters, tile indices, and address offsets. Plan register allocation carefully to avoid exhausting the 16 available general-purpose registers.

---

# On-chip Vector SRAM Memory Layout Convention

By default, the memory layout follows this format:

Given an input tensor with shape `[b, s, h]` where:
- `b` = batch size
- `s` = sequence length
- `h` = hidden size

The data stored in Vector SRAM is reshaped to `[h // VLEN, b, s, VLEN]`.

**Rationale:** The hidden dimension is split along the `VLEN` boundary to enable efficient multi-batch GEMM operations. This layout allows parallel processing across batches while maintaining vector-length alignment requirements.


### Example: Activation Tensor

For an activation tensor with shape `[batch=4, hidden=128]` and `VLEN=64`:

- Reshaped to: `[128//64, 4, 64]` = `[2, 4, 64]`
- Vector SRAM layout:
  - Address 0: First 64 elements of hidden dim for all batches
  - Address 256: Second 64 elements of hidden dim for all batches

### Output Tensor Location

**CRITICAL: Output location determines correctness!** The test harness reads output from a specific VRAM location. Writing to the wrong address results in low match rates even if computation is correct.

| Workload Type | Output Location | Notes |
|---------------|-----------------|-------|
| Activation (silu, relu, softmax, etc.) | **VRAM[0]** (in-place) | Overwrites input |
| Attention, MHA, GQA | **VRAM[0]** (in-place) | Overwrites Q |
| Linear, MM, BatchMatmul | **After input** | Preserves input |

Always check the workload specification for the expected output location before writing code.

# On-chip Matrix SRAM Memory Layout Convention

For a high-dimensional matrix stored in HBM, `(MLEN, MLEN)` tiles are extracted and stored in the On-chip Matrix SRAM using a **row-major tile layout**.

**Note:** Addresses are specified in units of data elements (not bytes).

## Tile Addressing

Each `(MLEN, MLEN)` tile is stored contiguously in row-major order. Tiles are addressed sequentially.

Consider a large matrix of shape `(2*MLEN, 2*MLEN)`, logically divided into four `(MLEN, MLEN)` tiles arranged as `[[0, 1], [2, 3]]`. The tiles are mapped as follows:
- **Address 0..MLEN-1:** Tile 0 of Matrix[0][0]
- **Address MLEN..2*MLEN-1:** Tile 1 of Matrix[0][1]
- **Address 2*MLEN..3*MLEN-1:** Tile 2 of Matrix[1][0]
- **Address 3*MLEN..4*MLEN-1:** Tile 3 of Matrix[1][1]

**Rationale:** The matrix is tiled and stored in the On-chip Matrix SRAM to enable efficient matrix multiplication operations. This row-major tile layout allows parallel processing across tiles while maintaining matrix-length alignment requirements.

---

# Scalar FP Memory (FP_MEM)

FP_MEM is a small memory for scalar floating-point constants. It is preloaded before execution with layer-specific constants. Use `S_LD_FP` to load values into FP registers.

**Example:**
```asm
S_LD_FP f1, gp0, 1    ; Load FP_MEM[1] into f1
S_LD_FP f3, gp0, 2    ; Load FP_MEM[2] into f3
```

The exact contents of FP_MEM depend on the layer type and are provided via the `fp_sram_layout` field in the workload configuration.

---


# Off-chip HBM Memory Convention

## MXFP Format

Data in HBM uses MXFP (Microscaling) format: each 8 elements share 1 scale byte. Each tensor is stored as `[elements][scales]` where scales size = elements / 8.

**HBM size = logical_size × 1.125** (accounts for scale bytes)

## HBM Address Registers

When multiple tensors are stored sequentially, account for MXFP overhead when computing base addresses:
```
tensor_hbm_size = tensor_element_count × 1.125
next_tensor_offset = previous_tensor_offset + previous_tensor_hbm_size
```

## Scale Register

`C_SET_SCALE_REG` tells the prefetch unit where to find scale factors. The relationship:
```
scale_location = base_addr + scale_reg + (element_offset / 8)
```

**Single tensor per base register (e.g., linear):** element_offset = 0, so `scale_reg = tensor_size`

**Multiple tensors sharing base register (e.g., attention Q/K/V):** Each tensor has different element_offset, so each needs: `scale_reg = scale_location - (element_offset / 8)`

Must call `C_SET_SCALE_REG` before each tensor's prefetch.

## Stride Register

`C_SET_STRIDE_REG` sets the row stride for strided prefetch operations.

**stride_reg = number of columns in HBM storage**, regardless of mathematical transpose:
- Tensor stored as [rows, cols] → stride_reg = cols
- Even if you compute X @ K^T, K is stored as [seq_len, head_dim] → stride_reg = head_dim

---

# Weight Prefetch Pattern (H_PREFETCH_M)

## HBM Weight Layout: Stride Mode (Column-Major)

Weights in HBM use **stride mode** (column-major layout).

For a weight matrix `W[K, out_features]`:
- Element `W[row, col]` is at HBM offset: `col * STRIDE_REG + row`
- Set `STRIDE_REG = out_features`

## Tile HBM Offset Calculation

For a `(MLEN, MLEN)` tile at position `(k_tile, out_tile)`:
```
HBM offset = out_tile * MLEN + k_tile * MLEN * STRIDE_REG
```

Call `C_SET_STRIDE_REG` when switching between weight matrices with different output dimensions.

---

# Test Environment Data Layout

In the test environment:
- **Activations are in HBM** at address 0. You MUST use `H_PREFETCH_V` to load them into Vector SRAM before use.
- **FP constants are pre-loaded** into FP_MEM (see `fp_sram_layout`).
- **Weights (if needed) are in HBM** after activations. Use `H_PREFETCH_M` to load them.

| Data | Location | Pre-loaded? | Action Required |
|------|----------|-------------|-----------------|
| Activations | HBM address 0 | No | Use `H_PREFETCH_V` to load to VRAM |
| Weights | HBM after activations | No | Use `H_PREFETCH_M` to load to MSRAM |
| FP constants | FP_MEM | Yes | Use `S_LD_FP` to load into FP registers |

**Important**: All activation and weight data must be explicitly prefetched from HBM before computation.

---

# Prefetch-Compute Pattern

The key principle for efficient computation is: **data must be in SRAM before it can be used**.

## Understanding SRAM as a Working Buffer

Think of Matrix SRAM and Vector SRAM as working buffers:
- `H_PREFETCH_M` copies data from HBM → Matrix SRAM at a specified SRAM address
- `H_PREFETCH_V` copies data from HBM → Vector SRAM at a specified SRAM address
- `M_MM` reads from SRAM addresses (not HBM!)

**Critical insight:** The first argument of `H_PREFETCH_*` is the **SRAM destination address**. If you prefetch multiple tiles, they must go to different SRAM addresses, otherwise they overwrite each other.

## Example: Why Multiple Prefetches Are Needed

For a matrix multiply that accumulates across K dimension with K > MLEN:
- You need K/MLEN weight tiles
- The inner loop does K/MLEN M_MM operations, each reading from a different Matrix SRAM tile
- **Before** the inner loop, prefetch all tiles to different SRAM addresses (tile i at address i × MLEN × MLEN)

## Mental Model

Before writing any M_MM instruction, ask: "What SRAM address does this read from? When was data written there?"

If you can't trace back to a prefetch that wrote to that exact SRAM address, the data won't be there.

---

# Activation Prefetch Pattern (H_PREFETCH_V)

## HBM vs VRAM Layout Difference

Activations in HBM and VRAM use **different layouts**:

| Memory | Layout | Description |
|--------|--------|-------------|
| HBM | `[batch, hidden]` row-major | Each batch is a contiguous row of `hidden` elements |
| VRAM | `[hidden//VLEN, batch, VLEN]` | Tiled by VLEN along hidden dimension |

This means the **VRAM destination address** and **HBM source offset** are computed differently.

## H_PREFETCH_V Operands

`H_PREFETCH_V rd, rs1, rs2, rstride, precision`

| Operand | Meaning | Computed From |
|---------|---------|---------------|
| `rd` | VRAM destination address | VRAM tile index × batch × VLEN |
| `rs1` | HBM source offset | Column offset = tile index × VLEN |
| `rs2` | HBM base address register | Usually `a0` |
| `rstride` | Use STRIDE_REG if 1 | Set to 1 for multi-batch loads |

**Critical:** `rd` and `rs1` are independent. Do NOT use the same value for both.

## Stride Mode vs Contiguous Mode

**Choose the right mode based on tensor shape:**

| Tensor Type | Shape | rstride | Reason |
|-------------|-------|---------|--------|
| 2D activation | `[batch, hidden]` | 1 | Rows are spaced by `hidden` |
| 1D bias | `[out_features]` | 0 | Contiguous, no striding |
| 1D vector | any 1D | 0 | Single row of data |

**Common bug:** Using `rstride=1` for 1D vectors like bias will load BLEN rows at stride intervals, reading garbage data beyond the actual tensor!

## Stride Mode Behavior

With `rstride=1`, H_PREFETCH_V loads `HBM_V_Prefetch_Amount` (typically `batch`) consecutive chunks, each spaced by `STRIDE_REG`:

```
Starting at HBM[rs1], loads:
  HBM[rs1 + 0*STRIDE_REG : +VLEN]  → VRAM[rd + 0*VLEN]
  HBM[rs1 + 1*STRIDE_REG : +VLEN]  → VRAM[rd + 1*VLEN]
  HBM[rs1 + 2*STRIDE_REG : +VLEN]  → VRAM[rd + 2*VLEN]
  ...
```

For activations, set `STRIDE_REG = hidden` (the row stride in HBM).

## Address Calculation Formula

For activation `[batch, hidden]` split into `num_tiles = hidden // VLEN` tiles:

```
For tile j (j = 0, 1, ..., num_tiles-1):
  VRAM destination (rd) = j × batch × VLEN
  HBM offset (rs1)      = j × VLEN
  STRIDE_REG            = hidden
```

**Key insight:**
- VRAM address grows by `batch × VLEN` per tile (accounts for all batches)
- HBM offset grows by just `VLEN` per tile (column offset within each row)

# Vector Operation Loop Requirement

## Prefetch vs Compute Granularity

`H_PREFETCH_V` and `V_*` instructions operate at different granularities:

| Operation | Granularity |
|-----------|-------------|
| `H_PREFETCH_V` | `HBM_V_Prefetch_Amount × VLEN` elements |
| `V_*` instructions | `VLEN` elements |

After prefetching N elements, you need `N / VLEN` vector operations to process all data.

## Loop Pattern

```asm
; After prefetch loads N elements to VRAM starting at base_addr:
S_ADDI_INT gp1, gp0, <base_addr>
C_LOOP_START gp2, <N / VLEN>
  V_<op> gp1, ...                  ; process VLEN elements
  S_ADDI_INT gp1, gp1, <VLEN>      ; advance pointer
C_LOOP_END gp2
```

Without the loop, only the first VLEN elements are processed.

---

# Workload Scaling and Prefetch Count

## Prefetch Capacity

Each `H_PREFETCH_V` loads `BLEN × VLEN` elements per call. When total data exceeds this capacity, multiple prefetches are required:

```
num_prefetches = ceil(total_elements / (BLEN × VLEN))
```

The number of prefetches scales with both batch size and hidden dimension. Workloads that fit in a single prefetch can use simple unrolled code, while larger workloads require loop-based tiling with proper address management.

## Tiling Strategy for Large Workloads

For workloads requiring multiple prefetches:
1. Determine the number of tiles needed based on data size and prefetch capacity
2. Allocate VRAM space for all tiles (or process tiles sequentially with proper accumulation)
3. Maintain a VRAM destination pointer that advances with each tile
4. Ensure tile processing order matches the expected output layout format

Reusing registers that hold constant values (like stride sizes) for multiple purposes can help manage register pressure in complex tiled workloads.

---

# VRAM to MSRAM Data Movement

No direct VRAM → MSRAM path exists. Use HBM as intermediate:
1. **H_STORE_V**: VRAM → HBM (converts bf16 → MXFP)
2. **H_PREFETCH_M**: HBM → MSRAM (decodes MXFP)

Both must use the same scale_reg: `scale_reg = (hbm_offset + num_elements) - hbm_offset / 8`