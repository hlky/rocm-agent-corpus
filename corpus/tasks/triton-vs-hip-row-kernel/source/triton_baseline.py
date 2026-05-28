import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(x, y, cols: tl.constexpr, block_cols: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_cols)
    mask = offsets < cols
    vals = tl.load(x + row * cols + offsets, mask=mask, other=0.0)
    tl.store(y + row, tl.sum(vals, axis=0))


def launch_triton_baseline(x, y, rows, cols, block_cols=1024):
    row_sum_kernel[(rows,)](x, y, cols, block_cols)
