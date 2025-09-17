# problem_8.py
import torch
import triton
import triton.language as tl
import math
from typing import Optional

"""
Valid + Causal Masking is applied to both phases and both forward and backward kernels.
"""

@triton.jit
def _flash_attention_forward_gqa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,              # <-- add M_ptr
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    m_stride_b, m_stride_h, m_stride_s,             # <-- add M strides
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel template for the forward pass of causal FlashAttention with GQA.
    """
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # 1. GQA
    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
        (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)

    qk_scale = softmax_scale * 1.44269504
    
    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        # Load K_j
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
            (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # Compute the attention scores (S_ij).
        s_ij = tl.dot(q_block.to(tl.float32), k_block.to(tl.float32))
        s_ij *= qk_scale

        v_block = v_block.to(tl.float32)

        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        scale_factor = tl.exp2(m_i - m_new)
        acc = acc * scale_factor[:, None]
        l_i = l_i * scale_factor
        p_ij = tl.exp2(s_ij - m_new[:, None])
        acc = acc + tl.dot(p_ij, v_block)
        l_i = l_i + tl.sum(p_ij, axis=1)
        m_i = m_new

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # Load K_j
        k_offsets = start_n + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
            (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        s_ij = tl.dot(q_block.to(tl.float32), k_block.to(tl.float32))
        s_ij *= qk_scale

        v_block = v_block.to(tl.float32)

        # Build mask
        # Lower triangle true
        causal = q_offsets[:, None] >= k_offsets[None, :]
        valid = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
        mask = causal & valid

        # Apply mask BEFORE tile max so future tokens don't affect m_i
        s_ij = tl.where(mask, s_ij, -float("inf"))

        # Online softmax update
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        scale_factor = tl.exp2(m_i - m_new)

        p_ij = tl.exp2(s_ij - m_new[:, None])

        acc = acc * scale_factor[:, None] + tl.dot(p_ij, v_block)
        l_i = l_i * scale_factor + tl.sum(p_ij, axis=1)
        m_i = m_new

    # 4. Normalize and write the final output block.
    l_i_safe = l_i[:, None] + 1e-6
    acc = acc / l_i_safe
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
        (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)

    lse_log2 = m_i + tl.log2(l_i + 1e-6)
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    tl.store(m_ptrs, lse_log2, mask=q_offsets < SEQ_LEN)

@triton.jit
def _flash_attention_backward_gqa_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, M_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    M_stride_b, M_stride_h, M_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Params
    softmax_scale, SEQ_LEN, N_Q_HEADS, N_KV_HEADS,
    # Constexpr
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ids
    q_block_idx = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)

    # Map to batch + q-head
    b_idx = batch_head_id // N_Q_HEADS
    q_head_idx = batch_head_id % N_Q_HEADS

    # GQA mapping
    group_size  = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # Row offsets for this block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = q_offsets < SEQ_LEN
    col = tl.arange(0, HEAD_DIM)

    # Load ptrs
    q_ptrs = Q_ptr + b_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + col[None, :])
    do_ptrs = dO_ptr + b_idx * do_stride_b + q_head_idx * do_stride_h + (q_offsets[:, None] * do_stride_s + col[None, :])
    o_ptrs = O_ptr + b_idx * o_stride_b + q_head_idx * o_stride_h + (q_offsets[:, None] * o_stride_s + col[None, :])
    M_ptrs = M_ptr + b_idx * M_stride_b + q_head_idx * M_stride_h + q_offsets * M_stride_s
    lse_log2_row = tl.load(M_ptrs, mask=row_mask, other=-float('inf'))

    q_block = tl.load(q_ptrs,  mask=row_mask[:, None], other=0.0).to(tl.float32)
    do_block = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    o_block = tl.load(o_ptrs,  mask=row_mask[:, None], other=0.0).to(tl.float32)

    qk_scale = softmax_scale * 1.44269504

    # Pass 0: delta = sum(dO * O) per row
    delta = tl.sum(do_block * o_block, axis=1)

    # Pass 1: recompute m_i and l_i (online) over causal prefix
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Off-diagonal tiles (strictly before this block)
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        k_ptrs = K_ptr + b_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_blk = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)

        # Scores
        s = tl.dot(q_block, k_blk) * qk_scale

        causal = q_offsets[:, None] >= k_offsets[None, :]
        valid = row_mask[:, None] & kv_valid[None, :]
        s = tl.where(causal & valid, s, -float("inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)

        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Diagonal block range
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        k_ptrs = K_ptr + b_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_blk = tl.load(k_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)

        s = tl.dot(q_block, k_blk) * qk_scale

        causal = q_offsets[:, None] >= k_offsets[None, :]
        #valid = row_mask[:, None] & kv_valid[None, :]
        s = tl.where(causal, s, -float("inf"))

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_new)

        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Pass 2: compute grads
    dQ_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # off-diagonal tiles
    for start_n in range(0, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        # K for scores: (HEAD_DIM, BLOCK_N)
        k_cols_ptrs = K_ptr + b_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_cols = tl.load(k_cols_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)

        # K for dQ: (BLOCK_N, HEAD_DIM)
        k_rows_ptrs = K_ptr + b_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_rows = tl.load(k_rows_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)

        # V: (BLOCK_N, HEAD_DIM)
        v_ptrs = V_ptr + b_idx * v_stride_b + kv_head_idx * v_stride_h + \
            (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_blk = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)

        # Scores (BLOCK_M, BLOCK_N)
        s = tl.dot(q_block, k_cols) * qk_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        valid = row_mask[:, None] & kv_valid[None, :]
        mask = causal & valid
        s = tl.where(mask, s, -float("inf"))

        p = tl.exp2(s - lse_log2_row[:, None])
        p = tl.where(mask, p, 0.0)

        dV_tile = tl.dot(p.T, do_block)      # (BLOCK_N, HEAD_DIM)
        dp = tl.dot(do_block, v_blk.T)  # (BLOCK_M, BLOCK_N)
        # dS = dp - p * delta[:, None]  # (BLOCK_M, BLOCK_N) - doesn't work
        dS = (dp - delta[:, None]) * p

        dQ_acc += tl.dot(dS, k_rows) * softmax_scale
        dK_tile = tl.dot(dS.T, q_block) * softmax_scale

        dk_ptrs = dK_ptr + b_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
            (k_offsets[:, None] * dk_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        dv_ptrs = dV_ptr + b_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
            (k_offsets[:, None] * dv_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        # Doing this means casting the atomic values to bf16, it fails test cases
        # Instead, create fp32 accumulators in HBM
        # tl.atomic_add(dk_ptrs, dK_tile.to(dk_ptrs.dtype.element_ty), mask=kv_valid[:, None])
        # tl.atomic_add(dv_ptrs, dV_tile.to(dv_ptrs.dtype.element_ty), mask=kv_valid[:, None])
        tl.atomic_add(dk_ptrs, dK_tile, mask=kv_valid[:, None])
        tl.atomic_add(dv_ptrs, dV_tile, mask=kv_valid[:, None])

    # Diagonal tiles
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid  = k_offsets < SEQ_LEN

        # K for scores: (HEAD_DIM, BLOCK_N)
        k_cols_ptrs = K_ptr + b_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_cols = tl.load(k_cols_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)

        # K for dQ: (BLOCK_N, HEAD_DIM)
        k_rows_ptrs = K_ptr + b_idx * k_stride_b + kv_head_idx * k_stride_h + \
            (k_offsets[:, None] * k_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        k_rows = tl.load(k_rows_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)

        # V: (BLOCK_N, HEAD_DIM)
        v_ptrs = V_ptr + b_idx * v_stride_b + kv_head_idx * v_stride_h + \
            (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_blk = tl.load(v_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)

        s = tl.dot(q_block, k_cols) * qk_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        valid = row_mask[:, None] & kv_valid[None, :]
        mask = causal & valid
        s = tl.where(mask, s, -float("inf"))

        p = tl.exp2(s - lse_log2_row[:, None])
        p = tl.where(mask, p, 0.0)

        dV_tile = tl.dot(p.T, do_block)
        dp = tl.dot(do_block, v_blk.T)
        # dS = dp - p * delta[:, None] doesn't work
        dS = (dp - delta[:, None]) * p

        dQ_acc += tl.dot(dS, k_rows) * softmax_scale
        dK_tile = tl.dot(dS.T, q_block) * softmax_scale

        dk_ptrs = dK_ptr + b_idx * dk_stride_b + kv_head_idx * dk_stride_h + \
            (k_offsets[:, None] * dk_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        dv_ptrs = dV_ptr + b_idx * dv_stride_b + kv_head_idx * dv_stride_h + \
            (k_offsets[:, None] * dv_stride_s + tl.arange(0, HEAD_DIM)[None, :])

        tl.atomic_add(dk_ptrs, dK_tile, mask=kv_valid[:, None])
        tl.atomic_add(dv_ptrs, dV_tile, mask=kv_valid[:, None])

    # Store dQ (unique writer)
    dq_ptrs = dQ_ptr + b_idx * dq_stride_b + q_head_idx * dq_stride_h + \
        (q_offsets[:, None] * dq_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(dq_ptrs, dQ_acc.to(dQ_ptr.dtype.element_ty), mask=row_mask[:, None])


class FlashAttention2Function(torch.autograd.Function):
    """
    Triton implementation of FlashAttention-2, supports causal attention and GQA.
    """
    @staticmethod
    def forward(ctx, q, k, v, is_causal=True, softmax_scale: Optional[float] = None):
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        assert is_causal, "This kernel only supports causal attention"
        assert n_heads % n_kv_heads == 0, "num_attention_heads must be divisible by num_kv_heads"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)

        o = torch.empty_like(q)
        M = torch.empty((batch, n_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)
        
        # TODO: Add your forward kernel here
        _flash_attention_forward_gqa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_heads,
            n_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.num_heads = n_heads
        ctx.num_kv_heads = n_kv_heads
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        batch, n_heads, seq_len, head_dim = q.shape
        n_kv_heads = ctx.num_kv_heads

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        # DO NOT CHANGE THE DTYPES IN THE TEMPLATE
        # dq = torch.empty_like(q, dtype=torch.float32)
        # dk = torch.zeros_like(k, dtype=torch.float32)
        # dv = torch.zeros_like(v, dtype=torch.float32)
        
        # [OPTIONAL BONUS] STUDENT IMPLEMENTATION REQUIRED
        # Implement the Triton backward kernel for GQA from scratch.
        # You should:
        #   1. Precompute delta = sum(dO * O)
        #   2. Recompute attention probabilities P = softmax(QK^T)
        #   3. Use delta + dO to accumulate gradients for dq, dk, dv
        #   4. Respect GQA mapping and causal mask

        # Instead, specify the dtype on the accumulation tensors
        dk_acc = torch.zeros_like(k, dtype=torch.float32)
        dv_acc = torch.zeros_like(v, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

        _flash_attention_backward_gqa_kernel[grid](
            # Pointers
            q, k, v, o, do, M,
            dq, dk_acc, dv_acc,
            # Strides (q, k, v, o, dO, dQ, dK, dV)
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            # Params
            ctx.softmax_scale, seq_len, n_heads, n_kv_heads,
            # Constexpr
            HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # Write them back to HBM
        # copy_ is in place
        dk.copy_(dk_acc)
        dv.copy_(dv_acc)

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None


def flash_attention_gqa(q, k, v, is_causal=True, softmax_scale=None):
    return FlashAttention2Function.apply(q, k, v, is_causal, softmax_scale)