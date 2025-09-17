import torch
import triton
import triton.language as tl
import math
from typing import Optional

"""
Why is it that non_sink & in_window works for phase 2 too?
Although our implementation does non_sink & causal?
Because the window is usually set up to only include causal positions
in the diagonal block.

Naming convetion (diff from problem 7)
Kc is [D,N] - block of key vectors arranged as cols
Vt is [N,D] - block of value vectors, transposed for matmul

BATCH_SIZE is passed to backward kernel, but not used
But could be used for bounds checking

Phase 0: sink_cols & causal
Phase 1: non_sink & window_mask & pre_diag_mask
Phase 2: non_sink & causal
"""


@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ids
    q_block_idx = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)
    batch_idx = batch_head_id // N_Q_HEADS
    q_head_idx = batch_head_id % N_Q_HEADS

    # GQA mapping
    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # Accumulators
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load Q
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = q_offsets < SEQ_LEN
    cols = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
        (q_offsets[:, None] * q_stride_s + cols[None, :])
    q_block = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)

    qk_scale = softmax_scale * 1.44269504

    # Sliding window
    q_start = q_block_idx * BLOCK_M
    win_left = q_start - (WINDOW_SIZE - 1)
    window_start = tl.maximum(0, win_left)
    diag_start = q_block_idx * BLOCK_M

    # Phase 0: Sink Tiles
    for start_n in range(0, SINK_SIZE, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        Kc_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[None, :]*k_stride_s + cols[:, None])
        V_ptrs = V_ptr + batch_idx*v_stride_b + kv_head_idx * \
            v_stride_h + (k_offsets[:, None]*v_stride_s + cols[None, :])

        Kc = tl.load(Kc_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)
        Vt = tl.load(V_ptrs,  mask=kv_valid[:, None], other=0.0).to(tl.float32)

        S = tl.dot(q_block, Kc) * qk_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        sink_cols = k_offsets[None, :] < SINK_SIZE
        #valid = row_mask[:, None] & kv_valid[None, :]
        mask = sink_cols & causal

        S = tl.where(mask, S, -float('inf'))

        row_has = tl.max(mask, axis=1) > 0
        m_ij = tl.max(S, axis=1)
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        alpha = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)
        P = tl.where(row_has[:, None], tl.exp2(S - m_new[:, None]), 0.0)

        acc = acc * alpha[:, None] + tl.dot(P, Vt)
        l_i = l_i * alpha + tl.sum(P, axis=1)
        m_i = m_new

    # Phase 1: Off-Diagonal Tiles
    for start_n in range(window_start, diag_start, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        Kc_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[None, :]*k_stride_s + cols[:, None])
        V_ptrs = V_ptr + batch_idx*v_stride_b + kv_head_idx * \
            v_stride_h + (k_offsets[:, None]*v_stride_s + cols[None, :])

        Kc = tl.load(Kc_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)
        Vt = tl.load(V_ptrs,  mask=kv_valid[:, None], other=0.0).to(tl.float32)

        S = tl.dot(q_block, Kc) * qk_scale

        dist = q_offsets[:, None] - k_offsets[None, :]
        in_window = (dist >= 0) & (dist < WINDOW_SIZE)
        pre_diag = k_offsets[None, :] < diag_start
        non_sink = k_offsets[None, :] >= SINK_SIZE
        # valid = row_mask[:,None] & kv_valid[None,:]
        mask = pre_diag & non_sink & in_window

        S = tl.where(mask, S, -float('inf'))

        row_has = tl.max(mask, axis=1) > 0
        m_ij = tl.max(S, axis=1)
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        alpha = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)
        P = tl.where(row_has[:, None], tl.exp2(S - m_new[:, None]), 0.0)

        acc = acc*alpha[:, None] + tl.dot(P, Vt)
        l_i = l_i*alpha + tl.sum(P, axis=1)
        m_i = m_new

    # Phase 2: Diagonal Tiles
    for start_n in range(diag_start, (q_block_idx+1)*BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        Kc_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[None, :]*k_stride_s + cols[:, None])
        V_ptrs = V_ptr + batch_idx*v_stride_b + kv_head_idx * \
            v_stride_h + (k_offsets[:, None]*v_stride_s + cols[None, :])

        Kc = tl.load(Kc_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)
        Vt = tl.load(V_ptrs,  mask=kv_valid[:, None], other=0.0).to(tl.float32)

        S = tl.dot(q_block, Kc) * qk_scale

        # dist = q_offsets[:,None] - k_offsets[None,:]
        # in_window = (dist >= 0) & (dist < WINDOW_SIZE)
        # valid = row_mask[:,None] & kv_valid[None,:]
        non_sink = k_offsets[None, :] >= SINK_SIZE
        causal = q_offsets[:, None] >= k_offsets[None, :]
        mask = non_sink & causal

        S = tl.where(mask, S, -float('inf'))

        row_has = tl.max(mask, axis=1) > 0
        m_ij = tl.max(S, axis=1)
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        alpha = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)
        P = tl.where(row_has[:, None], tl.exp2(S - m_new[:, None]), 0.0)

        acc = acc * alpha[:, None] + tl.dot(P, Vt)
        l_i = l_i * alpha + tl.sum(P, axis=1)
        m_i = m_new

    # Normalize
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    O = acc / l_i_safe[:, None]
    o_ptrs = O_ptr + batch_idx*o_stride_b + q_head_idx * \
        o_stride_h + (q_offsets[:, None] * o_stride_s + cols[None, :])
    tl.store(o_ptrs, O.to(O_ptr.dtype.element_ty), mask=row_mask[:, None])

    # Store per-row log2sumexp in M
    lse_log2 = m_i + tl.log2(l_i + 1e-6)
    m_ptrs = M_ptr + batch_idx * m_stride_b + \
        q_head_idx * m_stride_h + q_offsets*m_stride_s
    tl.store(m_ptrs, lse_log2, mask=row_mask)


@triton.jit
def _flash_attention_backward_swa_kernel(
    # In/Out Pointers
    Q_ptr, K_ptr, V_ptr, dO_ptr, M_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    do_stride_b, do_stride_h, do_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    d_stride_b, d_stride_h, d_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s,
    dv_stride_b, dv_stride_h, dv_stride_s,
    # Parameters
    softmax_scale,
    BATCH_SIZE: int,
    N_Q_HEADS: int,
    N_KV_HEADS: int,
    SEQ_LEN: int,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Tile Sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 1) Ids
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # 2) GQA mapping
    group_size = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // group_size

    # 3) Indices
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = q_offsets < SEQ_LEN
    cols = tl.arange(0, HEAD_DIM)

    # 4) load blocks
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * \
        q_stride_h + (q_offsets[:, None] * q_stride_s + cols[None, :])
    do_ptrs = dO_ptr + batch_idx * do_stride_b + q_head_idx * \
        do_stride_h + (q_offsets[:, None] * do_stride_s + cols[None, :])
    m_ptrs = M_ptr + batch_idx * m_stride_b + \
        q_head_idx * m_stride_h + q_offsets * m_stride_s
    d_ptrs = D_ptr + batch_idx * d_stride_b + \
        q_head_idx * d_stride_h + q_offsets * d_stride_s

    Q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    dO = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    LSE = tl.load(m_ptrs, mask=row_mask, other=-float('inf')).to(tl.float32)
    delta = tl.load(d_ptrs, mask=row_mask, other=0.0).to(tl.float32)

    qk_scale = softmax_scale * 1.44269504
    dQ_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Sliding window
    q_start = q_block_idx * BLOCK_M
    win_left = q_start - (WINDOW_SIZE - 1)
    window_start = tl.maximum(0, win_left)
    diag_start = q_block_idx * BLOCK_M

    # Phase 0
    for start_n in range(0, SINK_SIZE, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        Kc_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[None, :]*k_stride_s + cols[:, None])
        Kr_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[:, None]*k_stride_s + cols[None, :])
        Vr_ptrs = V_ptr + batch_idx*v_stride_b + kv_head_idx * \
            v_stride_h + (k_offsets[:, None]*v_stride_s + cols[None, :])

        Kc = tl.load(Kc_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)
        Kr = tl.load(Kr_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)
        Vr = tl.load(Vr_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)

        S = tl.dot(Q, Kc) * qk_scale
        causal = q_offsets[:, None] >= k_offsets[None, :]
        sinkcol = k_offsets[None, :] < SINK_SIZE
        # valid = row_mask[:,None] & kv_valid[None,:]
        mask = sinkcol & causal

        S = tl.where(mask, S, -float('inf'))
        P = tl.where(mask, tl.exp2(S - LSE[:, None]), 0.0)

        dV = tl.dot(P.T, dO)
        dp = tl.dot(dO, Vr.T)
        dS = (dp - delta[:, None]) * P

        dQ_acc += tl.dot(dS, Kr) * softmax_scale
        dK = tl.dot(dS.T, Q) * softmax_scale

        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * \
            dk_stride_h + (k_offsets[:, None]*dk_stride_s + cols[None, :])
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * \
            dv_stride_h + (k_offsets[:, None]*dv_stride_s + cols[None, :])
        tl.atomic_add(dk_ptrs, dK, mask=kv_valid[:, None])
        tl.atomic_add(dv_ptrs, dV, mask=kv_valid[:, None])

    # Phase 1: Off-Diagonal Tiles
    for start_n in range(window_start, diag_start, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        Kc_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[None, :] * k_stride_s + cols[:, None])
        Kr_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[:, None] * k_stride_s + cols[None, :])
        Vr_ptrs = V_ptr + batch_idx*v_stride_b + kv_head_idx * \
            v_stride_h + (k_offsets[:, None] * v_stride_s + cols[None, :])

        Kc = tl.load(Kc_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)
        Kr = tl.load(Kr_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)
        Vr = tl.load(Vr_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)

        S = tl.dot(Q, Kc) * qk_scale

        dist = q_offsets[:, None] - k_offsets[None, :]
        in_window = (dist >= 0) & (dist < WINDOW_SIZE)
        pre_diag = k_offsets[None, :] < diag_start
        non_sink = k_offsets[None, :] >= SINK_SIZE
        # valid    = row_mask[:,None] & kv_valid[None,:]
        mask = pre_diag & non_sink & in_window

        S = tl.where(mask, S, -float('inf'))
        P = tl.where(mask, tl.exp2(S - LSE[:, None]), 0.0)

        dV = tl.dot(P.T, dO)
        dp = tl.dot(dO, Vr.T)
        dS = (dp - delta[:, None]) * P

        dQ_acc += tl.dot(dS, Kr) * softmax_scale
        dK = tl.dot(dS.T, Q) * softmax_scale

        dk_ptrs = dK_ptr + batch_idx*dk_stride_b + kv_head_idx * \
            dk_stride_h + (k_offsets[:, None] * dk_stride_s + cols[None, :])
        dv_ptrs = dV_ptr + batch_idx*dv_stride_b + kv_head_idx * \
            dv_stride_h + (k_offsets[:, None] * dv_stride_s + cols[None, :])
        tl.atomic_add(dk_ptrs, dK, mask=kv_valid[:, None])
        tl.atomic_add(dv_ptrs, dV, mask=kv_valid[:, None])

    # Phase 2: Diagonal Tiles
    for start_n in range(diag_start, (q_block_idx+1)*BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        kv_valid = k_offsets < SEQ_LEN

        Kc_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[None, :] * k_stride_s + cols[:, None])
        Kr_ptrs = K_ptr + batch_idx*k_stride_b + kv_head_idx * \
            k_stride_h + (k_offsets[:, None] * k_stride_s + cols[None, :])
        Vr_ptrs = V_ptr + batch_idx*v_stride_b + kv_head_idx * \
            v_stride_h + (k_offsets[:, None] * v_stride_s + cols[None, :])

        Kc = tl.load(Kc_ptrs, mask=kv_valid[None, :], other=0.0).to(tl.float32)
        Kr = tl.load(Kr_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)
        Vr = tl.load(Vr_ptrs, mask=kv_valid[:, None], other=0.0).to(tl.float32)

        S = tl.dot(Q, Kc) * qk_scale

        # dist = q_offsets[:,None] - k_offsets[None,:]
        # in_window = (dist >= 0) & (dist < WINDOW_SIZE)
        # valid = row_mask[:,None] & kv_valid[None,:]
        non_sink = k_offsets[None, :] >= SINK_SIZE
        causal = q_offsets[:, None] >= k_offsets[None, :]
        mask = non_sink & causal

        S = tl.where(mask, S, -float('inf'))
        P = tl.where(mask, tl.exp2(S - LSE[:, None]), 0.0)

        dV = tl.dot(P.T, dO)
        dp = tl.dot(dO, Vr.T)
        dS = (dp - delta[:, None]) * P

        dQ_acc += tl.dot(dS, Kr) * softmax_scale
        dK = tl.dot(dS.T, Q) * softmax_scale

        dk_ptrs = dK_ptr + batch_idx * dk_stride_b + kv_head_idx * \
            dk_stride_h + (k_offsets[:, None] * dk_stride_s + cols[None, :])
        dv_ptrs = dV_ptr + batch_idx * dv_stride_b + kv_head_idx * \
            dv_stride_h + (k_offsets[:, None] * dv_stride_s + cols[None, :])
        tl.atomic_add(dk_ptrs, dK, mask=kv_valid[:, None])
        tl.atomic_add(dv_ptrs, dV, mask=kv_valid[:, None])

    # 5) store dQ
    dq_ptrs = dQ_ptr + batch_idx * dq_stride_b + q_head_idx * \
        dq_stride_h + (q_offsets[:, None] * dq_stride_s + cols[None, :])
    tl.store(dq_ptrs, dQ_acc.to(dQ_ptr.dtype.element_ty), mask=row_mask[:, None])


class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len),
            device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # TODO: Add your backward kernel here
        dk_acc = torch.zeros_like(k, dtype=torch.float32)
        dv_acc = torch.zeros_like(v, dtype=torch.float32)

        D = (do.float() * o.float()).sum(dim=-1)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

        _flash_attention_backward_swa_kernel[grid](
            q, k, v, do, M, D,
            dq, dk_acc, dv_acc,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            D.stride(0), D.stride(1), D.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk_acc.stride(0), dk_acc.stride(1), dk_acc.stride(2),
            dv_acc.stride(0), dv_acc.stride(1), dv_acc.stride(2),
            softmax_scale,
            batch,
            n_q_heads,
            n_kv_heads,
            seq_len,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        dk.copy_(dk_acc)
        dv.copy_(dv_acc)

        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def flash_swda_with_sink(q, k, v, window_size: int, sink_size: int = 0, is_causal: bool = True, scale: Optional[float] = None):
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)
