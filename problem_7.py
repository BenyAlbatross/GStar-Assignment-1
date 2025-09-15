import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
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
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # 1. Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

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

    # --- STUDENT IMPLEMENTATION REQUIRED HERE ---
    # Combine the GQA, SWA, and Sink logic.
    # Combine all code from previous problems, and add the sink logic.
    # You should have 3 phases:
    # 1. Phase 0: Sink blocks that are before the sliding window
    # 2. Phase 1: Off-Diagonal Blocks (within the window)
    # 3. Phase 2: Diagonal Blocks
    
    # Cast Q once
    q_block = q_block.to(tl.float32)
    q_start = q_block_idx * BLOCK_M
    win_left = q_start - (WINDOW_SIZE - 1)
    window_start = tl.maximum(0, win_left)

    diag_start = q_block_idx * BLOCK_M

    # Phase 0: Attetion sink only
    for start_n in range(0, SINK_SIZE, BLOCK_N): # We use the whole matrix, but only the first SINK_SIZE columns which can be > 1. So not necessarily lower triangular!!
        #Load K
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Load V
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # 2. Compute the attention scores (S_ij).
        k_block = k_block.to(tl.float32)
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        v_block = v_block.to(tl.float32)

        # Masks
        sink_cols = (k_offsets[None, :] < SINK_SIZE)
        causal    = (q_offsets[:, None] >= k_offsets[None, :])
        #valid     = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
        mask      = sink_cols & causal #Causal is needed, unlike Phase 1, because we use the whole matrix

        s_ij = tl.where(mask, s_ij, -float('inf'))

        # Row has anything valid in this tile?
        row_has = tl.max(mask, axis=1) > 0

        # Online softmax update
        m_ij  = tl.max(s_ij, axis=1)
        # Only update rows that have something valid
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        # Calculate scale factor only for rows that have something valid
        scale_factor = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)

        # Probabilities only for rows that have something valid
        p_ij = tl.where(row_has[:, None], tl.exp2(s_ij - m_new[:, None]), 0.0)

        acc = acc * scale_factor[:, None] + tl.dot(p_ij, v_block)
        l_i = l_i * scale_factor + tl.sum(p_ij, axis=1)
        m_i = m_new
    
    # Phase 1: Off-Diagonal Blocks (within the window), excl sinks
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        # Load K
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Load V
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # 2. Compute the attention scores (S_ij).
        k_block = k_block.to(tl.float32)
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        v_block = v_block.to(tl.float32)

        # EXCLUDE sinks (already handled in Phase 0)
        non_sink = k_offsets[None, :] >= SINK_SIZE
        
        # Sliding window mask
        dist = q_offsets[:, None] - k_offsets[None, :] #(BLOCK_M, BLOCK_N)
        window_mask = (dist >= 0) & (dist < WINDOW_SIZE)

        # Validity mask
        # valid_mask = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)

        # Prevent overlap with diagonal tile:
        pre_diag_mask = k_offsets[None, :] < diag_start

        # Combine masks
        mask = window_mask & pre_diag_mask & non_sink #valid not needed
        s_ij = tl.where(mask, s_ij, -float('inf'))

        # Row has anything valid in this tile?
        row_has = tl.max(mask, axis=1) > 0

        # online softmax update
        m_ij  = tl.max(s_ij, axis=1)
        # Only update rows that have something valid
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        # Calculate scale factor only for rows that have something valid
        scale_factor = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)

        # Probabilities only for rows that have something valid
        p_ij = tl.where(row_has[:, None], tl.exp2(s_ij - m_new[:, None]), 0.0)

        acc = acc * scale_factor[:, None] + tl.dot(p_ij, v_block)
        l_i = l_i * scale_factor + tl.sum(p_ij, axis=1)
        m_i = m_new

    # Phase 2: Diagonal Blocks
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        # Load K
        k_offsets = start_n + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)

        # Load V
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)

        # 2. Compute the attention scores (S_ij).
        k_block = k_block.to(tl.float32)
        s_ij = tl.dot(q_block, k_block)
        s_ij *= qk_scale
        v_block = v_block.to(tl.float32)

        # NON sink mask
        non_sink = k_offsets[None, :] >= SINK_SIZE

        # Sliding window mask
        # dist = q_offsets[:, None] - k_offsets[None, :] #(BLOCK_M, BLOCK_N)
        # window_mask = (dist >= 0) & (dist < WINDOW_SIZE)

        # Combine masks
        causal = q_offsets[:, None] >= k_offsets[None, :] #Lower triangle true
        # valid = (q_offsets[:, None] < SEQ_LEN) & (k_offsets[None, :] < SEQ_LEN)
        mask = causal & non_sink #window and valid not needed for diag

        # Apply mask BEFORE tile max so future tokens don't affect m_i
        s_ij = tl.where(mask, s_ij, -float("inf"))

        # Row has anything valid in this tile?
        row_has = tl.max(mask, axis=1) > 0

        # online softmax update
        m_ij  = tl.max(s_ij, axis=1)
        # Only update rows that have something valid
        m_new = tl.where(row_has, tl.maximum(m_i, m_ij), m_i)
        # Calculate scale factor only for rows that have something valid
        scale_factor = tl.where(row_has, tl.exp2(m_i - m_new), 1.0)

        # Probabilities only for rows that have something valid
        p_ij = tl.where(row_has[:, None], tl.exp2(s_ij - m_new[:, None]), 0.0)

        acc = acc * scale_factor[:, None] + tl.dot(p_ij, v_block)
        l_i = l_i * scale_factor + tl.sum(p_ij, axis=1)
        m_i = m_new
    # --- END OF STUDENT IMPLEMENTATION ---

    # 4. Normalize and write the final output block.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)


def flash_attention_forward(q, k, v, is_causal=True, window_size=128, sink_size=4):
    """
    Python wrapper for the SWA-enabled GQA causal FlashAttention kernel with attention sink support.
    """
    # Shape checks
    batch, n_q_heads, seq_len, head_dim = q.shape
    _, n_kv_heads, _, _ = k.shape
    
    # Assertions
    assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3]
    assert k.shape == v.shape
    assert head_dim <= 128
    assert n_q_heads % n_kv_heads == 0
    assert is_causal, "This kernel only supports causal attention"
    
    o = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    BLOCK_M, BLOCK_N = 128, 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)

    _flash_attention_forward_swa_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
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
    return o