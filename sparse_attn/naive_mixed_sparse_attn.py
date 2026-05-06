import torch


@torch.inference_mode()
def naive_mixed_sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlens: torch.Tensor,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    B, H, N_CTX, D = q.shape
    NUM_ROWS = (N_CTX + block_size_M - 1) // block_size_M

    q_f32 = (q * (sm_scale * 1.44269504)).float()
    out = torch.zeros_like(q_f32)
    offs_n = torch.arange(block_size_N, device=q.device)

    for b in range(B):
        seqlen = int(seqlens[b])
        k_b = k[b].float()
        v_b = v[b].float()
        q_b = q_f32[b]

        for h in range(H):
            bc_h = block_count[b, h]
            cc_h = column_count[b, h]
            bo_h = block_offset[b, h]
            ci_h = column_index[b, h]
            k_h = k_b[h]
            v_h = v_b[h]

            for start_m in range(NUM_ROWS):
                q_start = start_m * block_size_M
                if q_start >= seqlen:
                    continue
                q_end = min(q_start + block_size_M, seqlen)
                q_len = q_end - q_start

                q_blk = q_b[h, q_start:q_end]
                q_pos = q_start + offs_n[:q_len]

                n_slash = int(bc_h[start_m])
                n_vert = int(cc_h[start_m])

                m_i = torch.full((q_len,), float("-inf"), dtype=torch.float32, device=q.device)
                l_i = torch.zeros(q_len, dtype=torch.float32, device=q.device)
                acc = torch.zeros(q_len, D, dtype=torch.float32, device=q.device)

                # --- slash blocks: batch all K/V into one big matmul ---
                if n_slash > 0:
                    blk_starts = bo_h[start_m, :n_slash]  # [n_slash]

                    blk_offsets = blk_starts[:, None] + offs_n[None, :]  # [n_slash, BLOCK_N]
                    blk_valid = blk_offsets < seqlen
                    blk_lens = blk_valid.sum(dim=1)  # actual length of each block
                    all_pos = blk_offsets[blk_valid]  # flattened valid positions

                    k_all = k_h[all_pos]  # [total_slash, D]
                    v_all = v_h[all_pos]

                    qk_all = q_blk @ k_all.T  # [q_len, total_slash]

                    # Process each block's portion with online softmax
                    pos = 0
                    for si in range(n_slash):
                        k_len = int(blk_lens[si])
                        if k_len == 0:
                            continue
                        qk = qk_all[:, pos:pos + k_len]

                        if causal:
                            ks = int(blk_starts[si])
                            k_pos = ks + offs_n[:k_len]
                            qk = qk.masked_fill(k_pos[None, :] > q_pos[:, None], float("-inf"))

                        m_i_new = torch.maximum(m_i, qk.amax(1))
                        alpha = torch.exp2(m_i - m_i_new)
                        p = torch.exp2(qk - m_i_new[:, None])

                        acc *= alpha[:, None]
                        acc += p @ v_all[pos:pos + k_len]

                        l_i = l_i * alpha + p.sum(1)
                        m_i = m_i_new
                        pos += k_len

                # --- vertical columns ---
                if n_vert > 0:
                    for vi in range(0, n_vert, block_size_N):
                        vi_end = min(vi + block_size_N, n_vert)
                        cols = ci_h[start_m, vi:vi_end]
                        valid = cols < seqlen
                        cols_f = cols[valid]
                        k_len = cols_f.shape[0]

                        if k_len == 0:
                            continue

                        k_blk = k_h[cols_f]
                        v_blk = v_h[cols_f]

                        qk = q_blk @ k_blk.T
                        qk = qk.masked_fill(~valid[None, :], float("-inf"))

                        m_i_new = torch.maximum(m_i, qk.amax(1))
                        alpha = torch.exp2(m_i - m_i_new)
                        p = torch.exp2(qk - m_i_new[:, None])

                        acc *= alpha[:, None]
                        acc += p @ v_blk

                        l_i = l_i * alpha + p.sum(1)
                        m_i = m_i_new

                acc /= l_i[:, None]
                out[b, h, q_start:q_end] = acc

    return out.to(q.dtype)
