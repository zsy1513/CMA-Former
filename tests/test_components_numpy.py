"""
Numpy-only sanity tests for CMA-Former tensor operations.

This module runs without PyTorch: we re-implement the critical
shape-transformation logic in numpy and verify the math.

Tests:
  T1: Patch unfolding  -- shape (B, M, L) -> (B, M, N, P)
  T2: Per-patch exog mean -- (B, M, L, d_x) -> (B, M, N, d_x)
  T3: Cross-market attention shape transform -- (B, M, N, D) <-> (B*N, M, D)
  T4: Regime-aware MoE mixture math
  T5: Coupling-prior gate sigmoid math
  T6: Pool attention math reproducibility

Run: python tests/test_components_numpy.py
"""

import numpy as np

np.random.seed(0)

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =====================================================================
# T1: PatchEmbedding -- patch unfolding
# =====================================================================

def test_patch_unfold():
    """T1: PyTorch tensor.unfold(dim, size, step) puts the new patch dim at
    the END of the shape, not at index `dim`. This test simulates that
    behaviour exactly using stride_tricks so we never re-introduce the
    'permute the wrong dim' bug.
    """
    from numpy.lib.stride_tricks import as_strided

    def torch_like_unfold(arr, dim, size, step):
        n = (arr.shape[dim] - size) // step + 1
        arr = np.ascontiguousarray(arr)
        new_shape = list(arr.shape)
        new_shape[dim] = n
        new_shape.append(size)
        new_strides = list(arr.strides)
        new_strides[dim] = arr.strides[dim] * step
        new_strides.append(arr.strides[dim])
        return as_strided(arr, shape=new_shape, strides=new_strides)

    B, M, L, d_x, d_z = 4, 4, 336, 3, 9
    P, S = 16, 8
    N = (L - P) // S + 1                                      # = 41

    price = np.random.randn(B, M, L).astype(np.float32)
    exog  = np.random.randn(B, M, L, d_x).astype(np.float32)
    glob  = np.random.randn(B, L, d_z).astype(np.float32)

    # Price: (B, M, L) -> (B, M, N, P)
    patches = torch_like_unfold(price, dim=2, size=P, step=S)
    assert patches.shape == (B, M, N, P), patches.shape

    # Exog: (B, M, L, d_x) -> (B, M, N, d_x, P), then mean over patch (last) dim
    exog_unf = torch_like_unfold(exog, dim=2, size=P, step=S)
    assert exog_unf.shape == (B, M, N, d_x, P), exog_unf.shape
    exog_patch = exog_unf.mean(axis=-1)
    assert exog_patch.shape == (B, M, N, d_x)

    # Glob: (B, L, d_z) -> (B, N, d_z, P), then mean over patch dim
    glob_unf = torch_like_unfold(glob, dim=1, size=P, step=S)
    assert glob_unf.shape == (B, N, d_z, P), glob_unf.shape
    glob_patch = glob_unf.mean(axis=-1)
    assert glob_patch.shape == (B, N, d_z)
    glob_patch_bc = np.broadcast_to(glob_patch[:, None, :, :], (B, M, N, d_z))

    # Numerical check on exog mean
    b_, m_, n_, k = 1, 2, 10, 1
    expected = exog[b_, m_, n_*S:n_*S+P, k].mean()
    assert np.isclose(expected, exog_patch[b_, m_, n_, k], atol=1e-6)

    # Final concat
    cat = np.concatenate([patches, exog_patch, glob_patch_bc], axis=-1)
    expected_dim = P + d_x + d_z
    assert cat.shape == (B, M, N, expected_dim), cat.shape

    print(f'T1 passed: patches{patches.shape}, '
          f'concat last_dim={cat.shape[-1]} (P+d_x+d_z={expected_dim})')


# =====================================================================
# T2: Per-patch exog mean
# =====================================================================

def test_exog_mean():
    B, M, L, dx = 2, 4, 336, 3
    P, S = 16, 8
    N = (L - P) // S + 1

    exog = np.random.randn(B, M, L, dx).astype(np.float32)

    # Per-patch mean
    out = np.zeros((B, M, N, dx), dtype=np.float32)
    for i in range(N):
        out[:, :, i, :] = exog[:, :, i*S:i*S+P, :].mean(axis=2)

    # Spot check
    n = 5
    expected = exog[0, 1, n*S:n*S+P, :].mean(axis=0)
    assert np.allclose(out[0, 1, n, :], expected, atol=1e-6)

    print(f'T2 passed: exog patch mean shape = {out.shape}')


# =====================================================================
# T3: Cross-market attention shape transform
#   x: (B, M, N, D) -> reshape to (B*N, M, D), do MHA across M, reshape back
# =====================================================================

def test_cross_market_reshape():
    B, M, N, D = 2, 4, 41, 128
    H = 8
    dh = D // H

    x = np.random.randn(B, M, N, D).astype(np.float32)

    # 1. permute (B, M, N, D) -> (B, N, M, D)
    h = np.transpose(x, (0, 2, 1, 3))                         # (B, N, M, D)
    # 2. reshape (B*N, M, D)
    h2 = h.reshape(B * N, M, D)
    # 3. inverse: (B*N, M, D) -> (B, N, M, D) -> (B, M, N, D)
    back = h2.reshape(B, N, M, D).transpose(0, 2, 1, 3)
    assert np.array_equal(back, x), 'cross-market reshape inverse failed'

    # Now do a synthetic attention on h2 over the M axis
    Wq = np.random.randn(D, D).astype(np.float32) * 0.02
    Wk = np.random.randn(D, D).astype(np.float32) * 0.02
    Wv = np.random.randn(D, D).astype(np.float32) * 0.02

    q = (h2 @ Wq).reshape(B*N, M, H, dh).transpose(0, 2, 1, 3)   # (B*N, H, M, dh)
    k = (h2 @ Wk).reshape(B*N, M, H, dh).transpose(0, 2, 1, 3)
    v = (h2 @ Wv).reshape(B*N, M, H, dh).transpose(0, 2, 1, 3)

    logits = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)        # (B*N, H, M, M)
    attn = softmax(logits, axis=-1)
    assert attn.shape == (B*N, H, M, M)

    # Each row should sum to 1
    row_sums = attn.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), f'row sums: {row_sums}'

    # Reshape attn to (B, N, H, M, M) for MSIM aggregation
    attn_for_dump = attn.reshape(B, N, H, M, M)
    # Aggregating across batch, patches, heads gives a (M, M) matrix
    msim = attn_for_dump.mean(axis=(0, 1, 2))
    assert msim.shape == (M, M)
    # Diagonal usually highest because q close to k for self
    print(f'T3 passed: attn shape = {attn.shape}, msim diag = '
          f'{np.diag(msim).round(3).tolist()}')


# =====================================================================
# T4: Regime-aware MoE mixture math
# =====================================================================

def test_regime_moe():
    B, M, D = 2, 4, 128
    H, Q, K = 24, 3, 3

    h = np.random.randn(B, M, D).astype(np.float32)

    # Three experts produce (B, M, H, Q) each
    # Synthetic: each expert is a linear projection
    Ws = [np.random.randn(D, H * Q).astype(np.float32) * 0.01 for _ in range(K)]
    experts = np.stack([(h @ W).reshape(B, M, H, Q) for W in Ws], axis=2)
    assert experts.shape == (B, M, K, H, Q)

    # Gate
    Wg = np.random.randn(D, K).astype(np.float32) * 0.01
    gate_logits = h @ Wg                                       # (B, M, K)
    gate = softmax(gate_logits, axis=-1)
    assert gate.shape == (B, M, K)
    assert np.allclose(gate.sum(axis=-1), 1.0, atol=1e-5)

    # Mixture
    preds = (gate[..., None, None] * experts).sum(axis=2)      # (B, M, H, Q)
    assert preds.shape == (B, M, H, Q)

    # Sanity: if gate is one-hot at expert 0, preds should equal experts[:,:,0]
    one_hot = np.zeros_like(gate)
    one_hot[..., 0] = 1.0
    preds_oh = (one_hot[..., None, None] * experts).sum(axis=2)
    assert np.allclose(preds_oh, experts[:, :, 0])

    print(f'T4 passed: MoE preds.shape = {preds.shape}, '
          f'gate sums OK, one-hot reduction OK')


# =====================================================================
# T5: Coupling-prior gating sigmoid math
# =====================================================================

def test_coupling_prior_gate():
    M = 4
    a = np.zeros((M, M))                                       # learnable, init 0
    beta = 1.0
    # NTC prior, normalized to [0, 1]; off-diagonal varies
    prior = np.array([
        [1.00, 0.80, 0.70, 0.50],
        [0.80, 1.00, 0.40, 0.65],
        [0.70, 0.40, 1.00, 0.85],
        [0.50, 0.65, 0.85, 1.00],
    ])
    g = sigmoid(a + beta * prior)
    # Diagonal should be ~ sigmoid(1) ≈ 0.731
    assert np.allclose(np.diag(g), sigmoid(1.0), atol=1e-6)
    # Higher prior -> higher gate
    assert g[0, 1] > g[0, 3]
    assert g[2, 3] > g[2, 1]
    # Apply log-gate as additive bias to attention logits
    log_g = np.log(g + 1e-6)
    # Test that softmax with this bias prefers the high-prior column
    logits = np.zeros((M, M))
    biased = logits + log_g
    attn = softmax(biased, axis=-1)
    # For row 0: column 1 (prior 0.8) should beat column 3 (prior 0.5)
    assert attn[0, 1] > attn[0, 3]
    print(f'T5 passed: gate diag={np.diag(g).round(3).tolist()}, '
          f'attn row 0 = {attn[0].round(3).tolist()}')


# =====================================================================
# T6: Pool attention math
# =====================================================================

def test_pool_attention():
    B, M, N, D = 2, 4, 41, 128
    tokens = np.random.randn(B, M, N, D).astype(np.float32) * 0.1

    # Learnable query: (1, 1, 1, D)
    q = np.random.randn(1, 1, 1, D).astype(np.float32) * 0.02
    q_b = np.broadcast_to(q, (B, M, 1, D))

    # Concat for attention: (B*M, 1+N, D)
    seq = np.concatenate([q_b, tokens], axis=2)
    seq2 = seq.reshape(B * M, 1 + N, D)

    q_ = seq2[:, :1, :]                                        # (B*M, 1, D)
    kv = seq2[:, 1:, :]                                        # (B*M, N, D)

    scores = q_ @ kv.transpose(0, 2, 1) / np.sqrt(D)           # (B*M, 1, N)
    attn = softmax(scores, axis=-1)
    assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-5)

    pooled = (attn @ kv).squeeze(1)                            # (B*M, D)
    pooled = pooled.reshape(B, M, D)
    assert pooled.shape == (B, M, D)
    print(f'T6 passed: pooled.shape = {pooled.shape}')


# =====================================================================
# Run all
# =====================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('CMA-Former numpy sanity tests')
    print('=' * 60)
    test_patch_unfold()
    test_exog_mean()
    test_cross_market_reshape()
    test_regime_moe()
    test_coupling_prior_gate()
    test_pool_attention()
    print('=' * 60)
    print('ALL TESTS PASSED')
    print('=' * 60)
