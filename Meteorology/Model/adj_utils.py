import numpy as np
import scipy.sparse as sp
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics.pairwise import cosine_similarity


def construct_adj(data, interval=24):
    """Cosine similarity matrix from training data (hourly: interval=24)."""
    data_mean = np.mean(
        [data[interval * i: interval * (i + 1)] for i in range(data.shape[0] // interval)],
        axis=0)
    data_mean = data_mean.squeeze().T  # (N, interval)
    sim = cosine_similarity(data_mean, data_mean)
    sim = np.exp((sim - sim.mean()) / (sim.std() + 1e-8))
    return sim.astype(np.float32)


def convert_to_undirected(adj_matrix):
    binary = np.where(adj_matrix != 0, 1, 0)
    symmetric = binary + binary.T
    symmetric = np.where(symmetric > 1, 1, symmetric)
    np.fill_diagonal(symmetric, 1)
    return symmetric.astype(np.float32)


def add_virtual_edges(adj_matrix, sim_matrix, q=80):
    """Vectorized: add edges where sim >= q-th percentile and no existing edge."""
    threshold = np.percentile(sim_matrix, q)
    virtual = (sim_matrix >= threshold) & (adj_matrix == 0)
    new_adj = adj_matrix.copy()
    new_adj[virtual] = 1
    new_adj[virtual.T] = 1
    return new_adj.astype(np.float32)


def calculate_symmetric_message_passing_adj(adj):
    """D^{-1/2} (A+I) D^{-1/2} normalization."""
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))
    adj_sp = sp.coo_matrix(adj)
    row_sum = np.array(adj_sp.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat = sp.diags(d_inv_sqrt)
    mp_adj = d_mat.dot(adj_sp).transpose().dot(d_mat).astype(np.float32)
    return mp_adj


def build_meteorology_adj(train_data, n_components=64, q=80, interval=24):
    """
    Build node embeddings from training data:
      1. Cosine similarity from training data
      2. Identity matrix as static base
      3. Add virtual edges (cosine sim >= q-th percentile)
      4. GCN symmetric normalization
      5. LLE dimensionality reduction -> (N, n_components)

    train_data: np.ndarray (T, N) or (T, N, F) — uses first feature if 3D
    Returns: np.ndarray (N, n_components), float32
    """
    if train_data.ndim == 3:
        train_data = train_data[..., 0]  # (T, N)

    N = train_data.shape[1]
    print(f"[adj] computing cosine similarity for {N} nodes ...")
    cos = construct_adj(train_data, interval=interval)

    # identity as static base, then add virtual edges
    identity = np.eye(N, dtype=np.float32)
    adj = add_virtual_edges(identity, cos, q=q)

    print(f"[adj] GCN normalization ...")
    adj_norm = calculate_symmetric_message_passing_adj(adj)
    adj_norm = np.asarray(adj_norm.todense()).astype(np.float32)

    print(f"[adj] LLE -> {n_components} dims ...")
    lle = LocallyLinearEmbedding(n_neighbors=n_components+100, n_components=n_components, n_jobs=-1)
    adj_emb = lle.fit_transform(adj_norm)

    return adj_emb.astype(np.float32)
