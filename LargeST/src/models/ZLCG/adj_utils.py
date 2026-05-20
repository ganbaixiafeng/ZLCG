import os

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.dataloader import load_adj_from_numpy


def construct_adj(data):
    # construct the adj through the cosine similarity
    data_mean = np.mean([data[24 * 4 * i: 24 * 4 * (i + 1)] for i in range(data.shape[0] // (24 * 4))], axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / tem_matrix.std())
    return tem_matrix


def calculate_symmetric_message_passing_adj(adj: np.ndarray) -> np.matrix:
    """Calculate the renormalized message passing adj in `GCN`.
    A = A + I
    return D^{-1/2} A D^{-1/2}

    Args:
        adj (np.ndarray): Adjacent matrix A

    Returns:
        np.matrix: Renormalized message passing adj in `GCN`.
    """

    # add self loop
    adj = adj + np.diag(np.ones(adj.shape[0], dtype=np.float32))

    # print("calculating the renormalized message passing adj, please ensure that self-loop has added to adj.")
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    mp_adj = d_mat_inv_sqrt.dot(adj).transpose().dot(
        d_mat_inv_sqrt).astype(np.float32)
    return mp_adj


def convert_to_undirected(adj_matrix):
    assert adj_matrix.shape[0] == adj_matrix.shape[1]

    binary_matrix = np.where(adj_matrix != 0, 1, 0)
    symmetric_matrix = binary_matrix + binary_matrix.T
    symmetric_matrix = np.where(symmetric_matrix > 1, 1, symmetric_matrix)
    np.fill_diagonal(symmetric_matrix, 1)

    return symmetric_matrix


def add_virtual_edges(adj_matrix, sim_matrix, q=80):
    threshold = np.percentile(sim_matrix, q)
    n = adj_matrix.shape[0]
    new_adj_matrix = adj_matrix.copy()

    for i in range(n):
        non_connected = np.where(adj_matrix[i] == 0)[0]
        sim_values = sim_matrix[i, non_connected]
        sorted_indices = np.argsort(sim_values)[::-1]

        for j_idx in sorted_indices:
            j = non_connected[j_idx]
            if sim_values[j_idx] >= threshold:
                new_adj_matrix[i, j] = 1
                new_adj_matrix[j, i] = 1
    return new_adj_matrix


def build_cosine_enhanced_message_adj(adj_path, data_path, q=80):
    traffic = np.load(os.path.join(data_path, '2019', 'his.npz'))['data'][..., :1]
    idx_train = np.load(os.path.join(data_path, '2019', 'idx_train.npy'))

    train_step = idx_train[-1]
    train_data = traffic[:train_step]

    cos = construct_adj(train_data)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = convert_to_undirected(adj_mx)
    adj_mx = add_virtual_edges(adj_mx, cos, q=q)

    mes = calculate_symmetric_message_passing_adj(adj_mx)
    mes = mes.astype(np.float32).todense()

    return cos, mes


def build_separate_cosine_physical_message_adj(adj_path, data_path, q=80):
    traffic = np.load(os.path.join(data_path, '2019', 'his.npz'))['data'][..., :1]
    idx_train = np.load(os.path.join(data_path, '2019', 'idx_train.npy'))

    train_step = idx_train[-1]
    train_data = traffic[:train_step]

    cos = construct_adj(train_data)
    matrix = np.zeros((716, 716), dtype=np.float32)
    cos = add_virtual_edges(matrix, cos, q=q)

    cos = calculate_symmetric_message_passing_adj(cos)
    cos = cos.astype(np.float32).todense()

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = convert_to_undirected(adj_mx)

    mes = calculate_symmetric_message_passing_adj(adj_mx)
    mes = mes.astype(np.float32).todense()

    return cos, mes
