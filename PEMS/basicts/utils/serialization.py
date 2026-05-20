import pickle

import torch
import numpy as np

from .adjacent_matrix_norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, \
    calculate_symmetric_message_passing_adj, calculate_transition_matrix


def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def dump_pkl(obj: object, file_path: str):
    """Dumplicate pickle data.

    Args:
        obj (object): object
        file_path (str): file path
    """

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_adj(file_path: str, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx

##根据余弦相似度添加虚拟边
def add_virtual_edges(adj_matrix, sim_matrix, q=80):
    threshold = np.percentile(sim_matrix, q)

    #基于余弦相似度添加虚拟边
    n = adj_matrix.shape[0]
    new_adj_matrix = adj_matrix.copy()

    # 对于每个节点对，如果它们之间没有边且相似度超过阈值，则添加边
    for i in range(n):
        # 获取当前节点没有连接的节点，并按相似度排序
        non_connected = np.where(adj_matrix[i] == 0)[0]
        sim_values = sim_matrix[i, non_connected]
        sorted_indices = np.argsort(sim_values)[::-1]  # 降序排列

        # 添加相似度最高的前k个节点
        for j_idx in sorted_indices:
            j = non_connected[j_idx]
            if sim_values[j_idx] >= threshold:
                new_adj_matrix[i, j] = 1
                new_adj_matrix[j, i] = 1  # 保持对称性

    return new_adj_matrix
def load_my_adj(file_path_adj: str,file_path_cos: str):

    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path_adj)
        _, _, cos_mx = load_pkl(file_path_cos)
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path_adj)
        cos_mx = load_pkl(file_path_cos)

    adj_mx = add_virtual_edges(adj_mx,cos_mx)

    adj = [calculate_symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]

    return adj, adj_mx

def load_virtual(node_num,file_path_cos: str):

    try:
        # METR and PEMS_BAY
        _, _, cos_mx = load_pkl(file_path_cos)
    except ValueError:
        # PEMS04
        cos_mx = load_pkl(file_path_cos)


    matrix = np.zeros((node_num, node_num), dtype=np.float32)
    adj_mx = add_virtual_edges(matrix,cos_mx)

    adj = [calculate_symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]

    return adj, adj_mx

def load_adj_np(file_path: str, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    adj_mx = np.load(file_path)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx

def load_node2vec_emb(file_path: str) -> torch.Tensor:
    """load node2vec embedding

    Args:
        file_path (str): file path

    Returns:
        torch.Tensor: node2vec embedding
    """

    # spatial embedding
    with open(file_path, mode="r") as f:
        lines = f.readlines()
        temp = lines[0].split(" ")
        num_vertex, dims = int(temp[0]), int(temp[1])
        spatial_embeddings = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(" ")
            index = int(temp[0])
            spatial_embeddings[index] = torch.Tensor([float(ch) for ch in temp[1:]])
    return spatial_embeddings
