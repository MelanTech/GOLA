# Created by Zekai Shao
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0

import argparse
import os
from pathlib import Path

import safetensors.torch
import torch
from k_means_constrained import KMeansConstrained
from tqdm import tqdm


def rank_matrix_rows_by_svd(matrix, n_components=None):
    """
    Rank the rows of a matrix by their importance based on SVD
    :param matrix: numpy array of shape (n_rows, n_features)
    :param n_components: number of components to consider for ranking
    :return: sorted_indices: numpy array of shape (n_rows,)
    """
    # Centering the matrix
    mean = torch.mean(matrix, dim=0)
    centered_matrix = matrix - mean

    U, S, Vt = torch.linalg.svd(centered_matrix)

    # Select the number of components
    if n_components is None:
        n_components = S.shape[0]
    else:
        n_components = min(n_components, S.shape[0])

    # Compute the projections of each row on the principal components
    # Select the top n_components singular values and corresponding right singular vectors
    components = Vt[:n_components, :]

    # Calculate the importance of each row in the projections
    # Importance = projection length × weight of corresponding singular value
    projections = torch.matmul(centered_matrix, components.t())
    weighted_projections = projections * S[:n_components].unsqueeze(0)

    # Compute the importance score for each row (square root of the sum of squares)
    row_importance = torch.norm(weighted_projections, p=2, dim=1)

    # Sort the rows by importance score
    sorted_indices = torch.argsort(row_importance, descending=True)

    return sorted_indices


def kmeans_cluster(matrix, n_groups):
    """
    K-means clustering for matrix rows
    :param matrix: numpy array of shape (n_rows, n_features)
    :param n_groups: number of clusters
    :return: labels: numpy array of shape (n_rows,)
    """
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.cpu().numpy()
    else:
        matrix_np = matrix

    size = matrix_np.shape[1] // n_groups

    kmc = KMeansConstrained(n_groups, size_min=size, size_max=size, random_state=42)
    return kmc.fit_predict(matrix_np.T)


def split_params(matrix, labels, n_groups, transpose=False):
    """
    Split a matrix into n_groups sub-matrices according to labels
    :param matrix: numpy array of shape (n_rows, n_features)
    :param labels: numpy array of shape (n_rows,)
    :param n_groups: number of sub-matrices
    :param transpose: whether to transpose the sub-matrices
    :return: list of numpy arrays of shape (n_rows, n_features/n_groups)
    """
    param_group = []
    for i in range(n_groups):
        if transpose:
            param_group.append(matrix[labels == i, :])
        else:
            param_group.append(matrix[:, labels == i])

    return param_group


def grouping_params(A, B, n_retain, n_groups):
    """
    Group parameters A and B
    :param A: numpy array of shape (n_rows, n_features)
    :param B: numpy array of shape (n_features, n_columns)
    :param n_retain: number of retained parameters
    :param n_groups: number of groups
    :return: A_retained, B_retained, A_grouped, B_grouped
    """
    rank_indices = rank_matrix_rows_by_svd(B.T, n_retain)

    A_sorted = A[rank_indices, :]
    A_retained = A_sorted[:n_retain, :]
    B_sorted = B[:, rank_indices]
    B_retained = B_sorted[:, :n_retain]

    A_group = A_sorted[n_retain:, :]
    B_group = B_sorted[:, n_retain:]

    groups = kmeans_cluster(B_group, n_groups)

    A_grouped = split_params(A_group, groups, n_groups, transpose=True)
    B_grouped = split_params(B_group, groups, n_groups)

    return A_retained, B_retained, A_grouped, B_grouped


def check_params(A, B, A_retained, B_retained, A_grouped, B_grouped, n_groups, thresh=1e-5):
    """
    Check if the parameters are correctly grouped
    :param A: numpy array of shape (n_rows, n_features)
    :param B: numpy array of shape (n_features, n_columns)
    :param A_retained: numpy array of shape (n_retain, n_features)
    :param B_retained: numpy array of shape (n_features, n_retain)
    :param A_grouped: list of numpy arrays of shape (n_rows, n_features/n_groups)
    :param B_grouped: list of numpy arrays of shape (n_features, n_columns/n_groups)
    :param n_groups: number of groups
    :param thresh: threshold for checking the parameters
    :return: None
    """
    W = B @ A
    W_retained = B_retained @ A_retained
    W_grouped = None
    for i in range(n_groups):
        w_group = B_grouped[i] @ A_grouped[i]
        if W_grouped is None:
            W_grouped = w_group
        else:
            W_grouped = W_grouped + w_group
    W_recon = W_retained + W_grouped
    assert (W - W_recon).abs().mean() < thresh, "Parameters error is too large"


def duplicate_token_type_embed(weights):
    """
    Duplicate token type embed for online template
    :param weights: dict of weights
    :return: weights
    """

    hidden_dim = weights['blocks.0.attn.proj.lora.A'].shape[-1]
    embed = weights['token_type_embed']
    new_embed = torch.zeros((5, hidden_dim))
    new_embed[:2] = embed[:2]
    new_embed[2:4] = embed[:2]
    new_embed[4:] = embed[2:]

    del weights['token_type_embed']
    weights['token_type_embed'] = new_embed
    return weights


def generate_target_path(source_path, n_retain, n_groups):
    """
    Generate target file path based on source path and parameters
    Format: base_gola_r{retain}_g{groups}.bin
    """
    source_path = Path(source_path)
    parent_dir = source_path.parent
    filename = source_path.stem
    ext = source_path.suffix

    new_filename = f"{filename}_gola_r{n_retain}_g{n_groups}{ext}"

    target_path = parent_dir / new_filename
    return str(target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Group LoRA parameters with GOLA method')
    parser.add_argument('--weight', type=str, required=True, help='Path to source weights file')
    parser.add_argument('--n_retain', type=int, required=True, help='Number of retained parameters')
    parser.add_argument('--n_groups', type=int, required=True, help='Number of groups for clustering')
    args = parser.parse_args()

    N_RETAIN = args.n_retain
    N_GROUPS = args.n_groups
    source_weight_path = args.weight

    target_weight_path = generate_target_path(source_weight_path, N_RETAIN, N_GROUPS)

    print(f"Loading weights from: {source_weight_path}")
    print(f"Parameters: Retain={N_RETAIN}, Groups={N_GROUPS}")
    print(f"Will save processed weights to: {target_weight_path}")

    weight = safetensors.torch.load_file(source_weight_path)
    weight = duplicate_token_type_embed(weight)

    new_weight = {}

    for k, v in tqdm(weight.items(), desc="Processing weights"):
        if "lora.A" in k:
            prefix = k[:-len("lora.A")]
            A = v
            B = weight[prefix + "lora.B"]
            A_retained, B_retained, A_grouped, B_grouped = grouping_params(A, B, N_RETAIN, N_GROUPS)

            check_params(A, B, A_retained, B_retained, A_grouped, B_grouped, N_GROUPS)

            new_weight[prefix + "lora.A"] = A_retained.contiguous()
            new_weight[prefix + "lora.B"] = B_retained.contiguous()
            for i in range(N_GROUPS):
                new_weight[prefix + "lora.GA" f".{i}"] = A_grouped[i].contiguous()
                new_weight[prefix + "lora.GB" + f".{i}"] = B_grouped[i].contiguous()

        elif "lora.B" in k:
            pass
        else:
            new_weight[k] = v

    safetensors.torch.save_file(new_weight, target_weight_path)
    print(f"Successfully saved processed weights to: {target_weight_path}")