import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


PEMS_ROOT = Path(__file__).resolve().parents[2]


def construct_adj(data, steps_per_day=288):
    # construct the adj through the cosine similarity
    data_mean = np.mean(
        [data[steps_per_day * i: steps_per_day * (i + 1)] for i in range(data.shape[0] // steps_per_day)],
        axis=0)
    data_mean = data_mean.squeeze().T
    tem_matrix = cosine_similarity(data_mean, data_mean)
    tem_matrix = np.exp((tem_matrix - tem_matrix.mean()) / tem_matrix.std())
    return tem_matrix


def generate_cos_adj(args):
    data = np.load(args.data_file_path)["data"]
    data = data[..., args.target_channel]

    num_samples = data.shape[0] - (args.history_seq_len + args.future_seq_len) + 1
    train_num = round(num_samples * args.train_ratio)
    train_data = data[:train_num]

    adj = construct_adj(train_data, steps_per_day=args.steps_per_day)
    print(adj.shape)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "cos{0}.pkl".format(args.dataset_name))
    with open(output_path, "wb") as f:
        pickle.dump(adj, f)
    print("cosine adjacency saved to: {0}".format(output_path))


if __name__ == "__main__":
    DATASET_NAME = "PEMS04"
    OUTPUT_DIR = str(PEMS_ROOT / "datasets" / DATASET_NAME)
    DATA_FILE_PATH = str(PEMS_ROOT / "datasets" / "raw_data" / DATASET_NAME / "{0}.npz".format(DATASET_NAME))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str,
                        default=DATASET_NAME, help="Dataset name.")
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=12, help="History sequence length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=12, help="Future sequence length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=288, help="Steps per day.")
    parser.add_argument("--target_channel", type=int,
                        default=0, help="Selected target channel.")
    parser.add_argument("--train_ratio", type=float,
                        default=0.6, help="Train ratio.")
    args = parser.parse_args()

    generate_cos_adj(args)
