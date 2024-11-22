import numpy as np
import os
import sys
import argparse


from utils_jh import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a heatmap from data.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the results txt file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    input_path = (
        args.input_path + "/Seen/"
    )  # to_visualize/"2024-11-05 12:36:24.943797"/
    file_name = "results.txt"
    file_path = os.path.join(input_path, file_name)

    with open(file_path, "r") as f:
        results = []
        for line in f:
            parts = line.strip().split()
            if parts == []:
                break
            # print(parts)
            parts = [parts[0], parts[2], parts[4], parts[6]]
            results.append(parts)

    results.sort(key=lambda x: float(x[1]))
    output_path = f"sorted_results/{os.path.split(args.input_path)[-1]}"
    os.makedirs(f"{output_path}", exist_ok=True)
    print(f"{output_path}")
    with open(f"{output_path}/results_sorted.txt", "w") as f:
        for line in results:
            f.write("\t".join(line) + "\n")
