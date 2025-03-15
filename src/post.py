"""
This script copies processed output data from a source directory to a designated
store directory based on the provided experiment number. It also removes a compiled
database file if it exists in the source directory.
"""

import os
import shutil
from argparse import ArgumentParser
from config import Config


def parse_arguments() -> str:
    """
    Parse and return the experiment number from the command-line arguments.

    Returns:
        str: The experiment number provided via the --exp_num argument.
    """
    parser = ArgumentParser(description="Cleaning post and storing outputs")
    parser.add_argument("--exp_num", required=True, help="Experiment number")
    args = parser.parse_args()
    return args.exp_num


def store_outputs(exp_num: str) -> None:
    """
    Copy processed outputs to the store directory and remove the compiled database file if it exists.

    Args:
        exp_num (str): Experiment number used to form the destination directory name.
    """
    print(f"Storing outputs for scenario: {exp_num}")

    # Define source and destination directories
    src_proc_output_dir = os.path.join(".", "processed_output")
    dst_proc_output_dir = os.path.join(".", "store", f"{exp_num}_exp")

    # Copy the processed output directory to the destination directory
    shutil.copytree(src_proc_output_dir, dst_proc_output_dir)

    # Remove the compiled GCBM output database if it exists
    db_path = os.path.join(src_proc_output_dir, "compiled_gcbm_output.db")
    if os.path.exists(db_path):
        os.remove(db_path)


def main() -> None:
    """
    Main function to parse arguments, initialize configuration, and store outputs.
    """
    exp_num = parse_arguments()
    store_outputs(exp_num)


if __name__ == "__main__":
    main()
