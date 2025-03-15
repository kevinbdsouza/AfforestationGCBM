"""
This script performs the following steps:
    - Clears the output folder of any existing files or directories.
    - Loads an experiment configuration from a pickle file.
    - Copies specified experiment directories into the raw layers folder.
    - Copies the appropriate yield file based on the experiment configuration.
"""

import os
import shutil
import pickle
from argparse import ArgumentParser
from config import Config


def parse_arguments() -> str:
    """
    Parse and return the experiment number from command-line arguments.

    Returns:
        str: The experiment number provided via the --exp_num argument.
    """
    parser = ArgumentParser(description="Preparing for new scenario")
    parser.add_argument("--exp_num", required=True, help="Experiment number")
    args = parser.parse_args()
    return args.exp_num


def clear_directory_contents(directory: str) -> None:
    """
    Remove all files and subdirectories within the specified directory.

    Args:
        directory (str): The directory whose contents will be cleared.
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def load_experiment_config(exp_dir: str) -> object:
    """
    Load the experiment configuration from a pickle file in the experiment directory.

    Args:
        exp_dir (str): The directory containing the experiment configuration file.

    Returns:
        object: The experiment configuration object loaded from the pickle file.
    """
    config_path = os.path.join(exp_dir, "exp_cfg.pkl")
    with open(config_path, "rb") as cfg_file:
        exp_cfg = pickle.load(cfg_file)
    return exp_cfg


def copy_experiment_directories(exp_dir: str, raw_dir: str, dir_list: list[str]) -> None:
    """
    Copy each specified directory from the experiment folder to the raw folder,
    removing any existing destination directories.

    Args:
        exp_dir (str): The experiment directory where the source folders reside.
        raw_dir (str): The destination raw directory.
        dir_list (list[str]): A list of directory names to copy.
    """
    for directory in dir_list:
        src_dir = os.path.join(exp_dir, directory)
        dst_dir = os.path.join(raw_dir, directory)

        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)

        shutil.copytree(src_dir, dst_dir)


def copy_yield_file(exp_cfg: object, data_dir: str, input_db_dir: str) -> None:
    """
    Copy the yield file corresponding to the experiment configuration to the input database.

    Args:
        exp_cfg (object): The experiment configuration object that contains the yield_perc attribute.
        data_dir (str): The directory where yield curves are stored.
        input_db_dir (str): The destination directory for the yield file.
    """
    yield_perc = exp_cfg.yield_perc
    src_yield_file = os.path.join(data_dir, "yield_curves", f"yield_{yield_perc}.csv")
    dst_yield_file = os.path.join(input_db_dir, "yield.csv")
    shutil.copy(src_yield_file, dst_yield_file)


def main() -> None:
    """
    Main function to prepare a new scenario:
        - Parse command-line arguments.
        - Clear the GCBM project output directory.
        - Load the experiment configuration.
        - Copy the required directories.
        - Copy the yield file.
    """
    exp_num = parse_arguments()

    print(f"Running scenario: {exp_num}")

    # Clear the GCBM project output directory.
    gcbm_output_dir = os.path.join(".", "gcbm_project", "output")
    clear_directory_contents(gcbm_output_dir)

    # Define base directories.
    data_dir = os.path.join(".", "data")
    input_db_dir = os.path.join(".", "input_database")
    raw_dir = os.path.join(".", "layers", "raw")
    exps_dir = os.path.join(raw_dir, "exps")

    # Construct the experiment directory path.
    exp_dir = os.path.join(exps_dir, f"{exp_num}_exp")

    # Load the experiment configuration.
    exp_cfg = load_experiment_config(exp_dir)

    # Copy specified directories from the experiment folder to the raw folder.
    copy_experiment_directories(exp_dir, raw_dir, ["inventory", "disturbances"])

    # Copy the appropriate yield file.
    copy_yield_file(exp_cfg, data_dir, input_db_dir)


if __name__ == "__main__":
    main()
