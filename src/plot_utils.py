"""
Module for plotting yield curves and processing simulation database outputs.

This module contains functions to:
    - Plot yield curves based on CSV files for various ecozones and species groups.
    - Process simulation databases to compile yield-related statistics and generate plots.

It relies on configuration parameters (via a `cfg` object) and external plotting/utility
functions imported from util_functions.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from sqlite3 import connect
from util_functions import (
    plot_fri,
    plot_histogram,
    plot_cpool_years,
    plot_admin_eco_perhc,
    plot_fri_area_tec_line,
    Config,
)


def plot_yield_curves(cfg) -> None:
    """
    Plot yield curves for different species groups and ecozones based on CSV data.

    This function reads yield curve CSV files from a specified data directory,
    filters data for two specified ecozones, assigns species groups using mappings
    provided in the configuration, and then plots yield curves for each species group
    in each ecozone. The resulting plots are saved as PNG files in the plot directory.

    Parameters:
        cfg: Configuration object containing attributes such as:
            - data_dir: Base directory for data.
            - yield_perc_names: List of yield percentage names used in CSV filenames.
            - species_sci_name_dict: Mapping from species abbreviation to scientific name.
            - species_group_dict: Mapping from scientific name to species group.
            - plot_dir: Directory where plots will be saved.
    """
    # Directory containing yield curves CSV files
    yield_curves_path = os.path.join(cfg.data_dir, "yield_curves")
    yield_dict = {}
    ecozones = ["Taiga Plains", "Taiga Shield West"]
    age = np.arange(0, 101, 5)
    age_str = [str(a) for a in age]

    # Process each yield percentage CSV file
    for y_name in cfg.yield_perc_names:
        yield_dict[y_name] = {}

        csv_file = os.path.join(yield_curves_path, f"yield_{y_name}.csv")
        y_df = pd.read_csv(csv_file)
        # Filter rows that belong to the specified ecozones and reset index
        y_df = y_df.loc[y_df["eco_boundary"].isin(ecozones)].reset_index(drop=True)
        # Assign species group using mapping from the configuration
        y_df["group"] = None
        for idx in range(len(y_df)):
            sci_name = y_df.loc[idx, "LdSpp"]
            species_sci = cfg.species_sci_name_dict[sci_name]
            y_df.loc[idx, "group"] = cfg.species_group_dict[species_sci]

        # Extract yield values for each species group and ecozone
        for sp_group in range(1, 8):
            yield_dict[y_name][sp_group] = {}
            for eco in ecozones:
                subset_df = y_df.loc[(y_df["group"] == sp_group) & (y_df["eco_boundary"] == eco)].reset_index(drop=True)
                if not subset_df.empty:
                    row = subset_df.loc[0]
                    yields = list(row[age_str])
                    yield_dict[y_name][sp_group][eco] = yields

    # Generate and save yield curve plots
    for sp_group in range(1, 8):
        for eco in ecozones:
            plt.figure()
            plt.plot(age, yield_dict[cfg.yield_perc_names[0]][sp_group][eco], c="r", label="+40% MAT, -40% PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[1]][sp_group][eco], c="y", label="+20% MAT, -20% PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[2]][sp_group][eco], c="b", label="Mean MAT, PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[3]][sp_group][eco], c="c", label="-20% MAT, +20% PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[4]][sp_group][eco], c="g", label="-40% MAT, +40% PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[5]][sp_group][eco], c="m", label="-40% MAT, -40% PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[6]][sp_group][eco], c="orange", label="-20% MAT, -20% PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[7]][sp_group][eco], c="purple", label="+20% MAT, +20% PCP")
            plt.plot(age, yield_dict[cfg.yield_perc_names[8]][sp_group][eco], c="brown", label="+40% MAT, +40% PCP")
            plt.legend()
            plt.xlabel("Age (Years)")
            plt.ylabel("Yield Volume ($m^3$/ha)")
            # Optionally, add a title:
            # plt.title(f"Yield Curves for Species Group {sp_group} in {eco}")
            save_path = os.path.join(cfg.plot_dir, f"yc_{sp_group}_{eco.replace(' ', '')}.png")
            plt.savefig(save_path)
            plt.close()


def plot_db(cfg, mode: str) -> None:
    """
    Process simulation databases to compile yield data and generate plots.

    In "compute" mode, the function reads multiple simulation experiment databases,
    extracts relevant indicators, compiles yield data into a dictionary, and saves the
    results to a pickle file. In "plot" mode, it loads the precomputed data.
    Afterward, various plotting functions are called to generate visualizations.

    Parameters:
        cfg: Configuration object containing attributes such as:
            - exps_dir, store_dir, plot_dir: Directories for experiments, storage, and plots.
            - mc_iters: Number of Monte Carlo iterations.
            - sim_interval: Tuple indicating the start and end simulation years.
            - exp_name: Name of the experiment.
        mode (str): Operation mode; should be either "compute" to process the databases or "plot" to load precomputed data.
    """
    if mode == "compute":
        # Initialize yield dictionary using keys for clarity
        exp_input_path = os.path.join(cfg.exps_dir, f"{1}_exp")
        exp_cfg_path = os.path.join(exp_input_path, "exp_cfg.pkl")
        with open(exp_cfg_path, "rb") as file:
            exp_config = pickle.load(file)
        y_name = exp_config.yield_perc

        keys = [
            "area",
            "fri",
            "c_pool_final",
            "c_pool_years",
            "historic_age",
            "historic_lc",
            "historic_sp",
            "soil_type",
            "admin",
            "eco",
            "aff_sp",
            "perc_free",
            "dom_soil_p",
            "fire_frac",
            "perc_forest",
        ]
        yield_dict = {y_name: {key: [] for key in keys}}

        count = 0
        # Loop over Monte Carlo iterations
        for i in range(1, cfg.mc_iters + 1):
            exp_store_path = os.path.join(cfg.store_dir, f"{i}_exp")
            db_path = os.path.join(exp_store_path, "compiled_gcbm_output.db")

            if not os.path.exists(db_path):
                continue

            try:
                conn = connect(db_path)
                area_age_df = pd.read_sql('SELECT * FROM v_age_indicators', conn)
                pool_df = pd.read_sql('SELECT * FROM v_pool_indicators', conn)
                # disturbances_df is read but not used; kept for functionality parity
                disturbances_df = pd.read_sql('SELECT * FROM v_disturbance_indicators', conn)
            except Exception:
                continue

            try:
                area = area_age_df.loc[0, "area"]
            except Exception:
                continue

            count += 1
            # Load experiment configuration for current iteration
            exp_input_path = os.path.join(cfg.exps_dir, f"{i}_exp")
            exp_cfg_path = os.path.join(exp_input_path, "exp_cfg.pkl")
            with open(exp_cfg_path, "rb") as file:
                exp_config = pickle.load(file)

            # Read shapefiles for inventory and disturbances
            exp_inventory_shp = gpd.read_file(os.path.join(exp_input_path, "inventory", "inventory.shp"))
            exp_aff_shp = gpd.read_file(os.path.join(exp_input_path, "disturbances", "afforestation.shp"))

            try:
                exp_fire_shp = gpd.read_file(os.path.join(exp_input_path, "disturbances", "fire.shp"))
                if len(exp_fire_shp) == 0:
                    fri = 5000
                    fire_frac = 0
                else:
                    fri = float(exp_fire_shp.loc[0, "fri"])
                    fire_frac = exp_fire_shp.loc[0, "fire_frac"]
            except Exception:
                fri = 5000
                fire_frac = 0

            # Compute change in total ecosystem carbon pool over simulation interval
            init_pool = pool_df.loc[
                (pool_df["indicator"] == "Total Ecosystem") & (pool_df["year"] == 0)
                ]["pool_tc"].sum()
            c_pool_list = []
            te_pool_df = pool_df.loc[pool_df["indicator"] == "Total Ecosystem"]
            for year in range(cfg.sim_interval[0], cfg.sim_interval[1] + 1):
                te_pool = te_pool_df.loc[te_pool_df["year"] == year]["pool_tc"].sum() - init_pool
                c_pool_list.append(te_pool)

            # Append computed values to yield_dict
            yield_dict[y_name]["area"].append(area)
            yield_dict[y_name]["fri"].append(fri)
            yield_dict[y_name]["c_pool_final"].append(c_pool_list[-1])
            yield_dict[y_name]["c_pool_years"].append(c_pool_list)
            yield_dict[y_name]["historic_age"].append(exp_inventory_shp.loc[0, "age"])
            yield_dict[y_name]["historic_lc"].append(exp_inventory_shp.loc[0, "land_class"])
            yield_dict[y_name]["historic_sp"].append(exp_inventory_shp.loc[0, "LdSpp"])
            yield_dict[y_name]["soil_type"].append(exp_inventory_shp.loc[0, "soil_type"])
            yield_dict[y_name]["admin"].append(exp_inventory_shp.loc[0, "admin"])
            yield_dict[y_name]["eco"].append(exp_inventory_shp.loc[0, "eco"])
            yield_dict[y_name]["perc_forest"].append(exp_inventory_shp.loc[0, "perc_fores"])

            if "_bl_" in cfg.exp_name:
                yield_dict[y_name]["aff_sp"].append("NA")
                yield_dict[y_name]["perc_free"].append(0)
            else:
                yield_dict[y_name]["aff_sp"].append(exp_aff_shp.loc[0, "LdSpp"])
                yield_dict[y_name]["perc_free"].append(exp_aff_shp.loc[0, "perc_free"])

            yield_dict[y_name]["dom_soil_p"].append(exp_inventory_shp.loc[0, "dom_soil_p"])
            yield_dict[y_name]["fire_frac"].append(fire_frac)

        # Save the compiled yield data to a pickle file
        output_pickle = os.path.join(cfg.plot_dir, "yield_dict.pkl")
        with open(output_pickle, "wb") as f:
            pickle.dump(yield_dict, f)

        print(f"Plotting results for {count} experiments.")
    elif mode == "plot":
        # Load precomputed yield data from pickle file
        with open(os.path.join(cfg.plot_dir, "yield_dict.pkl"), "rb") as f:
            yield_dict = pickle.load(f)
    else:
        raise ValueError("Invalid mode. Use 'compute' or 'plot'.")

    # Generate plots using external utility functions
    plot_fri(cfg, yield_dict, smooth=True, spread=False)
    plot_histogram(cfg, yield_dict, var="area", xlabel="Area (Hectares)", title_add="Area")
    plot_histogram(cfg, yield_dict, var="fri", xlabel="Fire Return Interval (FRI)", title_add="FRI")
    plot_histogram(cfg, yield_dict, var="fire_frac", xlabel="Fire Fraction", title_add="Fire Fraction")
    plot_cpool_years(cfg, yield_dict, mode="combine")
    plot_admin_eco_perhc(cfg, yield_dict)
    plot_fri_area_tec_line(cfg, yield_dict, mode="compute", smooth=True, type="combine")
    print("Plotting completed.")


def main() -> None:
    """
    Main function to execute the plotting routines based on configuration and experiment names.

    For each experiment name specified, the configuration is updated, required directories are created,
    and the database plotting routine can be executed (currently commented out).
    """
    cfg = Config()

    exp_names = ["16_fire_bf"]
    main_plot_dir = cfg.plot_dir

    for exp_name in exp_names:
        print(f"Running experiment: {exp_name}")

        cfg.exp_name = exp_name
        cfg.exps_dir = os.path.join(cfg.exps_bck_dir, cfg.exp_name)
        cfg.store_dir = os.path.join(cfg.store_bck_dir, cfg.exp_name)
        cfg.plot_dir = os.path.join(main_plot_dir, cfg.exp_name)

        if not os.path.exists(cfg.plot_dir):
            os.makedirs(cfg.plot_dir)

        # Uncomment the following line to compute and plot database outputs:
        # plot_db(cfg, mode="compute")


if __name__ == "__main__":
    main()
