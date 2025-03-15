"""
Module for processing spatial and land cover data.

This module contains functions to:
    - Create a combined dataframe from grid and shapefile data.
    - Create yield curves based on environmental data.
    - Compile grid information by aggregating data from combined dataframes and environmental grids.
    - Generate simulation scenarios (inventory, afforestation, fire, and optionally mortality).
    - Identify and print the top 10 most common species mixture combinations.

Each function expects a configuration object (cfg) with the required directory paths and parameters.
"""

import os
import pickle
import json
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import Polygon, shape

# Local modules (assumed to provide additional functions such as regrid_sample, etc.)
from util_functions import *
from scenario_utils import *
from exp_config import ExpConfig


def create_combined_df(cfg):
    """
    Create a combined dataframe from spatial and land cover data and save as pickle files.

    Reads bounds from a CSV and iterates over grid coordinates defined by latitudes and longitudes.
    For each grid cell, it checks intersection with tile polygons from shapefiles and aggregates
    properties (e.g., species data, forest age, fire information, ecozone, and admin region) into
    a combined dataframe which is then saved to a pickle file.

    Parameters:
        cfg: Configuration object with attributes:
             - transf_dir: Directory containing transformation files (e.g., bounds.csv and tile shapefiles).
             - land_cover_dir: Directory to save the combined land cover pickle files.
    """
    # Define latitude and longitude boundaries
    lats = np.round(np.linspace(56.75, 70.75, 232), 2)
    lons = np.round(np.linspace(-141.25, -90.75, 816), 2)

    # Read bounds CSV and drop unnecessary column
    bounds_path = os.path.join(cfg.transf_dir, "bounds.csv")
    bounds_df = pd.read_csv(bounds_path).drop(["Unnamed: 0"], axis=1)

    # Iterate over a specific latitude index range (here, 171 to 171 inclusive)
    for i in range(len(lats)):
        print(i)
        combined_df = pd.DataFrame(columns=[
            "lat_min", "lat_max", "lon_min", "lon_max", "species", "species_per",
            "forest_age", "fire", "fire_year", "eco", "admin", "per_free", "per_forest"
        ])

        for j in range(len(lons)):
            # Skip the last column index to avoid index error
            if j == len(lons) - 1 or i == len(lats) - 1:
                continue

            # Define grid cell polygon
            coords = [
                (lats[i], lons[j]),
                (lats[i], lons[j + 1]),
                (lats[i + 1], lons[j + 1]),
                (lats[i + 1], lons[j])
            ]
            poly = shape(Polygon(coords))

            # Iterate over each bound row using itertuples for clarity
            for bound in bounds_df.itertuples(index=False):
                tile_coords = [
                    (bound.lat_min, bound.lon_min),
                    (bound.lat_min, bound.lon_max),
                    (bound.lat_max, bound.lon_max),
                    (bound.lat_max, bound.lon_min)
                ]
                tile_poly = shape(Polygon(tile_coords))
                if not (tile_poly.intersects(poly) or tile_poly.contains(poly)):
                    continue

                tile_shp_path = os.path.join(cfg.transf_dir, f"tile_{int(bound.tile)}.shp")
                with fiona.open(tile_shp_path) as tile_shp:
                    for row in tile_shp:
                        row_poly = shape(row["geometry"])
                        if not (poly.intersects(row_poly) or poly.contains(row_poly)):
                            continue

                        try:
                            per_water = row["properties"]["LC_WATER"]
                        except Exception:
                            continue

                        per_snow = row["properties"]["LC_SNOW_IC"]
                        per_rock = row["properties"]["LC_ROCK_RU"]
                        per_forest = row["properties"]["LC_FAO_FOR"]

                        # Skip tiles with high percentages of water, snow, rock, or forest
                        if (per_water + per_snow + per_rock + per_forest) > 65:
                            continue

                        # Calculate free percentage based on various land cover types
                        per_exposed = row["properties"]["LC_EXPOSED"]
                        per_bryoids = row["properties"]["LC_BRYOIDS"]
                        per_shrubs = row["properties"]["LC_SHRUBS"]
                        per_herbs = row["properties"]["LC_HERBS"]
                        per_wetland = row["properties"]["LC_WETLA_2"]
                        per_free = per_exposed + per_bryoids + per_shrubs + per_wetland + per_herbs

                        ecozone = row["properties"]["ECOZONE"]
                        admin = row["properties"]["JURISDICTI"]
                        forest_age = row["properties"]["AGE_AVG"]

                        # Process species information
                        sp_fields = ["_CM", "__2", "__4", "__6", "__8"]
                        sp_per_fields = ["__1", "__3", "__5", "__7", "__9"]
                        species = []
                        species_per = []
                        for sp_i in range(5):
                            sp = row["properties"].get("SPECIES" + sp_fields[sp_i])
                            if sp is None:
                                break
                            species.append(sp)
                            species_per.append(float(row["properties"]["SPECIES" + sp_per_fields[sp_i]]))

                        # Process fire information
                        fire = 0
                        fire_year = 0
                        disturbance = row["properties"]["SYMB_DISTU"]
                        if disturbance is not None:
                            dist_split = disturbance.split("_")
                            if dist_split[0] == "Fire":
                                fire = 1
                                fire_year = int(dist_split[1])

                        new_row = pd.Series({
                            "lat_min": lats[i],
                            "lat_max": lats[i + 1],
                            "lon_min": lons[j],
                            "lon_max": lons[j + 1],
                            "species": species,
                            "species_per": species_per,
                            "forest_age": forest_age,
                            "fire": fire,
                            "fire_year": fire_year,
                            "eco": ecozone,
                            "admin": admin,
                            "per_free": per_free,
                            "per_forest": per_forest
                        }).to_frame().T

                        combined_df = pd.concat([combined_df, new_row], ignore_index=True)

        out_fp = os.path.join(cfg.land_cover_dir, f"combined_df{i}.pkl")
        combined_df.to_pickle(out_fp)


def create_yield_curves(cfg):
    """
    Create yield curves CSV files for different yield percentage configurations.

    Reads a compiled grid dataframe and, for each yield percentage configuration, calculates yield values
    based on coefficients and mean MAT/PCP values for each ecozone. The resulting yield curves for different
    species are saved as CSV files in a designated directory.

    Parameters:
        cfg: Configuration object with attributes:
             - land_cover_dir: Directory where the compiled grid dataframe is stored.
             - yield_percs_mat: List of material percentage multipliers.
             - yield_percs_pcp: List of precipitation percentage multipliers.
             - yield_perc_names: List of yield percentage names corresponding to the multipliers.
             - yield_coeffs: Coefficients for yield calculations.
             - species_group_dict: Dictionary mapping species to group.
             - species_cm_name_dict: Dictionary mapping species to common names.
             - data_dir: Directory to save yield curve CSV files.
    """
    compiled_fp = os.path.join(cfg.land_cover_dir, "compile_grid_df.pkl")
    compiled_df = pd.read_pickle(compiled_fp)
    ecozones = compiled_df["eco"].unique()
    time_steps_int = np.arange(0, 101, 5)
    time_steps = [str(t) for t in time_steps_int]

    for mat_perc, pcp_perc, perc_name in zip(cfg.yield_percs_mat, cfg.yield_percs_pcp, cfg.yield_perc_names):
        yield_df = pd.DataFrame(columns=["LdSpp", "eco_boundary", "AIDBSPP"] + time_steps)
        for eco in ecozones:
            subset_df = compiled_df.loc[compiled_df["eco"] == eco]
            mean_mat = subset_df["mat"].mean()
            mean_pcp = subset_df["pcp"].mean()

            mat = mean_mat + (mean_mat * mat_perc)
            pcp = mean_pcp + (mean_pcp * pcp_perc)

            yield_lists = []
            for sp_group in range(0, 8):
                yield_list = []
                yc = cfg.yield_coeffs[sp_group]
                for t_s in time_steps_int:
                    if t_s == 0:
                        yield_value = 0
                    else:
                        yield_value = np.round(
                            np.exp(
                                yc[0] + yc[1] * mat + yc[2] * pcp +
                                (yc[3] + yc[4] * mat + yc[5] * pcp) / t_s
                            ) * yc[6],
                            4
                        )
                    yield_list.append(yield_value)
                yield_lists.append(yield_list)

            for sp, group in cfg.species_group_dict.items():
                sp_cn = cfg.species_cm_name_dict[sp]
                sp_aidb = sp if group == 0 else sp_cn
                new_row = pd.Series({"LdSpp": sp_cn, "eco_boundary": eco, "AIDBSPP": sp_aidb})
                yield_list = yield_lists[group]
                for y, t_s in zip(yield_list, time_steps):
                    new_row[t_s] = y

                yield_df = pd.concat([yield_df, new_row.to_frame().T], ignore_index=True)

        out_csv = os.path.join(cfg.data_dir, "yield_curves", f"yield_{perc_name}.csv")
        yield_df.to_csv(out_csv, index=False)


def compile_grid_info(cfg):
    """
    Compile grid information by aggregating combined dataframes and environmental data.

    Iterates over grid coordinates defined in the configuration, aggregates data from corresponding
    combined dataframes, and computes statistics such as average forest age, fire fraction, and ecozone
    information. Also averages MAT and PCP values from grid numpy arrays. The compiled dataframe is
    then saved as a pickle file.

    Parameters:
        cfg: Configuration object with attributes:
             - lats: List/array of latitude grid boundaries.
             - lons: List/array of longitude grid boundaries.
             - land_cover_dir: Directory where combined dataframes are stored.
             - data_dir: Directory containing MAT and PCP numpy arrays.
    """
    compile_grid_df = pd.DataFrame(columns=[
        "lat_min", "lat_max", "lon_min", "lon_max", "species", "species_perc",
        "forest_age", "fire_fraction", "fire_year", "eco", "admin", "perc_free", "mat", "pcp"
    ])

    mat_np = np.load(os.path.join(cfg.data_dir, "mat", "mat_grid.npy"))
    pcp_np = np.load(os.path.join(cfg.data_dir, "total_pcp", "pcp_grid.npy"))

    for i in range(len(cfg.lats)):
        combined_fp = os.path.join(cfg.land_cover_dir, f"combined_df{i}.pkl")
        combined_df = pd.read_pickle(combined_fp)

        for j in range(len(cfg.lons)):
            if j == len(cfg.lons) - 1 or i == len(cfg.lats) - 1:
                continue

            subset_df = combined_df.loc[
                (combined_df["lat_min"] == cfg.lats[i]) &
                (combined_df["lat_max"] == cfg.lats[i + 1]) &
                (combined_df["lon_min"] == cfg.lons[j]) &
                (combined_df["lon_max"] == cfg.lons[j + 1])
            ].reset_index(drop=True)

            if subset_df.empty:
                continue

            species_dict = {}
            forest_age_list = []
            fire_list = []
            fire_year_list = []
            eco_list = []
            admin_list = []
            perc_free_list = []
            perc_forest_list = []
            full_free_count = 0

            for idx in range(len(subset_df)):
                if subset_df.loc[idx, "eco"] == "Southern Arctic":
                    continue

                if subset_df.loc[idx, "per_free"] != 100.0:
                    for sp_dx, sp in enumerate(subset_df.loc[idx, "species"]):
                        species_dict.setdefault(sp, []).append(subset_df.loc[idx, "species_per"][sp_dx])
                else:
                    full_free_count += 1

                forest_age_list.append(subset_df.loc[idx, "forest_age"])
                fire_list.append(subset_df.loc[idx, "fire"])
                fire_year_list.append(subset_df.loc[idx, "fire_year"])
                eco_list.append(subset_df.loc[idx, "eco"])
                admin_list.append(subset_df.loc[idx, "admin"])
                perc_free_list.append(subset_df.loc[idx, "per_free"])
                perc_forest_list.append(subset_df.loc[idx, "per_forest"])

            species = []
            species_perc = []
            for sp, sp_per in species_dict.items():
                species.append(sp)
                denominator = (len(subset_df) - full_free_count) if (len(subset_df) - full_free_count) else 1
                species_perc.append(np.sum(sp_per) / denominator)

            avg_forest_age = (np.sum(forest_age_list) / (len(subset_df) - full_free_count)
                              if (len(subset_df) - full_free_count) else 0)
            fire_fraction = np.sum(fire_list) / len(subset_df)
            most_common_fire_year = Counter(fire_year_list).most_common(1)[0][0]
            most_common_eco = Counter(eco_list).most_common(1)[0][0]
            most_common_admin = Counter(admin_list).most_common(1)[0][0]
            avg_perc_free = np.sum(perc_free_list) / len(subset_df)
            avg_perc_forest = np.sum(perc_forest_list) / len(subset_df)

            # Average MAT and PCP from surrounding grid cells
            mat = (mat_np[i, j] + mat_np[i, j + 1] + mat_np[i + 1, j] + mat_np[i + 1, j + 1]) / 4
            pcp = (pcp_np[i, j] + pcp_np[i, j + 1] + pcp_np[i + 1, j] + pcp_np[i + 1, j + 1]) / 4

            new_row = pd.Series({
                "lat_min": cfg.lats[i],
                "lat_max": cfg.lats[i + 1],
                "lon_min": cfg.lons[j],
                "lon_max": cfg.lons[j + 1],
                "species": species,
                "species_perc": species_perc,
                "forest_age": avg_forest_age,
                "fire_fraction": fire_fraction,
                "fire_year": most_common_fire_year,
                "eco": most_common_eco,
                "admin": most_common_admin,
                "perc_free": avg_perc_free,
                "perc_forest": avg_perc_forest,
                "mat": mat,
                "pcp": pcp
            }).to_frame().T

            compile_grid_df = pd.concat([compile_grid_df, new_row], ignore_index=True)

    out_fp = os.path.join(cfg.land_cover_dir, "compile_grid_df.pkl")
    compile_grid_df.to_pickle(out_fp)


def create_scenarios(cfg):
    """
    Create simulation scenarios based on compiled grid information.

    Filters the compiled grid dataframe based on ecozones, administrative regions, and historical land
    classification. Then, for a number of Monte Carlo iterations, it creates experiment scenarios by generating
    GeoDataFrames for inventory, afforestation, fire, and (optionally) mortality. The experiment configuration is
    saved as a pickle file and the GeoDataFrames are exported as shapefiles.

    Parameters:
        cfg: Configuration object with attributes such as:
             - land_cover_dir: Directory of the compiled grid dataframe.
             - hist_land_class: Historical land classification identifier.
             - mc_iters: Number of Monte Carlo iterations.
             - exp_yield_perc: Experiment yield percentage.
             - exps_dir: Directory to save experiment scenarios.
             - gen_mortality: Boolean indicating whether to generate mortality data.
             - mortality_year: Year for mortality assignment.
             - fire_assignment: Method for assigning fire events ('binom' or 'burn_fraction').
             - hist_dist_type: Historical disturbance type.
             - aff_year: Year for afforestation.
             - run_baseline: Boolean indicating whether to run the baseline scenario.
             - (and other parameters used by helper functions)
    """
    compiled_fp = os.path.join(cfg.land_cover_dir, "compile_grid_df.pkl")
    compiled_df = pd.read_pickle(compiled_fp)

    # Filter based on ecozone and admin region
    compiled_df = compiled_df.loc[
        ((compiled_df["eco"] == "Taiga Plains") | (compiled_df["eco"] == "Taiga Shield West")) &
        (~compiled_df["admin"].isnull())
    ]
    compiled_df = compiled_df.loc[
        (compiled_df["admin"] == "Yukon") |
        (compiled_df["admin"] == "Northwest Territories") |
        (compiled_df["admin"] == "Saskatchewan") |
        (compiled_df["admin"] == "Manitoba")
    ]

    # Apply historical land classification filter
    if cfg.hist_land_class == "FL":
        compiled_df = compiled_df.loc[(compiled_df["forest_age"] != 0) & (compiled_df["perc_free"] < 70)]
    elif cfg.hist_land_class in ["CL", "NFL", "GL"]:
        compiled_df = compiled_df.loc[(compiled_df["forest_age"] == 0) | (compiled_df["perc_free"] >= 70)]

    exp_num = 0
    num_tiles = 1
    for _ in range(cfg.mc_iters):
        exp_num += 1
        exp_cfg = ExpConfig()
        exp_cfg.num_tiles = num_tiles
        exp_cfg.yield_perc = cfg.exp_yield_perc

        cur_exp_dir = os.path.join(cfg.exps_dir, f"{exp_num}_exp")
        os.makedirs(cur_exp_dir, exist_ok=True)

        inventory_dir = os.path.join(cur_exp_dir, "inventory")
        os.makedirs(inventory_dir, exist_ok=True)

        disturbances_dir = os.path.join(cur_exp_dir, "disturbances")
        os.makedirs(disturbances_dir, exist_ok=True)

        aff_row = 0
        inv_row = 0
        fire_row = 0
        gdf_inv = gpd.GeoDataFrame(columns=[
            "geometry", "age", "LdSpp", "soil_type", "land_class", "dom_soil_p",
            "admin", "eco"
        ])
        gdf_aff = gpd.GeoDataFrame(columns=["geometry", "year", "LdSpp", "perc_free"])
        gdf_fire = gpd.GeoDataFrame(columns=["geometry", "year", "fri", "fire_frac"])
        if cfg.gen_mortality:
            mort_row = 0
            gdf_mort = gpd.GeoDataFrame(columns=["geometry", "year"])

        compiled_sample = compiled_df.sample(num_tiles).reset_index(drop=True)

        for idx in range(len(compiled_sample)):
            row = compiled_sample.loc[idx]
            row = regrid_sample(row)

            age, land_class = get_age_land_class(cfg, row)
            inv_poly = get_inventory_poly(row)
            soil_type, dom_soil_perc = sample_soil_type(cfg, row)
            sp_historic = sample_species_historic(cfg, row)
            aff_poly, perc_free = get_afforestation_poly(row)
            sp_aff = sample_species_aff(cfg, row)

            if perc_free == 0 and not cfg.run_baseline:
                continue

            gdf_inv.loc[inv_row] = {
                'geometry': inv_poly,
                'age': age,
                'LdSpp': sp_historic,
                'soil_type': soil_type,
                'land_class': land_class,
                'dom_soil_p': dom_soil_perc,
                'admin': row["admin"],
                'eco': row["eco"],
                'perc_forest': row["perc_forest"],
                'mat': row["mat"],
                'hist_dist_type': cfg.hist_dist_type
            }
            inv_row += 1

            if not cfg.run_baseline:
                gdf_aff.loc[aff_row] = {
                    'geometry': aff_poly,
                    'year': cfg.aff_year,
                    'LdSpp': sp_aff,
                    'perc_free': perc_free
                }
                aff_row += 1

                gdf_inv.loc[inv_row] = {
                    'geometry': aff_poly,
                    'age': age,
                    'LdSpp': sp_historic,
                    'soil_type': soil_type,
                    'land_class': land_class,
                    'dom_soil_p': dom_soil_perc,
                    'admin': row["admin"],
                    'eco': row["eco"],
                    'perc_forest': row["perc_forest"],
                    'mat': row["mat"],
                    'hist_dist_type': cfg.hist_dist_type
                }
                inv_row += 1

            if cfg.gen_mortality:
                gdf_mort.loc[mort_row] = {
                    'geometry': aff_poly,
                    'year': cfg.mortality_year
                }
                mort_row += 1

            if cfg.fire_assignment == "binom":
                year_list, fri, poly_list, fire_fraction_mean, re_list = sample_fire_event_binom(cfg, row)
            elif cfg.fire_assignment == "burn_fraction":
                year_list, fri, poly_list, fire_fraction_mean, re_list = sample_fire_event_burnfrac(cfg, row)
            else:
                continue  # or handle unexpected method

            for ev, year in enumerate(year_list):
                gdf_fire.loc[fire_row] = {
                    'geometry': poly_list[ev],
                    'year': year,
                    'fri': fri,
                    'fire_frac': fire_fraction_mean
                }
                fire_row += 1

                if year != 2101:
                    assign_firearea_inventory(cfg, gdf_inv, inv_row, ev, year, poly_list,
                                              aff_poly, age, sp_historic, sp_aff, land_class)
                    gdf_inv.loc[inv_row, 'soil_type'] = soil_type
                    gdf_inv.loc[inv_row, 'dom_soil_p'] = dom_soil_perc
                    gdf_inv.loc[inv_row, 'admin'] = row["admin"]
                    gdf_inv.loc[inv_row, 'eco'] = row["eco"]
                    gdf_inv.loc[inv_row, 'perc_forest'] = row["perc_forest"]
                    gdf_inv.loc[inv_row, 'mat'] = row["mat"]
                    if re_list[ev] == 0:
                        gdf_inv.loc[inv_row, 'hist_dist_type'] = cfg.hist_dist_type
                    elif re_list[ev] == 1:
                        gdf_inv.loc[inv_row, 'hist_dist_type'] = "Wildfire"
                    inv_row += 1

        # Save experiment configuration and export shapefiles
        outfp_cfg = os.path.join(cur_exp_dir, "exp_cfg.pkl")
        with open(outfp_cfg, "wb") as outfp:
            pickle.dump(exp_cfg, outfp)

        gdf_inv.set_crs(epsg=4326, inplace=True)
        gdf_inv.to_file(os.path.join(inventory_dir, "inventory.shp"))

        gdf_aff.set_crs(epsg=4326, inplace=True)
        gdf_aff.to_file(os.path.join(disturbances_dir, "afforestation.shp"))

        gdf_fire.set_crs(epsg=4326, inplace=True)
        gdf_fire.to_file(os.path.join(disturbances_dir, "fire.shp"))

        if cfg.gen_mortality:
            gdf_mort.set_crs(epsg=4326, inplace=True)
            gdf_mort.to_file(os.path.join(disturbances_dir, "mortality.shp"))


def get_top_mixtures(cfg):
    """
    Identify and print the top 10 most common species mixture combinations.

    Loads the compiled grid dataframe, filters based on ecozone and forest cover, bins species percentages
    into intervals of 10, and prints the top 10 most frequent mixture combinations (as tuples of species and
    binned percentages).

    Parameters:
        cfg: Configuration object with attributes:
             - land_cover_dir: Directory where the compiled grid dataframe is stored.
    """
    def bin_percentage(p):
        return (int(p) // 10) * 10

    def get_mixture_key(row):
        species_list = row['species']
        perc_list = row['species_perc']
        binned = [(sp, bin_percentage(p)) for sp, p in zip(species_list, perc_list)]
        return tuple(sorted(binned, key=lambda x: x[0]))

    compiled_fp = os.path.join(cfg.land_cover_dir, "compile_grid_df.pkl")
    compiled_df = pd.read_pickle(compiled_fp)
    compiled_df = compiled_df.loc[
        (compiled_df["eco"] == "Taiga Plains") &
        (compiled_df["perc_forest"] > 60) &
        (~compiled_df["admin"].isnull())
    ]
    compiled_df['mix'] = compiled_df.apply(get_mixture_key, axis=1)
    top10 = compiled_df['mix'].value_counts().head(10)
    print(f"Top 10 mixture combinations (species, binned percentage): {top10}")


if __name__ == "__main__":
    # Instantiate configuration object (assumed to be provided by util_functions or similar)
    cfg = Config()

    # Example shapefile read (paths to be defined as needed)
    admin_eco_shp = gpd.read_file("")
    shp_dir = ""
    cfg.transf_dir = ""
    cfg.land_cover_dir = ""

    # Uncomment the following function calls as needed:
    # compile_grid_info(cfg)
    # create_yield_curves(cfg)
    # create_scenarios(cfg)
    # create_combined_df(cfg)
    # get_top_mixtures(cfg)
