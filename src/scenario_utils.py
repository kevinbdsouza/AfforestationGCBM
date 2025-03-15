import random
import numpy as np
from scipy.stats import weibull_min, binom
from shapely.geometry import Polygon
from util_functions import *  # Assumes necessary configuration and helper functions are imported


def sample_species_historic(cfg, row):
    """
    Sample a historic species from the given row based on species percentages.

    If the species list is empty, returns "Nope". Otherwise, randomly selects one species
    from the row using the provided percentages, then returns its common name from the config.

    Parameters:
        cfg: Configuration object containing species name mapping.
        row (dict-like): A record that includes keys "species" and "species_perc".

    Returns:
        str: The common name of the sampled species.
    """
    species = row["species"]
    if len(species) == 0:
        return "Nope"
    perc_list = np.round(row["species_perc"], 2)
    chosen_species = random.choices(species, weights=perc_list, k=1)[0]
    return cfg.species_cm_name_dict[chosen_species]


def sample_species_aff(cfg, row):
    """
    Sample an afforestation species from the given row.

    If the species list is empty, sample from the ecozone-specific dictionary; otherwise,
    sample from the given species list with associated percentages. The function returns the
    common name of the sampled species from the config.

    Parameters:
        cfg: Configuration object containing species mappings and ecozone details.
        row (dict-like): A record containing "species", "species_perc", and "eco".

    Returns:
        str: The common name of the sampled species.
    """
    species = row["species"]
    if len(species) == 0:
        species_dict = cfg.species_ecozones[row["eco"]]
        sp_list = list(species_dict.keys())
        sp_perc_list = list(species_dict.values())
        chosen_species = random.choices(sp_list, weights=sp_perc_list, k=1)[0]
    else:
        perc_list = np.round(row["species_perc"], 2)
        chosen_species = random.choices(species, weights=perc_list, k=1)[0]
    return cfg.species_cm_name_dict[chosen_species]


def get_age_land_class(cfg, row):
    """
    Determine the forest age and land class based on historic configuration.

    For land classes "CL", "NFL", or "GL", the age is set to 0. For "FL", the age is sampled
    from a normal distribution centered on the row's forest_age with a standard deviation equal
    to 10% of that value.

    Parameters:
        cfg: Configuration object containing historic land class information.
        row (dict-like): A record with a "forest_age" key.

    Returns:
        tuple: (age, land_class) where age is a float (or 0) and land_class is a string.
    """
    age_mean = row["forest_age"]
    age_std = 0.1 * age_mean
    land_class = cfg.hist_land_class
    if land_class in ["CL", "NFL", "GL"]:
        age = 0
    elif land_class == "FL":
        age = np.round(np.random.normal(age_mean, age_std, 1)[0], 2)
    return age, land_class


def sample_soil_type(cfg, row):
    """
    Sample a soil type for the given ecozone and row.

    Randomly selects a dominant soil type and a dominant soil percentage from the config.
    With probability equal to the dominant soil percentage, returns the dominant soil type;
    otherwise, returns "Average".

    Parameters:
        cfg: Configuration object containing soil type dictionary and dominant soil percentages.
        row (dict-like): A record containing the "eco" key.

    Returns:
        tuple: (soil_type (str), dom_soil_perc (float))
    """
    eco = row["eco"]
    dom_soil_type = random.choice(cfg.soil_type_dict[eco])
    dom_soil_perc = random.choice(cfg.dom_soil_perc)
    p = random.uniform(0, 1)
    soil_type = dom_soil_type if p < dom_soil_perc else "Average"
    return soil_type, dom_soil_perc


def get_inventory_poly(row):
    """
    Create a Polygon representing an inventory based on row boundaries.

    Constructs a rectangular polygon using the latitude and longitude bounds from the row.

    Parameters:
        row (dict-like): A record containing "lat_min", "lat_max", "lon_min", and "lon_max".

    Returns:
        Polygon: A Shapely Polygon object.
    """
    lat_min = row["lat_min"]
    lat_max = row["lat_max"]
    lon_min = row["lon_min"]
    lon_max = row["lon_max"]
    coords = [(lon_min, lat_min), (lon_max, lat_min),
              (lon_max, lat_max), (lon_min, lat_max)]
    return Polygon(coords)


def get_afforestation_poly(row):
    """
    Compute an afforestation polygon based on free area percentage.

    If the free area percentage (perc_free) is zero, returns None.
    Otherwise, computes a polygon representing the afforestation area by adjusting the lower latitude.

    Parameters:
        row (dict-like): A record containing "perc_free", "lat_min", "lat_max", "lon_min", and "lon_max".

    Returns:
        tuple: (afforestation Polygon or None, perc_free (float))
    """
    perc_free = np.round(row["perc_free"] / 100, 2)
    lat_min = row["lat_min"]
    lat_max = row["lat_max"]
    lon_min = row["lon_min"]
    lon_max = row["lon_max"]
    if perc_free == 0:
        aff_poly = None
    else:
        new_lat_min = lat_max - (lat_max - lat_min) * perc_free
        coords = [(lon_min, new_lat_min), (lon_max, new_lat_min),
                  (lon_max, lat_max), (lon_min, lat_max)]
        aff_poly = Polygon(coords)
    return aff_poly, perc_free


def sample_fire_event_binom(cfg, row):
    """
    Simulate fire events using a binomial probability model.

    Based on the configuration's fire return interval mode and fire fraction mode,
    this function generates a list of years, polygons representing fire events, and other
    relevant fire statistics.

    Parameters:
        cfg: Configuration object containing fire simulation parameters.
        row (dict-like): A record containing geographic bounds ("lat_min", "lat_max", "lon_min", "lon_max"),
                        "eco", and "fire_fraction".

    Returns:
        tuple: (year_list, fri, poly_list, fire_fraction, re_list) where:
            - year_list: List of years when fires occur.
            - fri: Fire return interval (float).
            - poly_list: List of Polygon objects representing fire-affected areas.
            - fire_fraction: Fire fraction value used in simulation.
            - re_list: List of indicators (0 or 1) for re-adjustment events.
    """
    lat_min_og = row["lat_min"]
    lat_max_og = row["lat_max"]
    lon_min_og = row["lon_min"]
    lon_max_og = row["lon_max"]

    # Determine fire return interval (fri)
    if cfg.fri_mode == "list":
        fri = random.choice(cfg.fire_return_interval)
    elif cfg.fri_mode == "weibull":
        eco = row["eco"]
        weibull_scale = random.choice(cfg.weibull_scale[eco])
        weibull_shape = random.choice(cfg.weibull_shape)
        fri = np.round(weibull_min.rvs(weibull_shape, scale=weibull_scale, size=1)[0], 2)

    year_list, poly_list, re_list = [], [], []
    if cfg.fire_frac_mode == "historic":
        fire_fraction = np.round(row["fire_fraction"], 2)
    elif cfg.fire_frac_mode == "van_wagner":
        # 'van_wagner' mode not implemented.
        fire_fraction = 0
    elif cfg.fire_frac_mode == "custom":
        fire_fraction = cfg.fire_frac

    if fire_fraction != 0:
        fire_year_prev = cfg.aff_year
        current_lat_min = lat_min_og
        re = 0
        for year in range(cfg.sim_interval[0], cfg.sim_interval[1] + 1):
            if year - cfg.aff_year <= 3:
                continue

            diff = year - fire_year_prev
            # Calculate probability of a fire event using binomial PMF
            prob = 1 - (binom.pmf(0, diff, 1 / fri) + binom.pmf(1, diff, 1 / fri))
            if random.uniform(0, 1) <= prob:
                current_lat_max = current_lat_min + (lat_max_og - lat_min_og) * fire_fraction
                if current_lat_max > lat_max_og:
                    current_lat_max = lat_max_og
                    coords = [(lon_min_og, current_lat_min), (lon_max_og, current_lat_min),
                              (lon_max_og, current_lat_max), (lon_min_og, current_lat_max)]
                    fire_poly = Polygon(coords)
                    poly_list.append(fire_poly)
                    year_list.append(year)
                    re_list.append(re)
                    temp_frac = (current_lat_max - current_lat_min) / (lat_max_og - lat_min_og)
                    fire_fraction_ud = fire_fraction - temp_frac
                    current_lat_min = lat_min_og
                    current_lat_max = current_lat_min + (lat_max_og - lat_min_og) * fire_fraction_ud
                    re = 1

                coords = [(lon_min_og, current_lat_min), (lon_max_og, current_lat_min),
                          (lon_max_og, current_lat_max), (lon_min_og, current_lat_max)]
                fire_poly = Polygon(coords)
                poly_list.append(fire_poly)
                year_list.append(year)
                re_list.append(re)
                fire_year_prev = year
                current_lat_min = current_lat_max

    if not year_list:
        year_list.append(2101)
        coords = [(lon_min_og, lat_min_og), (lon_max_og, lat_min_og),
                  (lon_max_og, lat_max_og), (lon_min_og, lat_max_og)]
        poly_list.append(Polygon(coords))

    return year_list, fri, poly_list, fire_fraction, re_list


def sample_fire_event_burnfrac(cfg, row):
    """
    Simulate fire events using a burn fraction based on the fire return interval.

    For a given geographic region, computes fire events by assuming a burn fraction equal
    to 1/fri. Generates lists of years, fire polygons, and re-adjustment flags.

    Parameters:
        cfg: Configuration object containing fire simulation parameters.
        row (dict-like): A record containing geographic bounds ("lat_min", "lat_max", "lon_min", "lon_max"),
                        and "eco".

    Returns:
        tuple: (year_list, fri, poly_list, fire_fraction, re_list) with similar structure to
               sample_fire_event_binom.
    """
    lat_min_og = row["lat_min"]
    lat_max_og = row["lat_max"]
    lon_min_og = row["lon_min"]
    lon_max_og = row["lon_max"]

    if cfg.fri_mode == "list":
        fri = random.choice(cfg.fire_return_interval)
    elif cfg.fri_mode == "weibull":
        eco = row["eco"]
        weibull_scale = random.choice(cfg.weibull_scale[eco])
        weibull_shape = random.choice(cfg.weibull_shape)
        fri = np.round(weibull_min.rvs(weibull_shape, scale=weibull_scale, size=1)[0], 2)

    year_list, poly_list, re_list = [], [], []
    fire_fraction = np.round(1 / fri, 2)

    if fire_fraction != 0:
        current_lat_min = lat_min_og
        re = 0
        for year in range(cfg.sim_interval[0], cfg.sim_interval[1] + 1):
            if year - cfg.aff_year <= 3:
                continue

            current_lat_max = current_lat_min + (lat_max_og - lat_min_og) * fire_fraction
            if current_lat_max > lat_max_og:
                current_lat_max = lat_max_og
                coords = [(lon_min_og, current_lat_min), (lon_max_og, current_lat_min),
                          (lon_max_og, current_lat_max), (lon_min_og, current_lat_max)]
                poly_list.append(Polygon(coords))
                year_list.append(year)
                re_list.append(re)
                temp_frac = (current_lat_max - current_lat_min) / (lat_max_og - lat_min_og)
                fire_fraction_ud = fire_fraction - temp_frac
                current_lat_min = lat_min_og
                current_lat_max = current_lat_min + (lat_max_og - lat_min_og) * fire_fraction_ud
                re = 1

            coords = [(lon_min_og, current_lat_min), (lon_max_og, current_lat_min),
                      (lon_max_og, current_lat_max), (lon_min_og, current_lat_max)]
            poly_list.append(Polygon(coords))
            year_list.append(year)
            re_list.append(re)
            current_lat_min = current_lat_max

    if not year_list:
        year_list.append(2101)
        coords = [(lon_min_og, lat_min_og), (lon_max_og, lat_min_og),
                  (lon_max_og, lat_max_og), (lon_min_og, lat_max_og)]
        poly_list.append(Polygon(coords))

    return year_list, fri, poly_list, fire_fraction, re_list


def regrid_sample(row):
    """
    Resample the geographic boundaries of a row to a finer grid.

    Divides the latitude and longitude ranges into increments of 0.002 and randomly selects
    adjacent grid cell boundaries to update the row.

    Parameters:
        row (dict-like): A record containing "lat_min", "lat_max", "lon_min", and "lon_max".

    Returns:
        dict-like: The updated row with new "lat_min", "lat_max", "lon_min", and "lon_max".
    """
    lat_min = row["lat_min"]
    lat_max = row["lat_max"]
    lon_min = row["lon_min"]
    lon_max = row["lon_max"]

    lats = np.arange(lat_min, lat_max, 0.002)
    lons = np.arange(lon_min, lon_max, 0.002)

    lat_idx = np.random.choice(len(lats) - 1)
    lon_idx = np.random.choice(len(lons) - 1)

    row["lat_min"] = np.round(lats[lat_idx], 3)
    row["lat_max"] = np.round(lats[lat_idx + 1], 3)
    row["lon_min"] = np.round(lons[lon_idx], 3)
    row["lon_max"] = np.round(lons[lon_idx + 1], 3)
    return row


def assign_firearea_inventory(cfg, gdf_inv, inv_row, ev, year, poly_list,
                              aff_poly, age, sp_historic, sp_aff, land_class):
    """
    Assign a fire event to an inventory record based on the fire and afforestation polygon bounds.

    Updates the GeoDataFrame (gdf_inv) at the specified row index with the fire event geometry,
    computed age, species assignment, and land classification based on the comparison of fire and
    afforestation lower latitude boundaries.

    Parameters:
        cfg: Configuration object containing simulation parameters.
        gdf_inv (GeoDataFrame): GeoDataFrame containing inventory records.
        inv_row: The index of the inventory row to update.
        ev (int): Index of the fire event in poly_list.
        year (int): The year in which the fire event occurs.
        poly_list (list): List of Polygon objects representing fire events.
        aff_poly (Polygon): Polygon representing the afforestation area.
        age (float): The age value computed for the record.
        sp_historic (str): Historic species common name.
        sp_aff (str): Afforestation species common name.
        land_class (str): Land classification from the historic data.

    Returns:
        GeoDataFrame: The updated GeoDataFrame with the assigned fire event.
    """
    fire_poly = poly_list[ev]
    latmin_fire = fire_poly.bounds[1]
    latmin_aff = aff_poly.bounds[1]
    aff_age = year - cfg.aff_year

    gdf_inv.loc[inv_row, 'geometry'] = fire_poly
    if latmin_fire < latmin_aff:
        gdf_inv.loc[inv_row, 'age'] = age
        gdf_inv.loc[inv_row, 'LdSpp'] = sp_historic
        gdf_inv.loc[inv_row, 'land_class'] = land_class
    else:
        gdf_inv.loc[inv_row, 'age'] = int(np.mean([aff_age, age]))
        gdf_inv.loc[inv_row, 'LdSpp'] = sp_aff
        gdf_inv.loc[inv_row, 'land_class'] = "FL"
    return gdf_inv
