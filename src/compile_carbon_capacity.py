import os
import json
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from util_functions import *
from config import Config


def transform_to_geojson(df, name="fl"):
    """
    Convert a DataFrame with bounding box coordinates into a GeoJSON file.

    The DataFrame must contain columns 'lat_min', 'lat_max', 'lon_min', and 'lon_max'
    to define the bounding box of each feature. Additional columns (except for "index" and
    "species") are added as properties.

    Parameters:
        df (pandas.DataFrame): Input DataFrame with geographic and property data.
        name (str, optional): Base name for the output GeoJSON file. Defaults to "fl".

    The resulting GeoJSON file is saved in the directory specified by cfg.land_cover_dir.
    """
    geo_data = {"type": "FeatureCollection", "features": []}
    # Exclude these columns from properties
    rem_cols = ["lat_min", "lat_max", "lon_min", "lon_max", "index", "species"]
    props_cols = [col for col in df.columns if col not in rem_cols]

    for i, row in df.iterrows():
        feature = {
            "type": "Feature",
            "id": i,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [row["lon_min"], row["lat_min"]],
                    [row["lon_max"], row["lat_min"]],
                    [row["lon_max"], row["lat_max"]],
                    [row["lon_min"], row["lat_max"]],
                    [row["lon_min"], row["lat_min"]]
                ]]
            },
            "properties": {}
        }
        for col in props_cols:
            value = row[col]
            if isinstance(value, np.float64):
                value = float(value)
            elif isinstance(value, np.int64):
                value = int(value)
            feature["properties"][col] = value
        geo_data["features"].append(feature)

    output_path = os.path.join(cfg.land_cover_dir, f"{name}.geojson")
    with open(output_path, 'w') as f:
        json.dump(geo_data, f)


def plot_props(canada_shp, all_props, geojson, name="fl"):
    """
    Create and save property plots for each property in all_props.

    For each property, a plot is generated with the Canadian boundaries in the background and
    the geojson data colored by the given property. The plot is saved to cfg.plot_dir/geojsons.

    Parameters:
        canada_shp (GeoDataFrame): GeoDataFrame of Canada boundaries.
        all_props (iterable): Iterable of property names (strings) to plot.
        geojson (GeoDataFrame): GeoDataFrame containing GeoJSON data.
        name (str, optional): Base name for the output files. Defaults to "fl".
    """
    for prop in all_props:
        fig, ax = plt.subplots(figsize=(8, 8))
        canada_shp.plot(ax=ax, facecolor="oldlace", edgecolor="dimgray")
        geojson.plot(ax=ax, column=prop, legend=True)
        # ax.set_title(prop)
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1, 1.2))
        save_path = os.path.join(cfg.plot_dir, "geojsons", f"{name}_{prop}.png")
        plt.savefig(save_path)
        plt.close()


def compute_carbon(combined_geo, fl_tec_years, fl_tec_std_years,
                   nfl_tec_years, nfl_tec_std_years,
                   bl_years, bl_std_years):
    """
    Compute carbon estimates for each feature in a GeoDataFrame and save the results.

    For each geographic feature, TEC (Total Ecosystem Carbon) values are adjusted based on the
    percentage of free and forested area. Computed values (and their CO2 equivalents) are stored as
    new columns in the GeoDataFrame. The updated GeoDataFrame is then saved as 'carbon.geojson'
    in the directory specified by cfg.land_cover_dir.

    Parameters:
        combined_geo (GeoDataFrame): GeoDataFrame with features to process.
        fl_tec_years (list-like): TEC values for forest land.
        fl_tec_std_years (list-like): TEC standard deviation for forest land.
        nfl_tec_years (list-like): TEC values for non-forest land.
        nfl_tec_std_years (list-like): TEC standard deviation for non-forest land.
        bl_years (list-like): Baseline TEC values.
        bl_std_years (list-like): Baseline TEC standard deviation values.
    """
    list_years = [25, 50, 75]

    for i in range(len(combined_geo)):
        land_type = combined_geo.loc[i, "type"]
        area = combined_geo.loc[i, "area"]
        perc_free = combined_geo.loc[i, "perc_free"]
        area_free = (perc_free / 100) * area
        perc_forest = combined_geo.loc[i, "perc_forest"]
        area_forest = (perc_forest / 100) * area

        for y in list_years:
            bl_tec = bl_years[y + 10]
            bl_tec_std = bl_std_years[y + 10]
            if land_type == "fl":
                tot_tec = fl_tec_years[y + 10]
                tot_tec_std = fl_tec_std_years[y + 10]
                aff_tec = (tot_tec - bl_tec) * area_free
                aff_tec_std = (tot_tec_std - bl_tec_std) * area_free
            elif land_type == "nfl":
                aff_tec = nfl_tec_years[y + 10] * area_free
                aff_tec_std = nfl_tec_std_years[y + 10] * area_free

            # Adjust baseline TEC based on forest area
            bl_tec_adjusted = area_forest * bl_tec
            bl_tec_std_adjusted = area_forest * bl_tec_std

            combined_geo.at[i, f"{y}_aff_tec"] = aff_tec
            combined_geo.at[i, f"{y}_aff_tec_std"] = aff_tec_std
            combined_geo.at[i, f"{y}_aff_tec_co2"] = aff_tec * 3.67
            combined_geo.at[i, f"{y}_aff_tec_std_co2"] = aff_tec_std * 3.67

            combined_geo.at[i, f"{y}_bl_tec"] = bl_tec_adjusted
            combined_geo.at[i, f"{y}_bl_tec_std"] = bl_tec_std_adjusted
            combined_geo.at[i, f"{y}_bl_tec_co2"] = bl_tec_adjusted * 3.67
            combined_geo.at[i, f"{y}_bl_tec_std_co2"] = bl_tec_std_adjusted * 3.67

    output_path = os.path.join(cfg.land_cover_dir, "carbon.geojson")
    combined_geo.to_file(output_path)


def plot_carbon(carbon_geo, canada_shp, plot_props_list):
    """
    Generate and save carbon-related plots for the specified properties.

    For each property in plot_props_list, the function prints the total carbon value, then
    creates a plot with the Canadian boundaries as background and the carbon data colored by the
    given property. The plot is saved to cfg.plot_dir/geojsons with a filename corresponding to the property.

    Parameters:
        carbon_geo (GeoDataFrame): GeoDataFrame containing carbon data.
        canada_shp (GeoDataFrame): GeoDataFrame of Canada boundaries.
        plot_props_list (iterable): Iterable of property names (strings) to plot.
    """
    for prop in plot_props_list:
        print(f"{prop}: {np.sum(carbon_geo[prop])}")

        fig, ax = plt.subplots(figsize=(8, 8))
        canada_shp.plot(ax=ax, facecolor="oldlace", edgecolor="dimgray")
        carbon_geo.plot(ax=ax, column=prop, legend=True)
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1, 1.2))
        save_path = os.path.join(cfg.plot_dir, "geojsons", f"{prop}.png")
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    cfg = Config()
    pass
