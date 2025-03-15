import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset
from ncload import NCLoad


def basemap_plot(plot_dir, data, plot_name, plot_title, lats, lons, cmap, cbar_ticks, mode):
    """
    Generate and save a basemap plot for the provided data.

    Parameters:
        plot_dir (str): Directory where the plot image will be saved.
        data (np.ndarray): 2D array of data values to plot.
        plot_name (str): Base name for the plot file.
        plot_title (str): Title for the plot.
        lats (np.ndarray): Array of latitudes.
        lons (np.ndarray): Array of longitudes.
        cmap (str): Colormap to use for the plot.
        cbar_ticks (list or None): Colorbar ticks if provided.
        mode (str): Plot mode. Valid options are "canada" and "globe".

    Raises:
        ValueError: If an unsupported mode is provided.
    """
    fig = plt.figure()

    if mode == "canada":
        m = Basemap(llcrnrlat=44, llcrnrlon=-148, urcrnrlat=75, urcrnrlon=-50,
                    resolution='l', projection='merc', lat_0=60, lon_0=-99)
    elif mode == "globe":
        m = Basemap(resolution='l', projection='cyl')
    else:
        raise ValueError("Unsupported mode: choose 'canada' or 'globe'")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    xi, yi = m(lon_grid, lat_grid)

    norm = TwoSlopeNorm(vcenter=0)
    cs = m.pcolormesh(xi, yi, data, cmap=cmap, norm=norm, shading="auto")
    m.drawcoastlines()
    m.drawcountries()

    cbar = m.colorbar(cs, location='bottom', pad="10%")
    if cbar_ticks:
        cbar.set_ticks(cbar_ticks)

    plt.title(plot_title)
    save_path = os.path.join(plot_dir, f"{plot_name}_{mode}.png")
    plt.savefig(save_path)
    plt.close()


def compute_albedo_std_year(data_dir):
    """
    Compute the average monthly standard deviation of albedo for a year.

    This function processes daily NetCDF files for one year. For each month, it
    calculates the standard deviation across the days, then returns the mean of
    these monthly standard deviations.

    Parameters:
        data_dir (str): Directory containing the daily NetCDF files.

    Returns:
        np.ndarray: 2D array representing the mean standard deviation of albedo.
    """
    days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    m = 0
    cum_days = 0
    albedo_std_year = []
    albedo_month = []

    for i in range(1, 366):
        filename = f"MODIS_Blue_Sky_Albedo_Climatology.CMG005.{i:03}.nc"
        file_path = os.path.join(data_dir, filename)
        albedo_nc = NCLoad(file_path)
        albedo_day = np.array(albedo_nc.get('hist_alb_clim')[:])
        albedo_month.append(albedo_day)

        # End of year: process remaining days
        if i == 365:
            albedo_arr = np.array(albedo_month)
            month_std = np.std(albedo_arr, axis=0)
            albedo_std_year.append(month_std)
        # End of month: compute and reset the monthly list
        elif (i - cum_days) == days_month[m]:
            m += 1
            cum_days += days_month[m]
            albedo_arr = np.array(albedo_month)
            month_std = np.std(albedo_arr, axis=0)
            albedo_std_year.append(month_std)
            albedo_month = []

    albedo_std_year = np.array(albedo_std_year)
    return np.mean(albedo_std_year, axis=0)


def load_albedo_std_year(plot_dir):
    """
    Load precomputed albedo standard deviation data from a .npy file and transpose it.

    Parameters:
        plot_dir (str): Directory containing the 'albedo_std_month.npy' file.

    Returns:
        np.ndarray: Transposed albedo standard deviation data.
    """
    data_path = os.path.join(plot_dir, "albedo_std_month.npy")
    albedo_std_year = np.load(data_path)
    return np.transpose(albedo_std_year)


def save_albedo_to_netcdf(output_file, lats, lons, albedo_data):
    """
    Save albedo data to a NetCDF file.

    Parameters:
        output_file (str): Path to the output NetCDF file.
        lats (np.ndarray): Array of latitudes.
        lons (np.ndarray): Array of longitudes.
        albedo_data (np.ndarray): 2D array of albedo data.
    """
    with Dataset(output_file, 'w', format='NETCDF3') as ncout:
        ncout.createDimension('lon', lons.size)
        ncout.createDimension('lat', lats.size)
        lonvar = ncout.createVariable('lon', 'float32', ('lon',))
        lonvar[:] = lons
        latvar = ncout.createVariable('lat', 'float32', ('lat',))
        latvar[:] = lats
        albedo_var = ncout.createVariable('albedo', 'float32', ('lat', 'lon'))
        albedo_var[:] = albedo_data


def main():
    """
    Main routine to compute or load albedo standard deviation, save the result to a NetCDF file,
    and generate plots using basemap.
    """
    # Set directories for input and output files
    data_dir = ""   # Directory containing raw NetCDF files (used in 'compute' mode)
    plot_dir = ""   # Directory where output files and plots will be saved

    # Define latitude and longitude arrays
    lats = np.arange(90, -90, -0.05)
    lons = np.arange(-180, 180, 0.05)

    # Select processing mode: "compute" to process raw data or "load" to load precomputed data.
    mode = "load"  # Change to "compute" if raw data processing is required

    if mode == "compute":
        albedo_std_year = compute_albedo_std_year(data_dir)
    elif mode == "load":
        albedo_std_year = load_albedo_std_year(plot_dir)
    else:
        raise ValueError("Invalid mode. Choose 'compute' or 'load'.")

    # Save the albedo data to a NetCDF file
    netcdf_output = os.path.join(plot_dir, "albedo_std_month.nc")
    save_albedo_to_netcdf(netcdf_output, lats, lons, albedo_std_year)

    # Plot settings
    cmap = "coolwarm"
    plot_title = ""

    # Generate plots in both "canada" and "globe" modes
    for plot_mode in ["canada", "globe"]:
        basemap_plot(plot_dir, albedo_std_year, "albedo_std_month", plot_title,
                     lats, lons, cmap, cbar_ticks=None, mode=plot_mode)


if __name__ == "__main__":
    main()
