import pandas as pd
import numpy as np
from pathlib import Path
from config import Config
import os

def load_data(paths: dict) -> dict:
    """Loads all necessary CSV files into a dictionary of pandas DataFrames."""
    dfs = {}
    for name, path in paths.items():
        try:
            dfs[name] = pd.read_csv(path)
            print(f"Successfully loaded {Path(path).name}")
        except FileNotFoundError:
            print(f"ERROR: File not found at {path}. Please check the path.")
            return None
        except Exception as e:
            print(f"An error occurred while loading {path}: {e}")
            return None
    return dfs

def calculate_soil_carbon(all_samples):
    """
    Calculates total soil carbon by combining all sample files and using the
    best available data for bulk density and carbon content, with correct unit conversions.
    """
    # --- Process each soil file individually to handle different column names and units ---
    # Drop rows where essential data for calculation is missing
    required_cols = [
        COLUMN_MAP['sample_upper_depth'], COLUMN_MAP['sample_lower_depth'],
        'unified_bulk_density', 'unified_carbon_percent'
    ]
    all_samples.dropna(subset=required_cols, inplace=True)

    # Calculate horizon thickness in meters from sample depths
    all_samples['thickness_m'] = (
        all_samples[COLUMN_MAP['sample_lower_depth']] - all_samples[COLUMN_MAP['sample_upper_depth']]
    ) / 100.0

    # Ensure thickness is non-negative
    all_samples = all_samples[all_samples['thickness_m'] >= 0].copy()

    # drop impossible values
    all_samples = all_samples[
        (all_samples['unified_bulk_density'].between(0.1, 1.5)) &  # 0.05–1.8 g cm-3
        (all_samples['unified_carbon_percent'].between(0, 60))  # 0–60 %
        ].copy()

    # Calculate carbon in Mg/ha for each sample's layer
    all_samples['carbon_mg_ha'] = (
        all_samples['thickness_m'] *
        all_samples['unified_bulk_density'] *
        (all_samples['unified_carbon_percent'] / 100.0) *
        10000
    )

    # Sum carbon across all sample layers for each plot-visit
    total_soil_carbon = all_samples.groupby(
        [COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']]
    )['carbon_mg_ha'].sum().reset_index()
    total_soil_carbon.rename(columns={'carbon_mg_ha': 'soil_c_mg_ha'}, inplace=True)
    return total_soil_carbon


def calculate_live_biomass_per_plot(tree_df, header_df, agb_col, status_col, dbh_col=None):
    """
    Calculates total live above-ground biomass (AGB) per plot-visit.
    Returns biomass in Megagrams per hectare (Mg/ha).
    """
    # Filter for live trees only (status code starts with 'L')
    live_mask = tree_df[status_col].astype(str).str.upper().str.startswith('L')
    live_df = tree_df.loc[live_mask].copy()

    # Sum biomass (in kg) per plot-visit
    biomass_kg = live_df.groupby([COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']])[agb_col].sum()

    # Calculate stem density (for large trees) if DBH is provided
    stems_per_plot = pd.Series(dtype=float)
    if dbh_col:
        stems_per_plot = live_df.groupby([COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']])[dbh_col].count()

    # Combine with header to get plot area for per-hectare conversion
    plot_summary = pd.DataFrame(biomass_kg).reset_index()
    if dbh_col:
        plot_summary = plot_summary.merge(
            pd.DataFrame(stems_per_plot).reset_index().rename(columns={dbh_col: 'stem_count'}),
            on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']],
            how='left'
        )

    plot_summary = plot_summary.merge(
        header_df[[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id'], COLUMN_MAP['plot_area']]],
        on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']],
        how='left'
    )

    # Convert biomass to Mg/ha
    plot_summary['agb_mg_ha'] = (plot_summary[agb_col] / 1000) / plot_summary[COLUMN_MAP['plot_area']]

    # Calculate stems per hectare
    if 'stem_count' in plot_summary.columns:
        plot_summary['stems_per_ha'] = plot_summary['stem_count'] / plot_summary[COLUMN_MAP['plot_area']]

    return plot_summary

def determine_leading_species(spc_lt_df, spc_st_df):
    """Determines the leading species (genus + species) by percentage from combined tables."""
    # Combine large and small tree species composition
    spc_lt_df['source'] = 'large'
    spc_st_df['source'] = 'small'
    spc_all = pd.concat([spc_lt_df, spc_st_df], ignore_index=True)

    # Clean up column names and create full species name
    spc_all[COLUMN_MAP['spc_percent']] = pd.to_numeric(spc_all[COLUMN_MAP['spc_percent']], errors='coerce')
    spc_all['full_species'] = (
        spc_all[COLUMN_MAP['spc_genus']].fillna('').str.strip() + ' ' +
        spc_all[COLUMN_MAP['spc_species']].fillna('').str.strip()
    ).str.strip()
    spc_all['genus_upper'] = spc_all[COLUMN_MAP['spc_genus']].str.upper()


    # Find the species with the maximum percentage for each plot-visit
    idx_max = spc_all.groupby([COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']])[COLUMN_MAP['spc_percent']].idxmax()
    idx_max = idx_max.dropna()
    leading_species_df = spc_all.loc[idx_max, [COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id'], 'full_species', 'genus_upper']].copy()
    leading_species_df.rename(columns={'full_species': 'lead_species'}, inplace=True)

    return leading_species_df


def process_all_data():
    """Main function to run the entire data aggregation pipeline."""
    print("--- Starting NFI Data Aggregation Pipeline (with TEC) ---")
    data = load_data(DATA_PATHS)
    if not data:
        print("Pipeline aborted due to loading errors.")
        return

    # 1. Process Live Carbon
    ltp_agb = calculate_live_biomass_per_plot(data['ltp_tree'], data['ltp_header'], COLUMN_MAP['lt_agb'],
                                              COLUMN_MAP['lt_status'], COLUMN_MAP['lt_dbh'])
    stp_agb = calculate_live_biomass_per_plot(data['stp_tree'], data['stp_header'], COLUMN_MAP['st_agb'],
                                              COLUMN_MAP['st_status'])
    plot_data = pd.merge(ltp_agb, stp_agb, on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']], how='outer',
                         suffixes=('_large', '_small')).fillna(0)
    plot_data['total_agb_mg_ha'] = plot_data['agb_mg_ha_large'] + plot_data['agb_mg_ha_small']
    header_info = data['ltp_header'][
        [COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id'], COLUMN_MAP['visit_date'], COLUMN_MAP['stand_age']]].copy()
    header_info['visit_year'] = pd.to_datetime(header_info[COLUMN_MAP['visit_date']]).dt.year
    plot_data = plot_data.merge(header_info, on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']], how='left')
    leading_species = determine_leading_species(data['ltp_spc'], data['stp_spc'])
    plot_data = plot_data.merge(leading_species, on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']], how='left')
    plot_data['species_type'] = np.where(plot_data['genus_upper'].isin(CONIFER_GENERA), 'CONIFER', 'BROADLEAF')
    plot_data['rs_ratio'] = plot_data['species_type'].map(RS_RATIOS)
    plot_data['total_bgb_mg_ha'] = plot_data['total_agb_mg_ha'] * plot_data['rs_ratio']

    plot_data['agb_c_mg_ha'] = plot_data['total_agb_mg_ha'] * CARBON_FRACTION
    plot_data['bgb_c_mg_ha'] = plot_data['total_bgb_mg_ha'] * CARBON_FRACTION
    plot_data['live_c_mg_ha'] = plot_data['agb_c_mg_ha'] + plot_data['bgb_c_mg_ha']

    # 2. Process Woody Debris Carbon
    wd_cols = ['plotbio_swd', 'plotbio_wd', 'plotbio_roundwd', 'plotbio_oddwd']
    wd_data = data['wd_summary'][[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']] + wd_cols].copy()
    wd_data['wd_c_mg_ha'] = wd_data[wd_cols].sum(axis=1) * CARBON_FRACTION
    plot_data = plot_data.merge(wd_data[[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id'], 'wd_c_mg_ha']],
                                on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']], how='left')

    # 3. Process Soil Carbon (for Stocks - uses full depth)
    ff_df = data['forest_floor'].copy();
    ff_df['unified_carbon_percent'] = pd.to_numeric(ff_df.get('tc_8mm'), errors='coerce') / 10.0;
    ff_df['unified_bulk_density'] = pd.to_numeric(ff_df.get('bulk_density_total'), errors='coerce')
    so_df = data['soil_org'].copy();
    so_df['unified_carbon_percent'] = pd.to_numeric(so_df.get('tc_8mm'), errors='coerce') / 10.0;
    so_df['unified_bulk_density'] = pd.to_numeric(so_df.get('bulk_density_total'), errors='coerce')
    sm_df = data['soil_mineral'].copy();
    sm_df['unified_carbon_percent'] = pd.to_numeric(sm_df.get('tc'), errors='coerce') / 10.0;
    sm_df['unified_bulk_density'] = pd.to_numeric(sm_df.get('bulk_density_total'), errors='coerce').fillna(
        pd.to_numeric(sm_df.get('bulk_density_2mm'), errors='coerce'))

    all_samples = pd.concat([ff_df, so_df, sm_df], ignore_index=True)

    soil_carbon_data = calculate_soil_carbon(all_samples.copy())  # Use a copy for total stock calculation
    plot_data = plot_data.merge(soil_carbon_data, on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']], how='left')

    # 4. Calculate Total Ecosystem Carbon (TEC) and define strata
    plot_data.fillna({'wd_c_mg_ha': 0, 'soil_c_mg_ha': 0}, inplace=True)
    plot_data['tec_mg_ha'] = plot_data['live_c_mg_ha'] + plot_data['wd_c_mg_ha'] + plot_data['soil_c_mg_ha']
    plot_data['density_class'] = np.where(plot_data['stems_per_ha'] >= DENSITY_THRESHOLD_SPH, 'Dense', 'Sparse')
    plot_data['age_bin'] = pd.cut(plot_data[COLUMN_MAP['stand_age']], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    plot_data['lead_species'].fillna('Unknown', inplace=True)

    # 5. Aggregate and Save Stock for Each Pool (using full depth soil C)
    print("\n--- Aggregating and Saving Detailed Stocks ---")
    pools_to_aggregate = {'AGB_C': 'agb_c_mg_ha', 'BGB_C': 'bgb_c_mg_ha', 'Live_C': 'live_c_mg_ha',
                          'Soil_C': 'soil_c_mg_ha', 'WD_C': 'wd_c_mg_ha', 'TEC': 'tec_mg_ha'}
    for pool_name, col_name in pools_to_aggregate.items():
        agg_dict = {'plots': (col_name, 'size'), f'mean_{col_name}': (col_name, 'mean'),
                    f'sd_{col_name}': (col_name, 'std')}
        stock_agg = plot_data.groupby(['age_bin', 'density_class', 'lead_species']).agg(**agg_dict).reset_index()
        stock_agg.dropna(subset=[f'mean_{col_name}'], inplace=True)
        stock_agg.to_csv(f"NFI_stock_{pool_name}.csv", index=False)
        print(f"Saved stock summary for {pool_name} to NFI_stock_{pool_name}.csv")

    # 6. Harmonize Soil Depth and Recalculate Soil C for Increments
    print("\n--- Harmonizing Soil Depth for Increment Calculation ---")
    max_depths = all_samples.groupby([COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']])[
        COLUMN_MAP['sample_lower_depth']].max().reset_index()
    max_depths.rename(columns={COLUMN_MAP['sample_lower_depth']: 'max_depth'}, inplace=True)

    # Find the minimum of the maximum depths for each plot (the common depth)
    plot_common_depths = max_depths.groupby(COLUMN_MAP['plot_id'])['max_depth'].transform('min')
    max_depths['common_depth'] = plot_common_depths

    # Filter samples to only include those within the common depth for each visit
    harmonized_samples = all_samples.merge(max_depths, on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']])
    harmonized_samples = harmonized_samples[
        harmonized_samples[COLUMN_MAP['sample_lower_depth']] <= harmonized_samples['common_depth']]

    # Recalculate soil C using only the harmonized sample set
    harmonized_soil_c = calculate_soil_carbon(harmonized_samples)
    harmonized_soil_c.rename(columns={'soil_c_mg_ha': 'soil_c_harmonized_mg_ha'}, inplace=True)

    # Merge harmonized soil C back into the main plot data
    plot_data = plot_data.merge(harmonized_soil_c, on=[COLUMN_MAP['plot_id'], COLUMN_MAP['visit_id']], how='left')
    plot_data['soil_c_harmonized_mg_ha'].fillna(0, inplace=True)

    # Recalculate TEC with harmonized soil for increment calculation
    plot_data['tec_harmonized_mg_ha'] = plot_data['live_c_mg_ha'] + plot_data['wd_c_mg_ha'] + plot_data[
        'soil_c_harmonized_mg_ha']

    # 7. Calculate and Aggregate Increments using Harmonized Data
    print("\n--- Calculating and Aggregating Detailed Increments ---")
    plot_data.sort_values([COLUMN_MAP['plot_id'], 'visit_year'], inplace=True)

    # Update the dictionary to use harmonized columns for soil and TEC
    pools_for_increment = {
        'AGB_C': 'agb_c_mg_ha', 'BGB_C': 'bgb_c_mg_ha', 'Live_C': 'live_c_mg_ha',
        'Soil_C': 'soil_c_harmonized_mg_ha', 'WD_C': 'wd_c_mg_ha', 'TEC': 'tec_harmonized_mg_ha'
    }

    for pool_col in pools_for_increment.values():
        plot_data[f'prev_{pool_col}'] = plot_data.groupby(COLUMN_MAP['plot_id'])[pool_col].shift(1)

    plot_data['prev_year'] = plot_data.groupby(COLUMN_MAP['plot_id'])['visit_year'].shift(1)

    increments = plot_data.dropna(subset=[f'prev_{p}' for p in pools_for_increment.values()] + ['prev_year']).copy()
    increments['delta_t'] = increments['visit_year'] - increments['prev_year']

    for pool_name, col_name in pools_for_increment.items():
        inc_col_name = f'inc_{col_name}_yr'
        increments[inc_col_name] = (increments[col_name] - increments[f'prev_{col_name}']) / increments['delta_t']

        increment_agg_dict = {'intervals': (inc_col_name, 'size'), f'mean_{inc_col_name}': (inc_col_name, 'mean'),
                              f'sd_{inc_col_name}': (inc_col_name, 'std')}

        increment_agg = increments.groupby(['age_bin', 'density_class', 'lead_species']).agg(
            **increment_agg_dict).reset_index()
        increment_agg.dropna(subset=[f'mean_{inc_col_name}'], inplace=True)

        output_filename = f"NFI_increment_{pool_name}.csv"
        increment_agg.to_csv(output_filename, index=False)
        print(f"Saved increment summary for {pool_name} to {output_filename}")

    print("\n--- Pipeline Finished ---")



if __name__ == '__main__':
    # To run this script:
    # 1. Make sure you have pandas installed (`pip install pandas`).
    # 2. Update the file paths in the DATA_PATHS dictionary at the top.
    # 3. Run the script from your terminal: `python your_script_name.py`

    cfg = Config()
    data_dir = os.path.join(cfg.data_dir, "GP_rounded_all_plots_NT", "imp")

    # --- Configuration & Constants ---

    # File paths to the raw NFI data tables.
    # The user should update these paths to match their local file locations.
    DATA_PATHS = {
        "ltp_header": data_dir + "/nt_gp_ltp_header.csv",
        "stp_header": data_dir + "/nt_gp_stp_header.csv",
        "ltp_tree": data_dir + "/nt_gp_ltp_tree.csv",
        "stp_tree": data_dir + "/nt_gp_stp_tree.csv",
        "ltp_spc": data_dir + "/nt_gp_ltp_tree_species_comp.csv",
        "stp_spc": data_dir + "/nt_gp_stp_tree_species_comp.csv",
        "wd_summary": data_dir + "/nt_gp_wd_summary.csv",
        "soil_horizon": data_dir + "/nt_gp_soil_horizon_desc.csv",
        "forest_floor": data_dir + "/nt_gp_for_flr_org_sample.csv",
        "soil_org": data_dir + "/nt_gp_soil_org_sample.csv",
        "soil_mineral": data_dir + "/nt_gp_soil_mineral_sample.csv",
    }

    # Output file names for the aggregated results.
    OUTPUT_STOCK_FILE =  data_dir + "/NFI_TierA_stock_by_stratum.csv"
    OUTPUT_INCREMENT_FILE = data_dir + "/NFI_TierA_annual_increment_by_stratum.csv"

    # --- Column Name Mapping ---
    # Maps conceptual names to the actual column names in the CSV files.
    COLUMN_MAP = {
        "plot_id": "nfi_plot",
        "visit_id": "meas_num",
        "visit_date": "meas_date",
        "stand_age": "site_age",
        "plot_area": "meas_plot_size",
        "lt_status": "lgtree_status",
        "st_status": "smtree_status",
        "lt_agb": "biomass_total",  # Above-ground biomass for large trees
        "st_agb": "smtree_biomass",  # Above-ground biomass for small trees
        "lt_dbh": "dbh",
        "spc_genus": "genus",
        "spc_species": "species",
        "spc_percent": "percent",
        "sample_upper_depth": "sample_upper",
        "sample_lower_depth": "sample_bottom"
    }

    # --- Analysis Parameters ---
    DENSITY_THRESHOLD_SPH = 900  # Stems per hectare for Sparse/Dense classification
    CARBON_FRACTION = 0.5  # Assumed carbon content of dry biomass
    AGE_BINS = [0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 500]
    AGE_LABELS = [f"{AGE_BINS[i]:03d}-{AGE_BINS[i + 1]:03d}" for i in range(len(AGE_BINS) - 1)]

    # Root-to-Shoot ratios for Below-Ground Biomass (BGB) estimation
    # Based on Canadian NFI defaults and IPCC guidelines.
    CONIFER_GENERA = ['PICEA', 'ABIES', 'PINUS', 'LARI', 'TSUGA', 'THUJA']
    RS_RATIOS = {
        'CONIFER': 0.20,
        'BROADLEAF': 0.26
    }

    # --- Data Processing Functions ---

    process_all_data()


