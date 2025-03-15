"""
Config class that encapsulates all configuration parameters,
directory paths, species mappings, simulation settings, and other related parameters.
"""

import os
import numpy as np


class Config:
    """
    Configuration class for simulation settings.

    This class holds all configuration parameters, including geographical coordinates,
    simulation time settings, directory paths, species mappings, ecozone definitions,
    fire/yield parameters, and other miscellaneous settings required by the simulation.
    """

    def __init__(self) -> None:
        """Initialize all configuration parameters."""
        self._init_tiles_and_coordinates()
        self._init_simulation_parameters()
        self._init_flags()
        self._init_directories()
        self._init_species_mappings()
        self._init_ecozones_and_soil()
        self._init_fire_yield_parameters()
        self._init_tables_and_stratification()
        self._init_miscellaneous()

    def _init_tiles_and_coordinates(self) -> None:
        """Initialize tile IDs and coordinate arrays."""
        self.tiles = [
            943, 944, 905, 904, 865, 866, 906, 827, 867, 868, 828, 829, 789, 790, 791, 751, 752,
            792, 753, 713, 714, 754, 715, 675, 676, 636, 637, 677, 678, 638, 598, 599, 639, 679,
            640, 600, 560, 561, 601, 602, 562, 563, 523, 603, 564, 524, 525, 485, 486, 487, 448
        ]
        self.lats = np.round(np.linspace(56.75, 70.75, 232), 2)
        self.lons = np.round(np.linspace(-141.25, -90.75, 816), 2)
        self.lats_n = np.round(np.arange(56.75, 70.75, 0.002), 3)
        self.lons_n = np.round(np.arange(-141.25, -90.75, 0.002), 3)

    def _init_simulation_parameters(self) -> None:
        """Initialize simulation time parameters and Monte Carlo iterations."""
        self.aff_year = 2025
        self.sim_interval = (2015, 2100)
        self.mc_iters = 2000

    def _init_flags(self) -> None:
        """Initialize simulation flags and mode settings."""
        self.run_baseline = False
        self.gen_mortality = False
        self.mortality_year = 2020
        self.hist_dist_type = "Natural succession"  # Options: Wildfire, Natural succession
        self.hist_land_class = "NFL"  # Options: FL, CL, NFL, GL
        self.fri_mode = "weibull"  # Options: list, weibull
        self.fire_frac_mode = "historic"  # Options: historic, van_wagner, custom
        self.exp_yield_perc = "mean"
        self.fire_assignment = "binom"  # Options: binom, burn_fraction

    def _init_directories(self) -> None:
        """Initialize directory paths for data storage and input/output files."""
        cur_dir = os.getcwd()
        self.main_dir = os.path.abspath(os.path.join(cur_dir, ".."))
        self.data_dir = os.path.join(self.main_dir, "data")
        self.input_db_dir = os.path.join(self.main_dir, "input_database")
        self.store_dir = os.path.join(self.main_dir, "store")
        self.store_bck_dir = os.path.join(self.main_dir, "store_bck")
        self.plot_dir = os.path.join(self.data_dir, "plots_tmp")
        self.raw_dir = os.path.join(self.main_dir, "layers", "raw")
        self.exps_dir = os.path.join(self.raw_dir, "exps")
        self.exps_bck_dir = os.path.join(self.raw_dir, "bck")
        self.land_cover_dir = os.path.join(self.data_dir, "land_cover")
        self.nfi_shp_dir = os.path.join(self.land_cover_dir, "shapefiles")
        self.transf_dir = os.path.join(self.land_cover_dir, "transformed")
        self.exp_name = ""

    def _init_species_mappings(self) -> None:
        """Initialize species group and name mapping dictionaries."""
        self.species_group_dict = {
            "Not stocked": 0, "ABIE.AMA": 1, "ABIE.BAL": 5, "ABIE.LAS": 5, "ACER.MAC": 7,
            "ACER.RUB": 7, "ACER.SAH": 7, "ALNU.INC": 7, "ALNU.RUB": 7, "BETU.ALL": 7,
            "BETU.PAP": 6, "CHAM.NOO": 2, "FRAX.NIG": 7, "LARI.LAR": 7, "LARI.OCC": 7,
            "PICE.ABI": 4, "PICE.ENG": 4, "PICE.GLA": 4, "PICE.MAR": 4, "PICE.RUB": 4,
            "PICE.SIT": 4, "PINU.ALB": 3, "PINU.BAN": 3, "PINU.CON": 3, "PINU.PON": 3,
            "PINU.RES": 3, "PINU.STR": 3, "POPU.BAL": 6, "POPU.GRA": 6, "POPU.TRE": 6,
            "PSEU.MEN": 1, "QUER.RUB": 7, "THUJ.OCC": 2, "THUJ.PLI": 2, "TSUG.CAN": 2,
            "TSUG.HET": 2, "TSUG.MER": 2, "ULMU.AME": 7
        }

        self.species_cm_name_dict = {
            "Not stocked": "Nope", "ABIE.AMA": "Amabilis fir", "ABIE.BAL": "Balsam fir",
            "ABIE.LAS": "Subalpine fir (or alpine fir)",
            "ACER.MAC": "Bigleaf maple", "ACER.RUB": "Red maple", "ACER.SAH": "Sugar maple",
            "ALNU.INC": "Alder", "ALNU.RUB": "Red alder", "BETU.ALL": "Yellow birch",
            "BETU.PAP": "White birch", "CHAM.NOO": "Cedar", "FRAX.NIG": "Black ash",
            "LARI.LAR": "Tamarack", "LARI.OCC": "Western larch", "PICE.ABI": "Norway spruce",
            "PICE.ENG": "Engelmann spruce", "PICE.GLA": "White spruce", "PICE.MAR": "Black spruce",
            "PICE.RUB": "Red spruce", "PICE.SIT": "Sitka spruce", "PINU.ALB": "Whitebark pine",
            "PINU.BAN": "Jack pine", "PINU.CON": "Lodgepole pine", "PINU.PON": "Ponderosa pine",
            "PINU.RES": "Red pine", "PINU.STR": "Eastern white pine", "POPU.BAL": "Balsam poplar",
            "POPU.GRA": "Largetooth aspen", "POPU.TRE": "Trembling aspen", "PSEU.MEN": "Douglas-fir - Genus type",
            "QUER.RUB": "Red oak", "THUJ.OCC": "Eastern white-cedar", "THUJ.PLI": "Western redcedar",
            "TSUG.CAN": "Eastern hemlock", "TSUG.HET": "Western hemlock", "TSUG.MER": "Mountain hemlock",
            "ULMU.AME": "White elm"
        }

        # Generate a dictionary mapping common names to scientific names.
        self.species_sci_name_dict = {v: k for k, v in self.species_cm_name_dict.items()}

    def _init_ecozones_and_soil(self) -> None:
        """Initialize ecozone species distributions and soil type mappings."""
        self.species_ecozones = {
            "Taiga Cordillera": {"Black spruce": 85, "Lodgepole pine": 15},
            "Boreal Cordillera": {"Lodgepole pine": 37, "Black spruce": 36, "Subalpine fir": 33},
            "Montane Cordillera": {"Subalpine fir": 31, "Lodgepole pine": 27, "Engelmann spruce": 21,
                                   "Douglas-fir": 21},
            "Boreal Plains": {"Black spruce": 43, "Trembling aspen": 40, "Lodgepole pine": 17},
            "Taiga Plains": {"Black spruce": 81, "Trembling aspen": 13, "Lodgepole pine": 16},
            "Taiga Shield West": {"Black spruce": 73, "Lodgepole pine": 27},
            "Boreal Shield West": {"Black spruce": 74, "Trembling aspen": 14, "Lodgepole pine": 12}
        }
        self.soil_type_dict = {
            "Taiga Cordillera": "Brunisolic", "Boreal Cordillera": "Brunisolic",
            "Montane Cordillera": "Podzolic", "Boreal Plains": "Brunisolic",
            "Taiga Plains": ["Cryosolic", "Luvisolic (W. Canada)"],
            "Taiga Shield West": ["Cryosolic", "Brunisolic"],
            "Boreal Shield West": "Podzolic"
        }

    def _init_fire_yield_parameters(self) -> None:
        """Initialize fire return intervals, yield percentages, and yield coefficients."""
        self.dom_soil_perc = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        self.aff_perc = [0.2, 0.4, 0.6, 0.8, 1]
        self.fire_return_interval = [30, 60, 100, 200, 500, 800, 1000, 1500, 2000]
        self.fire_frac = 0.01
        self.weibull_scale = {
            "Taiga Plains": np.arange(100, 200, 5),
            "Taiga Shield West": np.arange(500, 700, 5)
        }
        self.weibull_shape = np.arange(1.05, 1.61, 0.05)

        self.yield_percs_mat = [-0.4, -0.2, 0, 0.2, 0.4, 0.4, 0.2, -0.2, -0.4]
        self.yield_percs_pcp = [-0.4, -0.2, 0, 0.2, 0.4, -0.4, -0.2, 0.2, 0.4]
        self.yield_perc_names = [
            "nm_4_np_4", "nm_2_np_2", "mean", "pm_2_pp_2", "pm_4_pp_4",
            "pm_4_np_4", "pm_2_np_2", "nm_2_pp_2", "nm_4_pp_4"
        ]
        self.yield_coeffs = {
            0: [0, 0, 0, 0, 0, 0, 0],
            1: [5.7, 0.0636, -0.0001, -173.859, 14.5291, 0, 1.0423],
            2: [5.7, 0.0636, -0.0001, -173.859, 14.5291, 0, 1.0423],
            3: [6.4755, 0.1271, -0.0008, -67.4993, 2.6486, 0.0119, 1.0777],
            4: [6.443, 0.0981, -0.0013, -37.4046, 7.5551, 0.0362, 1.2127],
            5: [4.8421, 0, 0.0007, -74.8932, 4.3921, 0, 1.2453],
            6: [6.6358, 0, -0.0004, -55.6634, 0.8537, 0, 1.0289],
            7: [6.605, 0, -0.0009, -47.1154, 1.6253, 0, 1.0819],
        }

    def _init_tables_and_stratification(self) -> None:
        """Initialize table names and stratification mapping."""
        self.tables = [
            "v_age_indicators", "v_disturbance_fluxes", "v_disturbance_indicators",
            "v_error_indicators", "v_flux_indicator_aggregates",
            "v_flux_indicator_aggregates_density", "v_flux_indicators",
            "v_flux_indicators_density", "v_pool_indicators", "v_stock_change_indicators",
            "v_stock_change_indicators_density", "v_total_disturbed_areas"
        ]
        self.strat_map = {
            "DE_H_G": {"sp": "Deciduous", "density": "High", "sq": "Good"},
            "DE_H_M": {"sp": "Deciduous", "density": "High", "sq": "Medium"},
            "DE_H_P": {"sp": "Deciduous", "density": "High", "sq": "Poor"},
            "DE_L_G": {"sp": "Deciduous", "density": "Low", "sq": "Good"},
            "DE_L_M": {"sp": "Deciduous", "density": "Low", "sq": "Medium"},
            "DE_L_P": {"sp": "Deciduous", "density": "Low", "sq": "Poor"},
            "MX_H_G": {"sp": "Mixed Wood", "density": "High", "sq": "Good"},
            "MX_H_M": {"sp": "Mixed Wood", "density": "High", "sq": "Medium"},
            "MX_H_P": {"sp": "Mixed Wood", "density": "High", "sq": "Poor"},
            "MX_L_G": {"sp": "Mixed Wood", "density": "Low", "sq": "Good"},
            "MX_L_M": {"sp": "Mixed Wood", "density": "Low", "sq": "Medium"},
            "MX_L_P": {"sp": "Mixed Wood", "density": "Low", "sq": "Poor"},
            "PN_H_G": {"sp": "Pine", "density": "High", "sq": "Good"},
            "PN_H_M": {"sp": "Pine", "density": "High", "sq": "Medium"},
            "PN_H_P": {"sp": "Pine", "density": "High", "sq": "Poor"},
            "PN_L_G": {"sp": "Pine", "density": "Low", "sq": "Good"},
            "PN_L_M": {"sp": "Pine", "density": "Low", "sq": "Medium"},
            "PN_L_P": {"sp": "Pine", "density": "Low", "sq": "Poor"},
            "SB_H_G": {"sp": "Black Spruce", "density": "High", "sq": "Good"},
            "SB_H_M": {"sp": "Black Spruce", "density": "High", "sq": "Medium"},
            "SB_H_P": {"sp": "Black Spruce", "density": "High", "sq": "Poor"},
            "SB_L_G": {"sp": "Black Spruce", "density": "Low", "sq": "Good"},
            "SB_L_M": {"sp": "Black Spruce", "density": "Low", "sq": "Medium"},
            "SB_L_P": {"sp": "Black Spruce", "density": "Low", "sq": "Poor"},
            "SW_H_G": {"sp": "White Spruce", "density": "High", "sq": "Good"},
            "SW_H_M": {"sp": "White Spruce", "density": "High", "sq": "Medium"},
            "SW_H_P": {"sp": "White Spruce", "density": "High", "sq": "Poor"},
            "SW_L_G": {"sp": "White Spruce", "density": "Low", "sq": "Good"},
            "SW_L_M": {"sp": "White Spruce", "density": "Low", "sq": "Medium"},
            "SW_L_P": {"sp": "White Spruce", "density": "Low", "sq": "Poor"}
        }

    def _init_miscellaneous(self) -> None:
        """Initialize miscellaneous mappings and lists."""
        self.species_nt_yld_map = {
            "Not stocked": "Nope", "ABIE.AMA": "Mixed Wood", "ABIE.BAL": "Mixed Wood",
            "ABIE.LAS": "Mixed Wood", "ACER.MAC": "Deciduous", "ACER.RUB": "Deciduous",
            "ACER.SAH": "Deciduous", "ALNU.INC": "Deciduous", "ALNU.RUB": "Deciduous",
            "BETU.ALL": "Deciduous", "BETU.PAP": "Deciduous", "CHAM.NOO": "Mixed Wood",
            "FRAX.NIG": "Deciduous", "LARI.LAR": "Deciduous", "LARI.OCC": "Deciduous",
            "PICE.ABI": "Black Spruce", "PICE.ENG": "Black Spruce", "PICE.GLA": "White Spruce",
            "PICE.MAR": "Black Spruce", "PICE.RUB": "Black Spruce", "PICE.SIT": "Black Spruce",
            "PINU.ALB": "Pine", "PINU.BAN": "Pine", "PINU.CON": "Pine", "PINU.PON": "Pine",
            "PINU.RES": "Pine", "PINU.STR": "Pine", "POPU.BAL": "Deciduous", "POPU.GRA": "Deciduous",
            "POPU.TRE": "Deciduous", "PSEU.MEN": "Mixed Wood", "QUER.RUB": "Deciduous",
            "THUJ.OCC": "Mixed Wood", "THUJ.PLI": "Mixed Wood", "TSUG.CAN": "Mixed Wood",
            "TSUG.HET": "Mixed Wood", "TSUG.MER": "Mixed Wood", "ULMU.AME": "Deciduous"
        }
        self.density_list = ["High", "Low"]
        self.site_quality_list = ["Good", "Medium", "Poor"]
