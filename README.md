# AfforestationGCBM
**Carbon Capture Capacity Estimation of Taiga Reforestation and Afforestation at the 
Western Boreal Edge Using Spatially Explicit Carbon Budget Modeling**

## Abstract
Canada’s northern boreal region offers significant potential for climate change mitigation through 
strategic tree planting. In this work, we employ Monte Carlo simulations alongside 
spatially explicit carbon budget modeling, integrating satellite forest inventory data 
and probabilistic fire regime scenarios, to project the evolution of total ecosystem 
carbon (TEC) from 2025 to 2100. Our analysis shows that reforesting historically 
forested lands and longer fire return intervals markedly enhance TEC, while high seedling 
mortality severely curtails carbon gains. Temperature emerges as the dominant control over TEC, 
with precipitation playing a secondary role. Overall, afforestation at the north-western 
boreal edge could sequester approximately 3.88 G tonnes of CO₂e over 75 years on around 
6.4 million hectares, particularly in the NT-Taiga Shield West zone. 
These findings underscore the importance of site preparation, species selection, 
wildfire management, and future climate conditions, and they call for further research 
into model refinement, economic feasibility, and associated regional impacts.

This code repository consists of the python files used for pre-processing, post-processing, 
visualization, the Windows batch file used to run GCBM, and some data files. 
Note that GCBM currently runs only on Windows, the remaining Python scripts are platform-independent 
and can be executed on any machine.

**Preprint:** [Afforestation North-Western Boreal](https://arxiv.org/abs/2503.12331)

### Prerequisites
- Windows to run GCBM 
- Python 3, [Conda](https://docs.conda.io/en/latest/) 

## Installation and Usage 

1. **Clone and Install Repository:**
   ```bash
   git clone git@github.com:kevinbdsouza/AfforestationGCBM.git
   cd AfforestationGCBM
   conda env create -f environment.yml
   conda activate gcbm
   ```
   
2. **Download and Install GCBM:** Follow instructions given at [GCBM](https://natural-resources.canada.ca/climate-change/climate-change-impacts-forests/generic-carbon-budget-model). 
   Install GCBM dependencies inside the conda environment. 

3. **Get Data and Prepare Scenarios:** 
   1. Download the [NTEMS-SBFI](https://opendata.nfis.org/mapserver/nfis-change_eng.html) data. 
      Export the tiles given in `cfg.tiles` and transform them to `EPSG:4326` `shp` files using 
      the `transform_shp_files()` function in `src/util_functions.py`. Use the 
      `store_bbox_tiles()` function in `src/util_functions.py` to create `bounds.csv`. 
   2. Use the `create_combined_df()` function in `src/utils.py` to create dataframes 
      grid cells in the chosen lat-lon grid. Then use the `compile_grid_info()` function
      to compile data for the whole grid along with `MAT`(`data/mat_grid.npy`) and 
      `PCP` (`data/pcp_grid.npy`). Climate data downloaded from [ClimateDataCA](https://climatedata.ca/download/?var=tg_mean).
   3. Generate yield curves using `create_yield_curves()` function `src/util_functions.py`. 
      One set of yield curves are given in `data/yield_mean.csv` and Timberworks Inc. 
      yield curves from Northwest Territories are given in `data/nt_yields.csv`. 
   3. Use the `create_scenarios()` function in `src/utils.py` to create scenarios. 
      Set the `cfg.exps_dir` same as `EXPS_DIR` in `run_gcbm.bat`. 
   4. Set all other directories in `src/config.py` as needed when 
      running all of the above. 

4. **Run GCBM:** Make sure to set the `GCBM_FOLDER_NAME` and all other paths 
   in `run_gcbm.bat` as per your system. Run the `run_gcbm.bat` file. 

5. **Analyze and Visualize Results:** Use the `plot_db()` function in `plot_utils.py` and 
   helper functions in `util_functions.py`. 

## Additional Resources
- For more details on how to run GCBM, see the training videos by [moja global](https://www.youtube.com/playlist?list=PL_WECUlMWiUmZYoPHNn6RnMSia5Naj5gE).

## License
This project is licensed under the MIT License. See the [MIT LICENSE](https://github.com/kevinbdsouza/AfforestationGCBM/blob/main/LICENSE) file for more details.