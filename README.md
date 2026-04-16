# YAP+
- YAP+ is a Python-based yield modeling and simulation tool for advanced packaging that supports yield analysis of arbitrary I/O pad layouts. Currently, the model is specifically designed for wafer-to-wafer (W2W) and die-to-wafer (D2W) hybrid bonding.
- A [GUI of YAP](http://nanocad.ee.ucla.edu:8081/yap_gui/) and the [user guide video](https://youtu.be/8hiKIQ6C7ng) is available.

Current active D2W design families in this repository are:
- `design_1_p5`
- `design_2_p10`
- `HBM_A`
- `HBM_B`

Legacy `design_1` / `design_2` assets are still present under `D2W/configs/old_configs/` or older handoff folders, but the examples below are written for the current active flow.

# File Structure
```
.
├── D2W/      # Code for D2W hybrid bonding
│   ├── configs/    # Golden and per-design configuration files
│   │   ├── GOLDEN.yaml
│   │   ├── design_1_p5/
│   │   ├── design_2_p10/
│   │   ├── HBM_A/
│   │   ├── HBM_B/
│   │   └── old_configs/
│   ├── input/      # Per-design 3dblox inputs, bump maps, and criticality files
│   │   ├── design_1_p5/
│   │   │   └── c10_r0_pg60_dm30/
│   │   │       ├── Center_IO/
│   │   │       ├── Edge_IO/
│   │   │       ├── Random_IO/
│   │   │       └── <chiplet_A>_to_<chiplet_B>_shared_nets.txt
│   │   ├── design_2_p10/
│   │   ├── HBM_A/
│   │   │   ├── Original/
│   │   │   ├── Center_IO/
│   │   │   ├── Edge_IO/
│   │   │   └── Random_IO/
│   │   ├── HBM_B/
│   │   └── old_bmap/
│   └── utils/      # Helper scripts for bump map / criticality processing
├── W2W/      # Code for W2W hybrid bonding
├── LICENSE
├── README.md
└── requirements.txt    # Requirements of Python packages
```

# Installation
1. Clone the repository
```
git clone -b yap+IO_assign https://github.com/Chen-Zhichao/YAP.git
cd ./YAP
```

2. (Optional) Create and activate a virtual environment:
```
conda create -n yap_env python=3.12
conda activate yap_env
```

3. Install dependencies:
```
pip install -r requirements.txt
```

# Usage
- Generate criticality file from bump map

  ```
  python D2W/utils/generate_criticality.py --force
  ```

  Generate both criticality profiles for explicit bump maps:

  ```
  python D2W/utils/generate_criticality.py --file D2W/input/design_1_p5/c10_r0_pg60_dm30/Center_IO/Compute_Small_From_Substrate_Silicon.bmap --profiles both --force
  python D2W/utils/generate_criticality.py --file D2W/input/design_2_p10/c10_r0_pg60_dm30/Center_IO/Compute_Large_0_From_Substrate_Organic.bmap --profiles both --force
  ```

  The supported profiles are:
  - `default`: replicated redundant nets tolerate `R-1` ESD failures and `R-1` mechanical failures
  - `esd_strict`: replicated redundant nets tolerate `0` ESD failures and `R-1` mechanical failures


- Run the simulator and model for D2W hybrid bonding.

  ```
  cd D2W
  ```

  Common variants are:
  - `Original`
  - `Center_IO`
  - `Edge_IO`   
  - `Random_IO`

  Example command to run the pad risk map calculator for D2W hybrid bonding for a single design

  ```
  python pad_risk_map_calculator.py --config configs/design_1_p5/design_1_p5.yaml --mode d2w_modeling --ds_name design_1_p5/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_1_p5/c10_r0_pg60_dm30/Center_IO --verbose

  python pad_risk_map_calculator.py --config configs/design_2_p10/design_2_p10.yaml --mode d2w_modeling --ds_name design_2_p10/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_2_p10/c10_r0_pg60_dm30/Center_IO --verbose
  ```

  Notes for large-pad-count analytical ESD maps:
  - ESD pad risk map generation now applies two automatic accelerations by default.
  - A coarse-grid ESD evaluation is computed and then interpolated back to the full pad map.
  - Candidate-pad pruning evaluates only pads near the deterministic first-touch edge when that pruning is safe.
  - Risk-map saving is also optimized: text `.risk.map` output is written with a vectorized writer, and per-mechanism risk-map PNGs are now saved by default.
  - `--plot` is only needed if you also want the extra interactive mechanism plots shown during modeling.
  - You can override the coarse-grid factor with `ESD_PAD_MAP_SUB_FACTOR` in the YAML.
  - If `ESD_PAD_MAP_SUB_FACTOR` is unset or `0`, a factor is chosen automatically from the active pad count.
  - Optional candidate-pruning knobs:
  - `ESD_ANALYTICAL_CANDIDATE_SIGMA_WINDOW`
  - `ESD_ANALYTICAL_CANDIDATE_MIN_PADS`
  - `ESD_ANALYTICAL_CANDIDATE_DISABLE_FRACTION`

  Example command to run the pad risk map calculator with the strict-ESD criticality profile

  ```
  python pad_risk_map_calculator.py --config configs/design_1_p5/design_1_p5.yaml --mode d2w_modeling --ds_name design_1_p5/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_1_p5/c10_r0_pg60_dm30/Center_IO --criticality-profile esd_strict --verbose

  python pad_risk_map_calculator.py --config configs/design_2_p10/design_2_p10.yaml --mode d2w_modeling --ds_name design_2_p10/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_2_p10/c10_r0_pg60_dm30/Center_IO --criticality-profile default --verbose

  python pad_risk_map_calculator.py --config configs/HBM_A/HBM_A.yaml --mode d2w_modeling --ds_name HBM_A/Center_IO --ds_dir input/HBM_A/Center_IO --verbose

  python pad_risk_map_calculator.py --config configs/HBM_A/HBM_A_overlay_pessimistic.yaml --mode d2w_modeling --ds_name HBM_A/Center_IO --ds_dir input/HBM_A/Center_IO --verbose

  ```

  Example command to run the pad risk map calculator for all variants of one or more designs

  ```
  ./run_design_pad_risk_maps.sh --ratio c10_r0_pg60_dm30 design_1_p5
  ./run_design_pad_risk_maps.sh --ratio c10_r0_pg60_dm30 design_1_p5 design_2_p10 HBM_A HBM_B
  ./run_design_pad_risk_maps.sh HBM_A HBM_B
  ```

  Example command to run pad risk map generation in parallel

  ```
  ./run_all_pad_risk_maps_parallel.sh --jobs 16 design_1_p5 design_2_p10 HBM_A HBM_B
  ./run_all_pad_risk_maps_parallel.sh --jobs 16 --skip-existing design_1_p5 design_2_p10 HBM_A HBM_B
  ```

  Notes:
  - Ratio-based designs such as `design_1_p5` and `design_2_p10` should be run with `--ratio`.
  - Current active ratios are `c0_r20_pg60_dm20`, `c5_r10_pg60_dm25`, and `c10_r0_pg60_dm30`.
  - `HBM_A` and `HBM_B` use direct variant folders and do not require a ratio.
  - The legacy wrapper `run_design_1_2_hbm_pad_risk_maps_parallel.sh` still exists in the repo but points to older design names and is not the recommended entry point for the current active datasets.

  Example command to run the simulator main for D2W hybrid bonding for a single design

  ```
  python simulator_main.py --config configs/design_1_p5/design_1_p5.yaml --mode d2w_simulation --ds_name design_1_p5/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_1_p5/c10_r0_pg60_dm30/Center_IO --criticality-profile default --verbose

  python simulator_main.py --config configs/design_2_p10/design_2_p10.yaml --mode d2w_simulation --ds_name design_2_p10/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_2_p10/c10_r0_pg60_dm30/Center_IO --criticality-profile default --verbose
  ```

  Example command to run the simulator with the strict-ESD criticality profile

  ```
  python simulator_main.py --config configs/design_1_p5/design_1_p5.yaml --mode d2w_simulation --ds_name design_1_p5/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_1_p5/c10_r0_pg60_dm30/Center_IO --criticality-profile esd_strict --verbose

  python simulator_main.py --config configs/design_2_p10/design_2_p10.yaml --mode d2w_simulation --ds_name design_2_p10/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_2_p10/c10_r0_pg60_dm30/Center_IO --criticality-profile default --verbose

  python simulator_main.py --config configs/HBM_A/HBM_A.yaml --mode d2w_simulation --ds_name HBM_A/Original --ds_dir input/HBM_A/Original --criticality-profile default --verbose

  python simulator_main.py --config configs/HBM_A/HBM_A_particle_pessimistic.yaml --mode d2w_simulation --ds_name HBM_A/Center_IO --ds_dir input/HBM_A/Center_IO --criticality-profile default --verbose

  python simulator_main.py --config configs/design_1_p5/design_1_p5.yaml --mode d2w_simulation --ds_name design_1_p5/c10_r0_pg60_dm30/Center_IO --ds_dir input/design_1_p5/c10_r0_pg60_dm30/Center_IO --criticality-profile default --save-failure-maps --verbose
  ```

  Example command to run D2W simulation for all variants of one or more designs

  ```
  ./run_design_simulations.sh --ratio c10_r0_pg60_dm30 design_1_p5
  ./run_design_simulations.sh --ratio c10_r0_pg60_dm30 --verbose design_1_p5 design_2_p10 HBM_A HBM_B
  ./run_design_simulations.sh HBM_A HBM_B
  ```

  Example command to run simulation experiments in parallel

  ```
  ./run_design_1_p5_design_2_p10_hbm_parallel.sh --jobs 16
  ./run_design_1_p5_design_2_p10_hbm_parallel.sh --jobs 16 --skip-existing
  ./run_design_1_p5_design_2_p10_hbm_parallel.sh --dry-run --jobs 4
  ```

  Example command to run the full `design_1_p5 / design_2_p10 / HBM_A / HBM_B` sweep used for result tables

  ```
  ./run_design_1_p5_design_2_p10_hbm_parallel.sh --jobs 16
  ./run_design_1_p5_design_2_p10_hbm_parallel.sh --dry-run --jobs 4
  ```

  Notes for the parallel simulation launcher:
  - Each experiment is forced to single-threaded execution.
  - Jobs sharing the same `ds_dir` are serialized; only different `ds_dir` values run in parallel.
  - `.bmap` inputs are no longer sorted in place during runtime. A sorted copy is created under `output/<ds_name>/temp/`.
  - Mechanical dishing caches are isolated by `ds_name + config + criticality_profile`, so different configs can run on the same `ds_dir` without sharing temp files.
  - Per-run generated interface YAML files are written under the design config folder (for example `D2W/configs/design_1_p5/`) with the `__<config_stem>__<criticality_profile>` suffix in the filename to avoid cross-run overwrites.
  - Runtime temp files are cleaned automatically after each experiment finishes.
  - Per-experiment logs are written to `output/<ds_name>/parallel_simulation__<config_stem>__<criticality_profile>.log`.
  - `run_design_1_p5_design_2_p10_hbm_parallel.sh` defaults to `design_1_p5` and `design_2_p10` across all active ratios with `Center_IO / Edge_IO / Random_IO`, plus `HBM_A` and `HBM_B` across `Original / Center_IO / Edge_IO / Random_IO`.
  - The legacy wrapper `run_design_1_2_hbm_simulations_parallel.sh` still exists in the repo but points to older design names and is not the recommended entry point for the current active datasets.

  Example commands for the current sensitivity-study helpers

  ```
  ./run_design1_p5_esd_sensitivity.sh
  ./run_design1_p5_esd_sensitivity_parallel.sh --jobs 8
  ./run_design2_p10_particle_sensitivity.sh
  ./run_design2_p10_particle_sensitivity_parallel.sh --jobs 16

  python D2W/utils/paper/plot_esd_sensitivity.py --input D2W/output/esd_sensitivity_check/esd_sensitivity_design_1_p5.tsv --output D2W/output/esd_sensitivity_check/esd_sensitivity_design_1_p5.png --ylim 85,100
  python D2W/utils/paper/plot_particle_sensitivity.py --input D2W/output/particle_sensitivity_design_2_p10.tsv --output D2W/output/particle_sensitivity_design_2_p10_bar.png
  ```

  Example command to package the current `design_1_p5 / design_2_p10 / HBM_A / HBM_B` configs and inputs for handoff

  ```
  tar -czf configs_input_design_1_p5_design_2_p10_HBM_A_HBM_B_20260414.tar.gz \
    D2W/configs/design_1_p5 D2W/configs/design_2_p10 D2W/configs/HBM_A D2W/configs/HBM_B \
    D2W/input/design_1_p5 D2W/input/design_2_p10 D2W/input/HBM_A D2W/input/HBM_B
  ```

# File Formats
**1. Bump Map (.bmap):**

   Format: `<instance> <bump_type> <x> <y> <port> <net>`

   Example: `Bump_0 uBUMP 115 1610 txdatasb txdatasb`

**2. Risk Map (.map):**

   Format: `<x> <y> <esd_failure_probability> <overlay_failure_probability> <particle_failure_probability> <mechanical_failure_probability>`

   Example: `115 1610 0.15 0.05 0.03 0.20`

   Note: Probabilities are float values between 0 and 1

   NOTE: ESD criticality is multiplied by esd_failure_probability.
         Mechanical criticality is multiplied by overlay_failure_probability, 
         particle_failure_probability, and mechanical_failure_probability.
         All four failure modes are considered in the optimization objective.

**3. Criticality (.txt):**

   Current Format: `<net1> [net2] [net3] ... <group_size> <tolerated_esd_failures> <tolerated_mechanical_failures>`
   
   Where:
   - `group_size`: Total number of pads/bumps in the redundancy group
   - `tolerated_esd_failures`: Number of ESD failures the group can tolerate before failing
   - `tolerated_mechanical_failures`: Number of mechanical failures the group can tolerate before failing

   Two filename variants are supported:
   - `*_criticality.txt`
     - Default profile
     - Replicated redundant signal nets tolerate `R-1` ESD failures and `R-1` mechanical failures
   - `*_criticality_esd_strict.txt`
     - Strict ESD profile
     - Replicated redundant signal nets tolerate `0` ESD failures and `R-1` mechanical failures
     - PG and dummy nets are unchanged relative to the default file

   Criticality values are calculated when reading the file:
   - esd_criticality = (group_size - tolerated_esd_failures) / group_size
   - mechanical_criticality = (group_size - tolerated_mechanical_failures) / group_size
   
   Examples:

   Single net with 5 pads, tolerates 4 ESD failures and 4 mechanical failures:
     `vccfwdio 5 4 4`
     (Results in esd_criticality = 0.2, mechanical_criticality = 0.2)
   
   Redundancy group with 4 pads, tolerates 1 ESD failure and 1 mechanical failure:
     `rxckRD rxckn rxckp rxtrk 4 1 1`
     (Results in esd_criticality = 0.75, mechanical_criticality = 0.75)
   
   Redundancy group with 34 pads, tolerates 2 ESD failures and 2 mechanical failures:
     `rxdata0 rxdata1 rxdata2 ... rxdata31 34 2 2`
     (Results in esd_criticality = 0.941, mechanical_criticality = 0.941)
   
   Legacy format (deprecated but still supported):
     `<net> <esd_criticality> <mechanical_criticality>`
     Example: `txdatasb 0.8 0.7`
   
   Note: 
   - Criticality values range from 0 (non-critical) to 1 (critical)
   - Values between 0 and 1 indicate redundancy where multiple failures can be tolerated
   - Multiple nets listed on the same line form a redundancy group sharing the same failure tolerance
   - Each net name should appear only once in the entire file
   - See UCIe_advanced_criticality.txt for a complete example of the current format

**4. 3dbv File (.3dbv):**

   Input file in 3dblox format. This file contains info including the die size and the path of 3dbf file. 

**5. 3dbf File (.3dbf):**

   Input file in 3dblox format. This file contains info including bump pitch and bump size.


# Output
**1.<interface>_risk__<config_stem>__<criticality_profile>.map**

  The risk map of the interface in a text format. Each line corresponds to a pad and contains the x and y coordinates of the pad, followed by the failure probabilities of different failure mechanisms.
  The `__<config_stem>__<criticality_profile>` suffix distinguishes baseline and each pessimistic case.

**1a.<interface>_<mechanism>_risk_map__<config_stem>__<criticality_profile>.png**

  Per-mechanism pad risk maps are written by default for `pad_risk_map_calculator.py`.
  `--plot` only controls the extra interactive plots shown during modeling.

**2.assembly_yield_summary__<config_stem>__<criticality_profile>.txt**

  The simulation summary in a text format. It includes:
  - simulation settings
  - runtime information
  - overall assembly yield
  - per-interface yield

**3.assembly_yield_per_interface__<config_stem>__<criticality_profile>.txt**

  The simulated assembly yield of each interface in a text format. Each line corresponds to an interface and contains the interface name and the simulated yield.

**4.assembly_fail_map_per_interface_dict__<config_stem>__<criticality_profile>.npz**

  The average failure count (across all simulation samples) of each pad in a pad map format for all failure mechanisms.
  This file is only written when both `--verbose` and `--save-failure-maps` are enabled.

**5.assembly_fail_vec_per_interface_dict__<config_stem>__<criticality_profile>.npz**

  The failure vector of the survival scenario of each die samples for all failure mechanisms.
  This file is written in verbose simulation mode even when failure-map PNG/NPZ saving is disabled.
  
  Example: die A, B, C, D, and E are simulated. A, B and D pass, and C and E fail. The failure vector of this failure mechanism is : `0, 0, 1, 0, 1`.

**6.simulation_failure_map_<mechanism>__<config_stem>__<criticality_profile>.png**

  Per-interface simulation failure heatmaps for:
  - `overlay`
  - `particle`
  - `mechanical`
  - `ESD`
  - `overall`
  
  These PNGs are only written when `--save-failure-maps` is enabled for `simulator_main.py`.

# Generator Utilities
Four helper scripts are provided to quickly generate starter files for testing:
  - `assign_bump_names.py`: Assign net names and port names to bump maps to raw bump maps.
  - `generate_criticality.py`: Generate criticality files from bump maps.

# Paper Link
To be continued...
