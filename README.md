# YAP+
- YAP+ is a Python-based yield modeling and simulation tool for advanced packaging that supports yield analysis of arbitrary I/O pad layouts. Currently, the model is specifically designed for wafer-to-wafer (W2W) and die-to-wafer (D2W) hybrid bonding.
- A [GUI of YAP](http://nanocad.ee.ucla.edu:8081/yap_gui/) and the [user guide video](https://youtu.be/8hiKIQ6C7ng) is available.
# File Structure
```
.
├── D2W/      # Code for D2W hybrid bonding
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
  python utils/generate_criticality.py input/design_0/CPU_From_interposer.bmap
  ```


- Run the simulator and model for W2W hybrid bonding.
  ```
  cd W2W
  ```

  Example command to run the pad risk map calculator for W2W hybrid bonding

  ```
  python pad_risk_map_calculator.py --config configs/design_0.yaml --mode w2w_modeling --ds_dir input/design_0 --bmap input/design_0/CPU_From_interposer.bmap --criticality input/design_0/CPU_From_interposer_criticality.txt --verbose
  ```

  Example command to run the simulator main for W2W hybrid bonding

  ```
  python simulator_main.py --config configs/design_0.yaml --mode w2w_simulation --ds_dir input/design_0 --bmap input/design_0/CPU_From_interposer.bmap --criticality input/design_0/CPU_From_interposer_criticality.txt --verbose
  ```

- Run the simulator and model for D2W hybrid bonding.
  ```
  cd D2W
  ```

  Example command to run the pad risk map calculator for D2W hybrid bonding

  ```
  python pad_risk_map_calculator.py --config configs/design_0.yaml --mode d2w_modeling --ds_dir input/design_0 --bmap input/design_0/CPU_From_interposer.bmap --criticality input/design_0/CPU_From_interposer_criticality.txt --verbose
  ```

  Example command to run the simulator main for D2W hybrid bonding

  ```
  python simulator_main.py --config configs/design_0.yaml --mode d2w_simulation --ds_dir input/design_0 --bmap input/design_0/CPU_From_interposer.bmap --criticality input/design_0/CPU_From_interposer_criticality.txt --verbose
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
**1.assembly_fail_map_dict.npz**

  The average failure count (across all simulation samples) of each pad in a pad map format for all failure mechanisms. The visualization will be generated by the simulation.

**2.assembly_fail_vec_dict.npz**

  The failure vector of the survival scenario of each die samples for all failure mechanisms. 
  
  Example: die A, B, C, D, and E are simulated. A, B and D pass, and C and E fail. The failure vector of this failure mechanism is : `0, 0, 1, 0, 1`.

# Generator Utilities
Four helper scripts are provided to quickly generate starter files for testing:
  - `generate_random_bump_map.py`: Generate random bump maps with power, ground, signal, and dummy bumps
  - `generate_criticality.py`: Generate criticality files from bump maps

# Paper Link
[YAP+: Pad-Layout-Aware Yield Modeling and Simulation for Hybrid Bonding](https://arxiv.org/abs/2511.05506)



