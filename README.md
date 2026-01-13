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
2. Install dependencies:
```
pip install -r requirements.txt
```

# Usage
- Run the simulator and model for W2W hybrid bonding.
  ```
  cd W2W
  ```

  Example command to run the pad risk map calculator for W2W hybrid bonding
  
  <!-- `python pad_risk_map_calculator.py --config configs/HBM_footprint_A_config.yaml --mode w2w_modeling --bmap input/HBM_footprint_A.bmap  --criticality input/HBM_footprint_A_criticality.txt` -->
  ```python pad_risk_map_calculator.py --config configs/random_10x10_50.yaml --mode w2w_modeling --bmap input/random_10x10_50.bmap  --criticality input/random_10x10_50_criticality.txt```

  Example command to run the simulator main for W2W hybrid bonding

  <!-- `python simulator_main.py --config configs/HBM_footprint_A_config.yaml --mode w2w_simulation --bmap input/HBM_footprint_A.bmap  --criticality input/HBM_footprint_A_criticality.txt` -->
  ```python simulator_main.py --config configs/random_10x10_50.yaml --mode w2w_simulation --bmap input/random_10x10_50.bmap  --criticality input/random_10x10_50_criticality.txt```

- Run the simulator and model for D2W hybrid bonding.
  ```
  cd D2W
  ```
  Example command to run the pad risk map calculator for D2W hybrid bonding

  <!-- `python pad_risk_map_calculator.py --config configs/HBM_footprint_A_config.yaml --mode d2w_modeling --bmap input/HBM_footprint_A.bmap  --criticality input/HBM_footprint_A_criticality.txt` -->
  ```python pad_risk_map_calculator.py --config configs/random_10x10_50.yaml --mode d2w_modeling --bmap input/random_10x10_50.bmap  --criticality input/random_10x10_50_criticality.txt```

  Example command to run the simulator main for D2W hybrid bonding

  <!-- `python simulator_main.py --config configs/HBM_footprint_A_config.yaml --mode d2w_simulation --bmap input/HBM_footprint_A.bmap  --criticality input/HBM_footprint_A_criticality.txt` -->
  ```python simulator_main.py --config configs/random_10x10_50.yaml --mode d2w_simulation --bmap input/random_10x10_50.bmap  --criticality input/random_10x10_50_criticality.txt```

# Paper Link
To be continued...


