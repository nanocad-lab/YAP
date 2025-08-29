# YAP+
- YAP+ is a Python-based yield modeling and simulation tool for advanced packaging that supports yield analysis of arbitrary I/O pad layouts. Currently, the model is specifically designed for wafer-to-wafer (W2W) and die-to-wafer (D2W) hybrid bonding.
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
git clone -b yaplus https://github.com/nanocad-lab/YAP.git
cd ./YAPlus
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
  Execute the `simulator_main.ipynb` for the simulation and `calculator_main.ipynb` for the modeling.
- Run the simulator and model for D2W hybrid bonding.
  ```
  cd D2W
  ```
  Execute the `simulator_main.ipynb` for the simulation and `calculator_main.ipynb` for the modeling.


# Paper Link
To be continued...
