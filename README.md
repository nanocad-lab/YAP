# YAP
- YAP is a python-based yield modeling and simulation tool for advanced packaging. Currently, the model is specifically designed for wafer-to-wafer (W2W) and die-to-wafer (D2W) hybrid bonding.
- A GUI of YAP is available at http://nanocad.ee.ucla.edu:8081/yap_gui/.
# File Structure
```
.
├── D2W/      # Code for D2W hybrid bonding
├── W2W/      # Code for W2W hybrid bonding
├── LICENSE
├── README.md
└── requirements.txt    # Requirements of python packages
```

# Installation
1. Clone the repository
```
git clone https://github.com/nanocad-lab/YAP.git
cd ./YAP
```
2. Install dependencies:
```
pip install -r requirements.txt
```

# Usage
- Run the simulator and model for W2W hybrid bonding.
  ```
  cd ./YAP/W2W
  ```
  Execute the `simulator_main.ipynb` for the simulation and `calculator_main.ipynb` for the modeling.
- Run the simulator and model for D2W hybrid bonding.
  ```
  cd ./YAP/D2W
  ```
  Execute the `simulator_main.ipynb` for the simulation and `calculator_main.ipynb` for the modeling.


# Paper Link
This project is associated with the paper: _YAP: Yield Modeling and Simulation for Advanced Packaging_. The paper has been accepted by DAC 2025 and is currently available at https://nanocad.ee.ucla.edu/wp-content/papercite-data/pdf/c133.pdf. This paper summarizes the major failure mechanisms that induce yield loss for the hybrid bonding process, including overlay errors, particle defects, and Cu recess variations, etc. The paper introduced a near-analytical yield modeling tool and a simulator for multiple failure mechanisms.
