Go to W2W directory

`cd W2W/`


Example command to run the pad risk map calculator for W2W hybrid bonding

<!-- `python pad_risk_map_calculator.py --config configs/HBM_footprint_A_config.yaml --mode w2w_modeling --bmap input/HBM_footprint_A.bmap  --criticality input/HBM_footprint_A_criticality.txt` -->
`python pad_risk_map_calculator.py --config configs/random_10x10_50.yaml --mode w2w_modeling --bmap input/random_10x10_50.bmap  --criticality input/random_10x10_50_criticality.txt`

Example command to run the simulator main for W2W hybrid bonding

<!-- `python simulator_main.py --config configs/HBM_footprint_A_config.yaml --mode w2w_simulation --bmap input/HBM_footprint_A.bmap  --criticality input/HBM_footprint_A_criticality.txt` -->
`python simulator_main.py --config configs/random_10x10_50.yaml --mode w2w_simulation --bmap input/random_10x10_50.bmap  --criticality input/random_10x10_50_criticality.txt`