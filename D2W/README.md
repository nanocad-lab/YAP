Go to D2W directory
`cd D2W/`
Example command to run the spatial correlation coefficients for D2W hybrid bonding
`python spatial_correlation_coefficients_main.py --config configs/HBM_footprint_A_config.yaml --mode d2w_simulation --bmap input/HBM_footprint_A.bmap`
Example command to run the pad risk map calculator for D2W hybrid bonding
`python pad_risk_map_calculator.py --config configs/HBM_footprint_A_config.yaml --mode d2w_modeling --bmap input/HBM_footprint_A.bmap`
Example command to run the simulator main for D2W hybrid bonding
`python simulator_main.py --config configs/HBM_footprint_A_config.yaml --mode d2w_simulation --bmap input/HBM_footprint_A.bmap`