# YAP+
- YAP+ is a Python-based yield modeling and simulation tool for advanced packaging that supports yield analysis of arbitrary I/O pad layouts. Currently, the model is specifically designed for wafer-to-wafer (W2W) and die-to-wafer (D2W) hybrid bonding.
- A [GUI of YAP](http://nanocad.ee.ucla.edu/yap_gui/) and the [user guide video](https://youtu.be/8hiKIQ6C7ng) is available.
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
git clone -b yap+ https://github.com/nanocad-lab/YAP.git
cd ./YAP
```
2. Install dependencies:
```
pip install -r requirements.txt
```

# Usage

YAP+ provides Python entry points for one configuration-driven run and notebooks
for interactive analysis. The Python entry points read all experiment parameters
from `configs/config.yaml`; they do not override die size, pitch, defect density,
replica spacing, sample count, or simulation count.

## Python Entry Points


Use `--config` to specify a YAML file explicitly. These examples use the
configuration files included in the repository. Relative paths are resolved
from the directory where the command is executed:

```bash
python D2W/calculator_main.py --config D2W/configs/config.yaml
python D2W/simulator_main.py --config D2W/configs/config.yaml
python W2W/calculator_main.py --config W2W/configs/config.yaml
python W2W/simulator_main.py --config W2W/configs/config.yaml
```

The selected YAML file must contain the configuration section required by the
entry point. Run any command with `--help` to see its expected section and
default path.

Generated pad layouts and compatible calculator caches are stored under
the corresponding `D2W/pad_bitmap/` or `W2W/pad_bitmap/` directory.

## Run Controls

- `pad_layout_pattern` selects the pad block placement pattern. The supported
  default is `center`.
- `pad_block_dim_um` sets the physical pad block width and height in um. The
  derived `pad_block_size` is calculated by dividing it by `PITCH_um`.
- `reuse_dilation: false` makes a calculator run rebuild its particle-dilation
  cache. Set it to `true` only when the saved cache was produced by the same pad
  layout.
- `USE_RANDOM_SEED: true` uses `RANDOM_SEED` for reproducible modeling and
  simulation. Set it to `false` for a nondeterministic run.
- In simulation sections, `simulation_times` controls the number of Monte Carlo
  batches. D2W uses `NUM_DIES` per batch; W2W uses `NUM_WAFERS` per batch.
- `approximate_set: 1` performs full pad-by-pad overlay simulation. Other values
  use the boundary approximation.
- `redundant_logical_pad_dist` is expressed in pad-pitch units. For example, a
  value of `80` at a 10 um pitch represents 800 um physical spacing.

## Notebooks

Use `calculator_main.ipynb` and `simulator_main.ipynb` in `D2W/` or `W2W/` for
interactive plotting and parameter sweeps. Notebook sweep cells may intentionally
override loaded values; the Python entry points always execute exactly one run
from the selected configuration section.


# Paper Link
[YAP+: Pad-Layout-Aware Yield Modeling and Simulation for Hybrid Bonding](https://ieeexplore.ieee.org/document/11363225)



