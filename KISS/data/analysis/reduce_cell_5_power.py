# Script to analyse the data from the cell 5 power test

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# Set the project path
project_path = Path("~/dev-02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')

# Different approach.Read in one TSV to a dataframe and o the analysis.
# Read in the data
data_path = project_path / 'output' / 'reduce_cell_5_power' / '2023_03_17' / 'reduce_cell_5_power'


