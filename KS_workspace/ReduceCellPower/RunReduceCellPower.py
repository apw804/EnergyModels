# Script for multiple runs of ReduceCellPower12.py
# Kishan Sthankiya
from ReduceCellPower_12 import test_01

script = "ReduceCellPower_12.py"

# Define the arguments
seeds = [i for i in range(1000)]
# the_variable_you_want_to_change = [i for i in range()]

# set the args to run
seed_arg = ' seed= '

# run the code using a loop
for i in seeds:
    arg = seed_arg + str(i)
    sys.argv = arg
    os.system()
