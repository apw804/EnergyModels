# Energy_models-0.1

This is the main repo for energy models that are developed for the QMUL-BT collaboration. Collaborators include @kishansthankiya and @keithbriggs, 
respectively.

Each folder in this repo specifies a different power model. Any additional dependencies will be outlined in the respective folder documentation.
Each model features a self-test if no arguements are passed to it. One demo script will be provided in each folder to demonstrate the power model.

## Dependencies

The power models rely on the following modules from the AIMM_simulator-1.0 package:

AIMM_simulator-1.0
├── AIMM_simulator_core_17.py
├── NR_5G_standard_functions_00.py
├── README.md
├── UMa_pathloss_model_01.py
├── UMi_pathloss_model_00.py




These can be cloned via SSH provided you have the correct access rights to the below repo:

```console
git clone git@github.com:apw804/AIMM_simulator-1.0.git
```
Any 

## Setup

1. Ensure that the dependency files are located in your PATH variable.
1. Open a Terminal and ensure you are in the directory of the energy model you wish to run.
1. The output from the model you run should create a .csv file which can then be used for future modelling purposes.
