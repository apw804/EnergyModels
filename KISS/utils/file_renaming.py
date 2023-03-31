
# Quick script to rename files in a directory

from pathlib import Path


project_path = Path("~/dev_02/EnergyModels/KISS").expanduser().resolve()
project_path_str = str(project_path)
print(f'Project path:{project_path}')
data_path = project_path / 'data' / 'output' / 'switch_n_cells_off' / '2023_03_27'
s1co_data = data_path / '01_cells_off'
s2co_data = data_path / '02_cells_off'
s3co_data = data_path / '03_cells_off'
s4co_data = data_path / '04_cells_off'
s5co_data = data_path / '05_cells_off'

def rename_files(directory, new_char, test: bool = True):
    # For each file in the directory, rename the file
    for i in directory.glob('*.tsv'):
        # Get the file name
        file_name = i.name
        # Split the file name on the underscore
        file_name_split = file_name.split('_')
        # Get the first part of the file name
        file_name_1 = file_name_split[0]
        # Replace the second character with a new character
        file_name_1 = file_name_1[:1] + new_char + file_name_1[2:]
        # Get the seocnd part of the file name
        file_name_2 = file_name_split[1]
        # Get the third part of the file name
        file_name_3 = file_name_split[2]
        # Join the parts of the file name together
        new_file_name = file_name_1 + '_' + file_name_2 + '_' + file_name_3 + '.tsv'
        # Rename the file if it isn't a test
        if not test:
           i.rename(i.parent / new_file_name)
        print(f'New file name: {new_file_name}')

rename_files(s5co_data, '5', test=False)
