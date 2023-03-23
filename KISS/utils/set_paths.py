from pathlib import Path


def set_project_path(project_path_str: str = "~/dev-02/EnergyModels/KISS"):
    # Set the project path
    project_path = Path(project_path_str).expanduser().resolve()
    project_path_str = str(project_path)
    print(f'Project path:{project_path}')
    return project_path

def set_data_path(data_dir_str:str, project_path: Path,):
    """ 
    Gets the data path relative to the project path.
    """

    # Split the data path string into a list
    data_dir_list = data_dir_str.split("/")

    # The data_path is a Path object with project path plus the data path list. It should be a directory
    data_dir = project_path / Path(*data_dir_list)

    while not data_dir.exists():
        file_name = data_dir.stem
        data_dir = data_dir.parent
    for child in data_dir.iterdir():
        if child.is_file():
            if file_name in child.stem:
              file_name = child.name
              data_path = child.parent
              break
    else:
        file_name = None
        print("File not found")
        data_path = data_dir
    print(f"Data path: {data_path}")
    print(f"File name: {file_name}")
    return data_path, file_name
    
    



project_path = set_project_path()
data_path, file_name = set_data_path("data/output/reduce_centre_cell_power/2023_03_17/rccp_s100_p43dBm", project_path)

print(data_path)
print(file_name)
