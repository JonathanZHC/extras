import json
import pandas as pd
from enum import IntEnum, unique
import wandb
import numpy as np


GRAVITY = 9.81


@unique
class DataVarIndex(IntEnum):
    '''A class that creates ids for the data.'''

    TIME = 0
    POS_X = 1
    POS_Y = 2
    POS_Z = 3
    ROLL = 4
    PITCH = 5
    YAW = 6
    VEL_X = 7
    VEL_Y = 8
    VEL_Z = 9
    ROLL_RATE = 10
    PITCH_RATE = 11
    YAW_RATE = 12
    CMD_ROLL = 13
    CMD_PITCH = 14
    CMD_YAW = 15
    CMD_THRUST = 16
    DES_POS_X = 17
    DES_POS_Y = 18
    DES_POS_Z = 19
    DES_YAW = 20
    DES_VEL_X = 21
    DES_VEL_Y = 22
    DES_VEL_Z = 23
    STATUS = 24  # One of the following: "TAKEOFF", "LAND", "TRACK_TRAJ"
    VICON_POS_X = 25
    VICON_POS_Y = 26
    VICON_POS_Z = 27
    VICON_ROLL = 28
    VICON_PITCH = 29
    VICON_YAW = 30
    ACC_X = 31
    ACC_Y = 32
    ACC_Z = 33
    ROLL_ACC = 34
    PITCH_ACC = 35
    YAW_ACC = 36


var_bounds = {
    DataVarIndex.CMD_THRUST: (2.0e4, 65535.0),
}


@unique
class Status(IntEnum):
    '''A class that creates ids for the status of the drone.'''
    TAKEOFF = 0
    LAND = 1
    TRACK_TRAJ = 2
    HOVER = 3
    VERTICAL = 4
    INTERPOLATE = 5
    HORIZONTAL = 6


match_status = {
        Status.TAKEOFF.name: Status.TAKEOFF,
        Status.LAND.name: Status.LAND,
        Status.TRACK_TRAJ.name: Status.TRACK_TRAJ,
        Status.HOVER.name: Status.HOVER,
        Status.VERTICAL.name: Status.VERTICAL,
        Status.INTERPOLATE.name: Status.INTERPOLATE,
        Status.HORIZONTAL.name: Status.HORIZONTAL,
        }


def load_data(filename):
    """Load the data from the csv file and return it as a numpy array."""
    # Read the data from the csv file, skipping the first row
    # and the last column has to be transformed using Status enum
    pd_data = pd.read_csv(filename)
    pd_data[DataVarIndex.STATUS.name] = pd_data[DataVarIndex.STATUS.name].apply(lambda s: match_status[s])
    data = pd_data.to_numpy()

    # There may be a mismatch in the number of columns and the number of DataVarIndex. Add dummy values for the missing columns
    num_columns = len(DataVarIndex)
    num_data_columns = data.shape[1]
    dummy_data = np.zeros((data.shape[0], num_columns - num_data_columns))
    data = np.hstack((data, dummy_data))

    return data


def get_file_path_from_run(wandb_project, run_name=None, file_name=None, use_latest=False, smoothed=False):
    """Get the file path from the specified run name or file name."""
    # Use the latest run if no run name or file name is provided
    if run_name is None and file_name is None:
        use_latest = True

    wandb_api = wandb.Api()
    runs = wandb_api.runs(wandb_project)
    
    if use_latest:
        # Get the latest run
        run = runs[0]
    elif run_name is not None:
        # Get the run with the specified name
        run = None
        for r in runs:
            if r.name == run_name:
                run = r
                break
        if run is None:
            raise ValueError("Run with name {} not found".format(run_name))
    elif file_name is not None:
        # Get the run with the specified file name
        run = None
        for r in runs:
            r_json = json.loads(r.json_config)
            run_file_name = r_json['file_path']['value'].rsplit('/')[-1]
            print(run_file_name, file_name)
            if run_file_name == file_name:
                run = r
                break
        if run is None:
            raise ValueError("Run with file name {} not found".format(file_name))
    else:
        raise ValueError("No run name or file name provided")
    
    run_json = json.loads(run.json_config)
    print("Run name: ", run.name)

    file_path = run_json['file_path']['value']
    traj_plane = run_json['traj_plane']['value']

    return file_path, traj_plane
