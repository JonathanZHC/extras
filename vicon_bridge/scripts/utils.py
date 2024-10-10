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
    DES_ROLL = 20
    DES_PITCH = 21
    DES_YAW = 22
    DES_VEL_X = 23
    DES_VEL_Y = 24
    DES_VEL_Z = 25
    STATUS = 26  # One of the following: "TAKEOFF", "LAND", "TRACK_TRAJ"
    VICON_POS_X = 27
    VICON_POS_Y = 28
    VICON_POS_Z = 29
    VICON_ROLL = 30
    VICON_PITCH = 31
    VICON_YAW = 32
    ACC_X = 33
    ACC_Y = 34
    ACC_Z = 35
    ROLL_ACC = 36
    PITCH_ACC = 37
    YAW_ACC = 38

    "----------for test----------"
    # to display the predicted state, 
    # can only choose 1 state at same time, 
    # don't need to change variable name
    x_0 = 39
    x_1 = 40
    x_2 = 41
    x_3 = 42
    x_4 = 43
    x_5 = 44
    x_6 = 45
    x_7 = 46
    x_8 = 47
    x_9 = 48
    x_10 = 49
    x_11 = 50
    x_12 = 51
    x_13 = 52
    x_14 = 53
    x_15 = 54
    x_16 = 55
    x_17 = 56
    x_18 = 57
    x_19 = 58
    x_20 = 59
    x_21 = 60
    x_22 = 61
    x_23 = 62
    x_24 = 63
    x_25 = 64
    x_26 = 65
    x_27 = 66
    x_28 = 67
    x_29 = 68
    x_30 = 69
    "----------for test----------"


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
    STATIC_OBSV = 7


match_status = {
        Status.TAKEOFF.name: Status.TAKEOFF,
        Status.LAND.name: Status.LAND,
        Status.TRACK_TRAJ.name: Status.TRACK_TRAJ,
        Status.HOVER.name: Status.HOVER,
        Status.VERTICAL.name: Status.VERTICAL,
        Status.INTERPOLATE.name: Status.INTERPOLATE,
        Status.HORIZONTAL.name: Status.HORIZONTAL,
        Status.STATIC_OBSV.name: Status.STATIC_OBSV,
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
