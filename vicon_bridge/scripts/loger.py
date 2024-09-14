#!/usr/bin/env python

import rospy
import numpy as np
import pandas as pd
from rospy.numpy_msg import numpy_msg
from vicon_bridge.msg import StateVector
from utils import DataVarIndex, Status, load_data
import matplotlib.pyplot as plt

class StateVectorListener:
    def __init__(self, raw_data_filepath, noise_data_filepath):

        self.num_columns = len(DataVarIndex)

        # Initialize all data variables
        self.data = np.empty((0, 10))
        self.raw_data = np.empty((0, self.num_columns))
        self.noise_data = np.empty((0, self.num_columns))

        # Initialize difference rate : avg |(estimated - raw) / (noise_affected - raw)|
        self.difference_avg = np.zeros(10)

        # Select which indices to do interpolation
        self.interpolate_indices = [
            DataVarIndex.TIME,
            DataVarIndex.POS_X, 
            DataVarIndex.POS_Y, 
            DataVarIndex.POS_Z,
            DataVarIndex.ROLL,
            DataVarIndex.PITCH,
            DataVarIndex.YAW,
            DataVarIndex.VEL_X, 
            DataVarIndex.VEL_Y, 
            DataVarIndex.VEL_Z,
        ]

        # Noise affected set
        self.noise_affected_indices = [
            DataVarIndex.POS_X, 
            DataVarIndex.POS_Y, 
            DataVarIndex.POS_Z,
            DataVarIndex.ROLL,
            DataVarIndex.PITCH,
            DataVarIndex.YAW,
        ]

        # Plot set
        self.plot_index = [
            DataVarIndex.POS_X, 
            DataVarIndex.POS_Y, 
            DataVarIndex.POS_Z,
            DataVarIndex.ROLL,
            DataVarIndex.PITCH,
            DataVarIndex.YAW,
        ]

        rospy.init_node('state_vector_listener', anonymous=True)
        rospy.Subscriber('estimated_state_sim', numpy_msg(StateVector), self.callback)

        # Read the raw test data & noise affected data from the csv file
        self.raw_data = load_data(raw_data_filepath)[1:, :]
        self.noise_data = load_data(noise_data_filepath)[1:, :]

    def callback(self, msg):
        # Trandform message from StateVector into array
        msg_array = np.array([[
            round((float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) * 1e-9), 3),
            msg.pos[0], 
            msg.pos[1], 
            msg.pos[2],
            msg.euler[0],
            msg.euler[1],
            msg.euler[2],
            msg.vel[0], 
            msg.vel[1], 
            msg.vel[2],
        ]])
        
        self.data = np.vstack((self.data, msg_array))
        
        #rospy.loginfo("Received message: %s", msg)

    def data_cleaning(self):

        '''Correcte un-accurate estimations due to incorrect initial state'''
        # Correcte the initial update for velocity which have large deviation 
        self.data[1, 1:] = self.data[2, 1:]
        self.data[0, 1:] = self.data[1, 1:]
        
        '''Use hold signals of last siganl to approximate value on missing timestamp'''
        # Calculate dt from DataVarIndex.TIME (smallest time difference)
        right_diff = np.diff(self.data[:, DataVarIndex.TIME], append=np.nan)
        right_diff = right_diff[:-1]
        dt = round(np.nanmin(np.abs([right_diff])), 3)

        # Generate correct timeline
        first_time = self.data[0, DataVarIndex.TIME]
        last_time = self.data[-1, DataVarIndex.TIME]
        timeline_ref = np.arange(first_time, last_time + dt, dt)
        
        # Insert row on last timestamp to row on missing timestamp
        epsilon = 1 #1e-4  # define upper limit of truncation error
        for time in timeline_ref:
            # when difference is smaller than upper limit of truncation error, treat in ta = tb
            if not np.any(np.abs(self.data[:, DataVarIndex.TIME] - time) < epsilon):
                prev_time_index = np.where(self.data[:, DataVarIndex.TIME] < time)[0][-1]
                new_row = self.data[prev_time_index, :].copy()
                new_row[DataVarIndex.TIME] = round(time, 3)
                self.data = np.insert(self.data, prev_time_index + 1, new_row, axis=0)

    def save_data(self, path):
        """Save the revieved data to a csv file."""
        # Add dummy values for the rest columns
        num_data_columns = self.data.shape[1]
        dummy_data = np.zeros((self.data.shape[0], self.num_columns - num_data_columns))
        self.data = np.hstack((self.data, dummy_data))
        
        # Save the data to a csv file with DataVarIndex as the header
        pd_data = pd.DataFrame(self.data, columns=[var.name for var in DataVarIndex])

        # Replace the status with the actual status string
        pd_data[DataVarIndex.STATUS.name] = pd_data[DataVarIndex.STATUS.name].apply(lambda s: Status(s).name)
        pd_data.to_csv(path, index=False)
        
        print("Data recieved and saved to: ", path)

        return path

    def difference_pretreatment(self):

        # Judge whether 3 data variables have same column numbers
        print(self.data.shape[0], self.raw_data.shape[0], self.noise_data.shape[0])
        if not self.data.shape[0] == self.raw_data.shape[0] == self.noise_data.shape[0]:
            rospy.loginfo('3 data variables are not in same shape, returns')
            return
        
        
        # Calculate difference & original noise: 
        self.difference = np.hstack((self.data[:, 0].reshape(-1, 1), self.data[:, 1:] - self.raw_data[:, 1:]))
        self.noise = np.hstack((self.noise_data[:, 0].reshape(-1, 1), self.noise_data[:, 1:] - self.raw_data[:, 1:]))

        # Calculate average difference for each index
        for index in self.noise_affected_indices:

            segma_difference = 0
            segma_noise = 0
            
            # Except for first & second column, while inapropriate initialization will cause large deviation at first step
            for time in range(2, self.data.shape[0]):
               
                estimated_data = self.data[time, index.value]
                raw_data = self.raw_data[time, index]
                noise_data = self.noise_data[time, index]
               
                # Calculate difference for specific index and time : segma(|(estimated - raw)|) / segma(|(noise_affected - raw)|)
                segma_difference += np.abs((estimated_data - raw_data))
                segma_noise += np.abs((noise_data - raw_data))

            self.difference_avg[index.value] = segma_difference / segma_noise
            rospy.loginfo('Vairable name: %s, Difference rate: %f' %(index, self.difference_avg[index.value]))

    def difference_plotting(self):

        time = self.difference[:, DataVarIndex.TIME]

        for index in self.plot_index:
            difference = self.difference[:, index]
            original_noise = self.noise[:, index]
            Y_lim = 1.2 * max(max(abs(difference)), max(abs(original_noise)))

            # 计算均值和方差
            mean_noise = np.mean(original_noise)
            std_noise = np.std(original_noise)

            # 创建子图
            fig, ax = plt.subplots(1, 1, figsize=(18, 8), sharex=False)

            # 绘制第一组噪声数据
            ax.plot(time, difference, label='Difference')
            ax.plot(time, original_noise, label='Noise', linestyle = '--', linewidth = 0.6)
            ax.axhline(mean_noise, color='r', linestyle='--', label='Mean')
            ax.axhline(mean_noise + 3 * std_noise, color='g', linestyle='--', label='Mean + 3 Std Dev')
            ax.axhline(mean_noise - 3 * std_noise, color='g', linestyle='--', label='Mean - 3 Std Dev')
            ax.set_title('Difference: estimated_state - real_state')
            ax.legend()
            ax.set_ylim([-Y_lim, Y_lim])  # 设置Y轴范围

            # 显示图形
            plt.tight_layout()
            plt.show()


if __name__ == '__main__':

    logfile_path = '/home/haocheng/Experiments/figure_8/data_20240604_145319.csv'
    save_filepath = logfile_path.replace('.csv', '_estimated_data_from_observer.csv')
    raw_data_filepath = logfile_path.replace('.csv', '_raw_data_from_vicon.csv')
    noise_data_filepath = logfile_path.replace('.csv', '_raw_data_add_noise_from_vicon.csv')

    listener = StateVectorListener(raw_data_filepath, noise_data_filepath)
    rospy.sleep(30.0)

    #listener.data_cleaning()
    listener.save_data(save_filepath)
    listener.difference_pretreatment()
    #listener.difference_plotting() # can only call after function difference_pretreatment()
    rospy.spin()

