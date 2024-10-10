#!/usr/bin/env python3

"""
ROS node for reading and publishing data from logfile.

PURPOSE
This ROS node takes in information from logfile (to replace data from vicon for offline case),
and to publish it to vicon node.

SUBSCRIBED TOPICS
/

PUBLISHED TOPICS
estimated_state
/current_coordinates (Only for backwards compatibility!)

"""

import rospy
import numpy as np
import pandas as pd
from geometry_msgs.msg import TransformStamped
import tf.transformations as tf
from scipy.interpolate import interp1d

from utils import DataVarIndex, Status, load_data
from vicon_bridge.msg import Command


class LogfilePublisher(object):

    def __init__(self, logfile_path, publish_rate = 200):
        """Initialization."""

        self.file_path = logfile_path
        
        self.publish_rate = publish_rate
        self.data = []  # To store log data
        self.data2publish = {}  # To store transformed data (to be published)
        self.frequency_raised = False

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
            DataVarIndex.ROLL_RATE, 
            DataVarIndex.PITCH_RATE, 
            DataVarIndex.YAW_RATE,
            DataVarIndex.CMD_ROLL,
            DataVarIndex.CMD_PITCH,
            DataVarIndex.CMD_YAW,
            DataVarIndex.CMD_THRUST,
            DataVarIndex.DES_POS_X,
            DataVarIndex.DES_POS_Y,
            DataVarIndex.DES_POS_Z,
            DataVarIndex.DES_ROLL,
            DataVarIndex.DES_PITCH,
            DataVarIndex.DES_YAW,
            DataVarIndex.DES_VEL_X,
            DataVarIndex.DES_VEL_Y,
            DataVarIndex.DES_VEL_Z,
            DataVarIndex.STATUS,
            DataVarIndex.VICON_POS_X,
            DataVarIndex.VICON_POS_Y,
            DataVarIndex.VICON_POS_Z,
            DataVarIndex.VICON_ROLL,
            DataVarIndex.VICON_PITCH,
            DataVarIndex.VICON_YAW,
        ]

        # Define information of noise
        self.noise_mean_position = 0.0
        self.noise_variance_position = 0.0 #0.005
        self.noise_mean_euler = 0.0
        self.noise_variance_euler = 0.0 #0.01

        # Initialize the ROS node and publisher for state / input
        rospy.init_node('logfile_publisher', anonymous=True)
        self.pub_state = rospy.Publisher('logdata_state', TransformStamped, queue_size=1)
        #rate = rospy.Rate(self.publish_rate)  # May conflict with function 'Wait until timestamp matched'
        self.pub_input = rospy.Publisher('logdata_input', Command, queue_size=1)
        #rate = rospy.Rate(self.publish_rate)  # May conflict with function 'Wait until timestamp matched'

        # Load logdata from logfile
        self.load_data_from_logfile(self.file_path)

        # Convert data form from StateVector to TransformStamped
        self.StateVector2TransformStamped()

        # Start running publish loop
        rospy.sleep(2.0)
        self.start_publish()

    def load_data_from_logfile(self, file_path):
        # Read the data from the csv file
        self.data = load_data(file_path)

        # Subtract the start time from the time values
        start_time = self.data[0, DataVarIndex.TIME]
        self.data[:, DataVarIndex.TIME] -= start_time

        # Clear all data with other indices than to be published
        self.data = self.data[:, self.interpolate_indices]

    def StateVector2TransformStamped(self):
        """
        Convert StateVector data to TransformStamped format.
        ----------
        Parameters
        ----------
        self.data : StateVector
          - self.data[:, DataVarIndex.TIME] : float64 , ndarray
          - self.data[:, DataVarIndex.POS_X] : float64 , ndarray
          - self.data[:, DataVarIndex.POS_Y] : float64 , ndarray
          - self.data[:, DataVarIndex.POS_Z] : float64 , ndarray
          - self.data[:, DataVarIndex.ROLL] : float64 , ndarray
          - self.data[:, DataVarIndex.PITCH] : float64 , ndarray
          - self.data[:, DataVarIndex.YAW] : float64 , ndarray
        ----------
        self.vicon : geometry_msgs.msg.TransformStamped
          - self.data2publish : dictionary
          - self.data2publish['timestamp'] : float64 , ndarray
          - self.data2publish['secs'] : float64 , ndarray
          - self.data2publish['nsecs'] : float64 , ndarray
          - self.data2publish['translation_x'] : float64 , ndarray
          - self.data2publish['translation_y'] : float64 , ndarray
          - self.data2publish['translation_z'] : float64 , ndarray
          - self.data2publish['rotation_x'] : float64 , ndarray
          - self.data2publish['rotation_y'] : float64 , ndarray
          - self.data2publish['rotation_z'] : float64 , ndarray
          - self.data2publish['rotation_w' : float64 , ndarray
        """

        # Storage translational data
        self.data2publish['translation_x'] = self.data[:, DataVarIndex.VICON_POS_X]
        self.data2publish['translation_y'] = self.data[:, DataVarIndex.VICON_POS_Y]
        self.data2publish['translation_z'] = self.data[:, DataVarIndex.VICON_POS_Z]

        # Initialize arrays for rotational data
        self.data2publish['rotation_x'] = []
        self.data2publish['rotation_y'] = []
        self.data2publish['rotation_z'] = []
        self.data2publish['rotation_w'] = []

        # Calculate rotational data
        for entry in self.data:
            roll = entry[DataVarIndex.VICON_ROLL]
            pitch = entry[DataVarIndex.VICON_PITCH]
            yaw = entry[DataVarIndex.VICON_YAW]

            quaternion = tf.quaternion_from_euler(roll, pitch, yaw)

            self.data2publish['rotation_x'].append(quaternion[0])
            self.data2publish['rotation_y'].append(quaternion[1])
            self.data2publish['rotation_z'].append(quaternion[2])
            self.data2publish['rotation_w'].append(quaternion[3])

        # Convert lists to numpy arrays
        self.data2publish['rotation_x'] = np.array(self.data2publish['rotation_x'])
        self.data2publish['rotation_y'] = np.array(self.data2publish['rotation_y'])
        self.data2publish['rotation_z'] = np.array(self.data2publish['rotation_z'])
        self.data2publish['rotation_w'] = np.array(self.data2publish['rotation_w'])

        # Calculate timestamp
        self.data2publish['timestamp'] = self.data[:, DataVarIndex.TIME]
        self.data2publish['secs'] = self.data2publish['timestamp'].astype(int)
        self.data2publish['nsecs'] = ((self.data2publish['timestamp'] - self.data2publish['secs']) * 1e9).astype(int)

    def start_publish(self):
        """
        Start publishing the data.
        ----------
        Publishing format
        ----------
        vicon : geometry_msgs.msg.TransformStamped
          - vicon.header.stamp.secs : float64 , ndarray
          - vicon.header.stamp.nsecs : float64 , ndarray
          - vicon.transform.translation.x : float64 , ndarray
          - vicon.transform.translation.y : float64 , ndarray
          - vicon.transform.translation.z : float64 , ndarray
          - vicon.transform.rotation.x : float64 , ndarray
          - vicon.transform.rotation.y : float64 , ndarray
          - vicon.transform.rotation.z : float64 , ndarray
          - vicon.transform.rotation.w : float64 , ndarray
        """

        # Wait for subscribers to connect
        while self.pub_state.get_num_connections() == 0:
            rospy.loginfo("Waiting for subscribers to connect...")
            rospy.sleep(1)
        while self.pub_input.get_num_connections() == 0:
            rospy.loginfo("Waiting for subscribers to connect...")
            rospy.sleep(1)
        rospy.loginfo("Subscribers connected, starting to publish...")

        # Start publishing
        start_time = rospy.Time.now().to_sec()

        for i in range(len(self.data2publish['timestamp'])):
            if rospy.is_shutdown():
                break

            # State to be published 
            vicon = TransformStamped()
            vicon.header.stamp.secs = self.data2publish['secs'][i]
            vicon.header.stamp.nsecs = self.data2publish['nsecs'][i]
            vicon.header.frame_id = "world"
            vicon.child_frame_id = "base_link"
            vicon.transform.translation.x = self.data2publish['translation_x'][i]
            vicon.transform.translation.y = self.data2publish['translation_y'][i]
            vicon.transform.translation.z = self.data2publish['translation_z'][i]
            vicon.transform.rotation.x = self.data2publish['rotation_x'][i]
            vicon.transform.rotation.y = self.data2publish['rotation_y'][i]
            vicon.transform.rotation.z = self.data2publish['rotation_z'][i]
            vicon.transform.rotation.w = self.data2publish['rotation_w'][i]

            # Input to be published 
            controller = Command()
            controller.header.stamp.secs = self.data2publish['secs'][i]
            controller.header.stamp.nsecs = self.data2publish['nsecs'][i]
            controller.CMD_ROLL = self.data[i, DataVarIndex.CMD_ROLL]
            controller.CMD_PITCH = self.data[i, DataVarIndex.CMD_PITCH]
            controller.CMD_YAW = self.data[i, DataVarIndex.CMD_YAW]
            controller.CMD_PWM = self.data[i, DataVarIndex.CMD_THRUST]

            # Calculate current time from publisher starting
            current_time = rospy.Time.now().to_sec()
            elapsed_time = current_time - start_time

            # Wait until timestamp matched 
            while elapsed_time < self.data2publish['timestamp'][i]:
                current_time = rospy.Time.now().to_sec()
                elapsed_time = current_time - start_time
                rospy.sleep(0.0001)
        
            # Publish current data
            self.pub_state.publish(vicon)
            #print("logdata_state published, timestamp: ", self.data2publish['timestamp'][i])
            self.pub_input.publish(controller)
            #print("logdata_input published, timestamp: ", self.data2publish['timestamp'][i])
            #self.rate.sleep()

if __name__ == '__main__':
    # Initialize parameters of publisher class
    logfile_path = '/home/haocheng/Experiments/figure_8/data_20240604_145319.csv'
    publish_rate = 60

    # Create an instance of publisher class
    logfile_publisher = LogfilePublisher(logfile_path, publish_rate)

    # Do not exit until shutdown
    rospy.spin()
