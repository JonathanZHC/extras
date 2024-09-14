#!/usr/bin/env python3

"""
ROS node for running state estimation on top of VICON.

PURPOSE
This ROS node takes in information from VICON, transforms it, and sends it to
the drone over the /current_coordinates topic.

SUBSCRIBED TOPICS
/vicon/MODEL/MODEL, e.g. /vicon/ARDroneShare/ARDroneShare

PUBLISHED TOPICS
estimated_state
/current_coordinates (Only for backwards compatibility!)

VERSION HISTORY
At some time - initial work (Chris McKinnon)
Jun 09, 2014 - initialy created (Tristan Laidlow)
Aug 18, 2014 - Changed to 'Kalman filter'-ish, added angular velocity, using
               arrays (Felix Berkenkamp)
Jun 18, 2015 - Add proper threading to avoid conflicts
"""

from __future__ import division, print_function
from threading import Lock
import json
import math
import numpy as np
import tf.transformations as tf
from quaternions import (omega_from_quat_quat, apply_omega_to_quat, global_to_body)
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

from vicon_bridge.msg import StateData


def state_vector_to_data(state_vector):
    """
    Take a StateVector msg and turn it into StateData.

    Parameters:
    -----------
    state_vector: StateVector.msg

    Returns:
    --------
    state_data: StateData.msg
    """
    state_data = StateData()

    state_data.x, state_data.y, state_data.z = state_vector.pos
    state_data.vx, state_data.vy, state_data.vz = state_vector.vel
    state_data.ax, state_data.ay, state_data.az = state_vector.acc
    state_data.roll, state_data.pitch, state_data.yaw = state_vector.euler

    return state_data


class StateEstimator(object):
    """
    Vicon state estimation and filtering.

    Parameters
    ----------
    filter_parameters : sequence of floats
        The 4 tuning parameters for the Kalman filter
    """

    def __init__(self, filter_parameters, observer_type='simple', model_file=None):
        """Initialization."""
        super(StateEstimator, self).__init__()

        # Lock for access to state
        # Makes sure the service does not conflict with normal updates
        self.state_access_lock = Lock()

        # define type of observer
        self.observer = observer_type  # 'simple' / 'EKF' / 'UKF'
        # define file path of identified model
        self.model_file = model_file  # json file with identified parameter

        # Initialize the state variables
        self.pos = np.array([0.0, 0.0, 0.0323], dtype=np.float64) # Initialized by collected intial data before
        self.vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.acc = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize rotations.
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.euler = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.omega_g = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize measurements
        self.pos_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.vel_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.acc_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.quat_meas = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.euler_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.omega_g_meas = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize old measurements
        self.pos_old = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.vel_old = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.quat_old = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.euler_old = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Correction to align vicon frame with body frame
        self.quat_corr = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        # Initialize time..at first time step trusts measurements to 100%
        self.time = 0.0
        self.time_meas = 0.0
        self.dt = 0.0

        # Define Gravitational Acceleration
        self.GRAVITY = 9.806

        # Import model parameters
        with open(self.model_file) as file:
            model = json.load(file)

        self.params_acc = model['params_acc']
        self.params_pitch_rate = model['params_pitch_rate']
        self.params_roll_rate = model['params_roll_rate']
        self.params_yaw_rate = model['params_yaw_rate']

        # Define output matrix
        self.G = np.eye(9)

        # Initialize the state vector
        self.x_old = np.concatenate((self.pos, self.vel, self.euler))

        # Initialize the input vector
        self.input_CMD = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if self.observer == 'simple':
            # Read in the parameters
            (self.tau_est_trans,
             self.tau_est_trans_dot,
             self.tau_est_trans_dot_dot,
             self.tau_est_rot,
             self.tau_est_rot_dot) = filter_parameters

        elif self.observer == 'EKF':
            # Define weight matrix
            # parameter tuning: Q smaller -> depend more on prediction
            self.Q = np.eye(9) * 0.03
            self.R = np.eye(9) * 0.1

            # Initialize the prediction matrix
            self.P_old = np.eye(9) * 1 # Initialized by collected intial data before

            # Initialize the state transfer matrix (df/dx for x0)
            self.F_old = np.eye(9)
            self.F_old[0, 3] += 1
            self.F_old[1, 4] += 1
            self.F_old[2, 5] += 1
            self.F_old[-3, -3] += self.params_pitch_rate[0][0]
            self.F_old[-2, -2] += self.params_roll_rate[0][0]
            self.F_old[-1, -1] += self.params_yaw_rate[0][0]
            self.F_old = self.F_old * self.dt

        elif self.observer == 'UKF':
            # Define sigma points
            points = MerweScaledSigmaPoints(9, alpha=0.1, beta=2., kappa=0)
            self.ukf = UKF(dim_x=9, dim_z=9, fx=self.x_update, hx=self.y_update, dt=self.dt, points=points)

            self.ukf.x = self.x_old
            self.ukf.P = np.eye(9) * 1

            self.ukf.Q = np.eye(9) * 0.04
            self.ukf.R = np.eye(9) * 0.1

    @property
    def rpy(self):
        """Convert quaternion to roll, pitch, yaw."""
        return tf.euler_from_quaternion(self.quat)

    @property
    def rpy_meas(self):
        """Convert quaternion to roll, pitch, yaw."""
        return tf.euler_from_quaternion(self.quat_meas)

    @property
    def quat_back(self):
        """Convert roll, pitch, yaw to quaternion."""
        roll = self.euler[0]
        pitch = self.euler[1]
        yaw = self.euler[2]
        return tf.quaternion_from_euler(roll, pitch, yaw)

    @property
    def omega_b(self):
        """Return the body angular velocity."""
        return global_to_body(self.quat, self.omega_g)

    def get_new_measurement(self, time, position, quaternion, input, dt=None):
        """Get a new measurement of position and orientation.

        Parameters
        ----------
        time: float
            The measurement time in seconds.
        position: ndarray
        quaternion: ndarray
        """

        # Record the time at which the state was determined
        self.time_meas = time
        # Calculate time difference, update time
        self.dt = self.time_meas - self.time

        if self.observer in ['EKF', 'UKF']:
            # Get the commanded input from controller
            self.input_CMD = input

        # Get the translational position from VICON
        self.pos_meas = position

        # Get the rotational position in the form of a quaternion
        self.quat_meas = quaternion
        # Apply correction
        self.quat_meas = tf.quaternion_multiply(self.quat_meas, self.quat_corr)
        # Two quaternions for every rotation, make sure we take
        # the one that is consistent with previous measurements
        if np.dot(self.quat_old, self.quat_meas) < 0.0:
            self.quat_meas = -self.quat_meas

        # transform quat_meas into euler_meas (function rpy_meas)
        self.euler_meas = self.rpy_meas
        #self.euler_meas = self.check_euler(self.rpy_meas)

        # Don't compute finite difference for impossibly small time differencesc
        # EG: for first input
        if self.dt <= 1e-15:
            return
        
        if self.observer == 'simple':
            # Numeric derivatives: Compute velocities
            self.vel_meas = (self.pos_meas - self.pos_old) / self.dt
            # Numeric derivatives: Compute accelerations
            self.acc_meas = (self.vel_meas - self.vel_old) / self.dt
            # Numeric derivatives: Compute angular velocity
            self.omega_g_meas = omega_from_quat_quat(self.quat_old, self.quat_meas, self.dt)

        elif self.observer == 'EKF':
            # Numeric derivatives: Compute velocities # used in 'self.x_meas'
            self.vel_meas = (self.pos_meas - self.pos_old) / self.dt
            # combine pos, vel, euler angles into state vector & output vector
            self.x_meas = np.concatenate((self.pos_meas, self.vel_meas, self.euler_meas))
            self.y_meas = self.G @ self.x_meas
            
        elif self.observer == 'UKF':
            # Numeric derivatives: Comput velocities
            self.vel_meas = (self.pos_meas - self.pos_old) / self.dt
            
            # combine pos, vel, euler angles into state vector & output vector
            self.x_meas = np.concatenate((self.pos_meas, self.vel_meas, self.euler_meas))
            self.y_meas = self.G @ self.x_meas
            
            # Update old measurements (make a copy)
            self.pos_old[:] = self.pos_meas
            self.vel_old[:] = self.vel_meas
            self.quat_old[:] = self.quat_meas
            self.x_old[:] = self.x_meas


    def prior_update(self):
        """Predict future states using a double integrator model."""
        # Acceleration and angular velocity are assumed constant
        # Update position and orientation
        self.pos += self.dt * self.vel + 0.5 * self.dt * self.dt * self.acc
        self.vel += self.dt * self.acc
        self.quat = apply_omega_to_quat(self.quat, self.omega_g, self.dt)


    def measurement_update(self):
        """Update state estimate with measurements."""
        # NOTE: Use raw position and quaternion - no low pass filter
        # Calculate current Kalman filter gains
        c1 = math.exp(-self.dt / self.tau_est_trans)
        c2 = math.exp(-self.dt / self.tau_est_trans_dot)
        c3 = math.exp(-self.dt / self.tau_est_trans_dot_dot)

        d1 = math.exp(-self.dt / self.tau_est_rot)
        d2 = math.exp(-self.dt / self.tau_est_rot_dot)

        # Wait while locked, then lock itself
        with self.state_access_lock:
            # Measurement updates
            self.pos = (1.0 - c1) * self.pos_meas + c1 * self.pos
            self.vel = (1.0 - c2) * self.vel_meas + c2 * self.vel
            self.acc = (1.0 - c3) * self.acc_meas + c3 * self.acc

            self.quat = (1.0 - d1) * self.quat_meas + d1 * self.quat
            self.omega_g = (1.0 - d2) * self.omega_g_meas + d2 * self.omega_g

            # Make sure that numerical errors don't pile up
            self.quat /= np.linalg.norm(self.quat)

        self.time = self.time_meas

        # Update old measurements (make a copy)
        self.pos_old[:] = self.pos_meas
        self.vel_old[:] = self.vel_meas
        self.quat_old[:] = self.quat_meas

    def EKF_update(self):
        # Don't compute finite difference for impossibly small time differencesc
        # EG: for first input
        if self.dt <= 1e-15:
            return

        # Wait while locked, then lock itself
        with self.state_access_lock:

            # Predictor
            self.P_pri = self.F_old @ self.P_old @ np.transpose(self.F_old) + self.Q
            self.x_pri = self.x_update(self.x_old, self.dt)

            # solve kalman gain
            self.K = self.P_pri @ np.transpose(self.G) @ np.linalg.inv(self.G @ self.P_pri @ np.transpose(self.G) + self.R)

            # Corrector
            self.P_post = (np.eye(9) - self.K @ self.G) @ self.P_pri
            self.x_post = self.x_pri + self.K @ (self.y_meas - self.G @ self.x_pri)

            # Measurement updates
            self.pos = self.x_post[:3]
            self.vel = self.x_post[3:6]
            self.acc = (self.vel - self.vel_old) / self.dt

            self.euler = self.x_post[6:]
            self.quat = self.quat_back

            # Two quaternions for every rotation, make sure we take
            # the one that is consistent with previous measurements
            if np.dot(self.quat_old, self.quat) < 0.0:
                self.quat = -self.quat
            # Make sure that numerical errors don't pile up
            self.quat /= np.linalg.norm(self.quat)

            # Measurement updates (omega)
            self.omega_g = omega_from_quat_quat(self.quat_old,
                                                self.quat,
                                                self.dt)

        # Update old value for next iteration (make a copy)
        self.time = self.time_meas
        # different from simple one, we use predicted value but not the observed value to do the update
        self.P_old = self.P_post
        self.x_old = self.x_post
        self.F_old = self.F_update()

        self.pos_old[:] = self.pos
        self.vel_old[:] = self.vel
        self.quat_old[:] = self.quat

    def UKF_update(self):
        if self.dt <= 1e-15:
            return

        with self.state_access_lock:
            self.ukf.predict()
            self.ukf.update(self.y_meas)

            self.x_post = self.ukf.x

            # Measurement updates
            self.pos = self.x_post[:3]
            self.vel = self.x_post[3:6]
            self.acc = (self.vel - self.vel_old) / self.dt

            self.euler = self.x_post[6:]
            self.quat = self.quat_back

            # Two quaternions for every rotation, make sure we take
            # the one that is consistent with previous measurements
            if np.dot(self.quat_old, self.quat) < 0.0:
                self.quat = -self.quat
            # Make sure that numerical errors don't pile up
            self.quat /= np.linalg.norm(self.quat)

            # Measurement updates (omega)
            self.omega_g = omega_from_quat_quat(self.quat_old,
                                                self.quat,
                                                self.dt)

        # Update old value for next iteration (make a copy)
        self.time = self.time_meas
        #self.pos_old[:] = self.pos
        #self.vel_old[:] = self.vel
        #self.quat_old[:] = self.quat

    def F_update(self):
        # Initialize F
        F = np.eye(9)

        # use identified model to calculate collective thrust
        transformed_thrust = self.params_acc[0] * self.input_CMD[3] + self.params_acc[1]

        # Calculate df/dx
        F[0, 3] += 1
        F[1, 4] += 1
        F[2, 5] += 1
        
        '''
        F[3, 7] += transformed_thrust * np.cos(self.x_old[7])
        F[4, 7] += transformed_thrust * np.sin(self.x_old[6]) * np.sin(self.x_old[7])
        F[4, 6] += -transformed_thrust * np.cos(self.x_old[6]) * np.cos(self.x_old[7])
        F[5, 7] += -transformed_thrust * np.cos(self.x_old[6]) * np.sin(self.x_old[7])
        F[5, 6] += -transformed_thrust * np.sin(self.x_old[6]) * np.cos(self.x_old[7])
        '''
        
        F[3, 6] += transformed_thrust * (- np.sin(self.x_old[6]) * np.sin(self.x_old[7]) * np.cos(self.x_old[8]) + np.cos(self.x_old[6]) * np.sin(self.x_old[8]))
        F[4, 6] += transformed_thrust * (- np.sin(self.x_old[6]) * np.sin(self.x_old[7]) * np.sin(self.x_old[8]) - np.cos(self.x_old[6]) * np.cos(self.x_old[8]))
        F[5, 6] += - transformed_thrust * np.sin(self.x_old[6]) * np.cos(self.x_old[7])
        F[3, 7] += transformed_thrust * np.cos(self.x_old[6]) * np.cos(self.x_old[7]) * np.cos(self.x_old[8])
        F[4, 7] += transformed_thrust * np.cos(self.x_old[6]) * np.cos(self.x_old[7]) * np.sin(self.x_old[8])
        F[5, 7] += - transformed_thrust * np.cos(self.x_old[6]) * np.sin(self.x_old[7])
        F[3, 8] += transformed_thrust * (- np.cos(self.x_old[6]) * np.sin(self.x_old[7]) * np.sin(self.x_old[8]) + np.sin(self.x_old[6]) * np.cos(self.x_old[8]))
        F[4, 8] += transformed_thrust * (np.cos(self.x_old[6]) * np.sin(self.x_old[7]) * np.cos(self.x_old[8]) + np.sin(self.x_old[6]) * np.sin(self.x_old[8]))
        F[5, 8] += 0
        

        F[-3, -3] += self.params_roll_rate[0][0]
        F[-2, -2] += self.params_pitch_rate[0][0]
        F[-1, -1] += self.params_yaw_rate[0][0]

        F = F * self.dt

        return F

    def x_update(self, x, dt):
        # 状态转移函数
        f_x = np.zeros(9)

        # use identified model to calculate collective thrust
        transformed_thrust = self.params_acc[0] * self.input_CMD[3] + self.params_acc[1]

        # Update f_x value
        f_x[0] = x[3]
        f_x[1] = x[4]
        f_x[2] = x[5]
        
        '''
        f_x[3] = transformed_thrust * np.sin(x[7])
        f_x[4] = -transformed_thrust * np.sin(x[6]) * np.cos(x[7])
        f_x[5] = transformed_thrust * np.cos(x[6]) * np.cos(x[7]) - self.GRAVITY
        '''
        
        f_x[3] = transformed_thrust * (np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) + np.sin(x[6]) * np.sin(x[8]))
        f_x[4] = transformed_thrust * (np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) - np.sin(x[6]) * np.cos(x[8]))
        f_x[5] = transformed_thrust * np.cos(x[6]) * np.cos(x[7]) - self.GRAVITY
        

        f_x[6] = self.params_roll_rate[0][0] * x[6] + self.params_roll_rate[1][0] * self.input_CMD[0]
        f_x[7] = self.params_pitch_rate[0][0] * x[7] + self.params_pitch_rate[1][0] * self.input_CMD[1]
        f_x[8] = self.params_yaw_rate[0][0] * x[8] + self.params_yaw_rate[1][0] * self.input_CMD[2]

        # Update x_next according to system dynamic
        x_next = x + dt * f_x

        return x_next

    def y_update(self, x):
        # update output
        return x

    '''
    def check_euler(self, euler):
        for angle in euler:
            if abs(angle) > (2*math.pi/3) and abs(angle) < math.pi:
                angle = np.sign(angle)* (math.pi - abs(angle))
            
        return euler
    '''

    