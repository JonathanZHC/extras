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

import math
import numpy
import tf.transformations as tf
from quaternions import (omega_from_quat_quat, apply_omega_to_quat,
                         global_to_body)

from vicon_bridge.msg import StateData


def state_vector_to_data(state_vector):
        """
        Take a StateVector msg and turn it into StateData.

        Parameters:
        -----------
        state_vector: StateVector.msg

        Returns:
        --------
        state_data: StateDate.msg
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

    def __init__(self, filter_parameters):
        """Initializaiton."""
        super(StateEstimator, self).__init__()

        # Lock for access to state
        # Makes sure the service does not conflict with normal updates
        self.state_access_lock = Lock()

        # Read in the parameters
        (self.tau_est_trans,
         self.tau_est_trans_dot,
         self.tau_est_trans_dot_dot,
         self.tau_est_rot,
         self.tau_est_rot_dot) = filter_parameters

        # Initialize the state variables
        self.pos = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.vel = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.acc = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)

        # Initialize rotations (quaternions).
        self.quat = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float64)
        self.omega_g = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)

        # Initialize measurements
        self.pos_meas = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.vel_meas = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.acc_meas = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.quat_meas = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float64)
        self.omega_g_meas = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)

        # Initialze old measurements
        self.pos_old = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.vel_old = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)
        self.quat_old = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float64)

        # Correction to align vicon frame with body frame
        self.quat_corr = numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float64)

        # Initialize time..at first time step trusts measurements to 100%
        self.time = 0.0
        self.time_meas = 0.
        self.dt = 0.0

    @property
    def rpy(self):
        """Convert quaternion to roll, pitch, yaw."""
        return tf.euler_from_quaternion(self.quat)

    @property
    def omega_b(self):
        """Return the body angular velocity."""
        return global_to_body(self.quat, self.omega_g)

    def get_new_measurement(self, time, position, quaternion):
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

        # Get the translational position from VICON
        self.pos_meas = position

        # Get the rotational position in the form of a quaternion
        self.quat_meas = quaternion

        # Apply correction
        self.quat_meas = tf.quaternion_multiply(self.quat_meas, self.quat_corr)

        # Two quaternions for every rotation, make sure we take
        # the one that is consistent with previous measurements
        if numpy.dot(self.quat_old, self.quat_meas) < 0.0:
            self.quat_meas = -self.quat_meas

        # Don't compute finite difference for impossibly small time differences
        if self.dt <= 1e-15:
            return

        # Numeric derivatives: Comput velocities
        self.vel_meas = (self.pos_meas - self.pos_old) / self.dt

        # Numeric derivatives: Compute accelerations
        self.acc_meas = (self.vel_meas - self.vel_old) / self.dt

        # Numeric derivatives: Compute angular velocity
        self.omega_g_meas = omega_from_quat_quat(self.quat_old,
                                                 self.quat_meas,
                                                 self.dt)

        # Update old measurements (make a copy)
        self.pos_old[:] = self.pos_meas
        self.vel_old[:] = self.vel_meas
        self.quat_old[:] = self.quat_meas

    def prior_update(self):
        """Predict future states using a double integrator model."""
        # Acceleration and angular velocity are assumed constant
        # Update position and orientation
        self.pos += self.dt * self.vel  # + 0.5 * self.dt * self.dt * self.acc
        # self.vel += self.dt * self.acc
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
            self.quat /= tf.vector_norm(self.quat)

            self.time = self.time_meas
