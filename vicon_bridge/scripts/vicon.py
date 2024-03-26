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
Jan 17, 2017 - Fix Folder structure and coding style
               (Melissa Greeff, Yassine Nemmour)
"""

from __future__ import division, print_function
import rospy

import numpy
import tf.transformations as tf
from quaternions import (apply_omega_to_quat,
                                                global_to_body)

from vicon_bridge.srv import GetState
from std_msgs.msg import Empty
from geometry_msgs.msg import TransformStamped
from vicon_bridge.msg import StateData
from vicon_bridge.msg import StateVector
from state_estimation import state_vector_to_data, \
    StateEstimator

# Needed to send numpy.array as a msg
from rospy.numpy_msg import numpy_msg


class ViconCoordinates(object):
    """
    Vicon state estimation and filtering.

    Parameters:
    -----------
    filter_parameters : float
        The 4 tuning parameters for the Kalman filter.

    Attributes:
    -----------
        estimator: StateEstimator

        sub_vicon: rospy.Subscriber
            Subscribed to state-estimation topic.
        pub_state: rospy.Publisher
            Publishes estimated state.
        pub_state_old: rospy.Publisher
            Publish previous state.
        sub_cal: rospy.Subscriber
            Get calibration data from parameter server.
        srv_vicon: rospy.Service
            Fetch current state of object in vicon.
    """

    def __init__(self, filter_parameters):
        """Initializaiton."""
        super(ViconCoordinates, self).__init__()

        # Lock for access to state
        # Makes sure the service does not conflict with normal updates
        self.estimator = StateEstimator(filter_parameters)

        model_name = MODEL + '/' + MODEL

        # Subscribe to the VICON information
        self.sub_vicon = rospy.Subscriber('/vicon/' + model_name,
                                          TransformStamped,
                                          self.estimate_state)

        # Publish to the /current_coordinates topic
        # queue_size: Only continous time controllers should
        # subscribe to this, and they only care about the latest
        # state information. Everything else should use the service
        self.pub_state = rospy.Publisher('estimated_state',
                                         numpy_msg(StateVector),
                                         queue_size=1)
        self.pub_state_old = rospy.Publisher('current_coordinates',
                                             StateData,
                                             queue_size=1)

        # Subscribe to the /new_calibration topic
        # (updates parameters after calibration)
        self.sub_cal = rospy.Subscriber('/new_calibration', Empty,
                                        self.update_calibration)

        # Wait for vicon to actually publish
        rospy.loginfo('vicon.py: Waiting for vicon to publish {0}'
                      .format(MODEL))
        while self.sub_vicon.get_num_connections() == 0:
            rospy.sleep(0.1)
        rospy.loginfo('vicon.py: Got data for {0}'.format(MODEL))

        # Launch a service which gets the current state
        # (persistent connection --> some extra speed)
        self.srv_vicon = rospy.Service(MODEL + '_get_state', GetState,
                                       self.get_current_state, True)

        # Initialize attitude correction
        self.estimator.quat_corr[:] = (
            rospy.get_param(model_name + '/q_corr/x', 0.),
            rospy.get_param(model_name + '/q_corr/y', 0.),
            rospy.get_param(model_name + '/q_corr/z', 0.),
            rospy.get_param(model_name + '/q_corr/w', 1.))

    def convert_measurement(self, vicon):
        """Convert vicon measurements and calculate velocities.

        Parameters
        ----------
        vicon : geometry_msgs.msg.TransformStamped

        Returns
        -------
        time : float
        position : ndarray
        quaternion : ndarray
        """
        # Record the time at which the state was determined
        time = (float(vicon.header.stamp.secs)
                + float(vicon.header.stamp.nsecs) * 1e-9)

        # Get the translational position from VICON
        position = numpy.array([vicon.transform.translation.x,
                               vicon.transform.translation.y,
                               vicon.transform.translation.z])

        # Get the rotational position in the form of a quaternion
        quaternion = numpy.array([vicon.transform.rotation.x,
                                 vicon.transform.rotation.y,
                                 vicon.transform.rotation.z,
                                 vicon.transform.rotation.w])

        return time, position, quaternion

    def estimate_state(self, vicon):
        """Calculate the current state based on the VICON information.

        Parameters
        ----------
        vicon : geometry_msgs.msg.TransformStamped
        """
        time, position, quaternion = self.convert_measurement(vicon)

        # Compute the velocities from the current and past measurements
        self.estimator.get_new_measurement(time, position, quaternion)

        # Predict future states (double integrator)
        # dt is not always constant, can't do it after publish_vicon()
        self.estimator.prior_update()

        # Update with measurements
        self.estimator.measurement_update()

        self.publish_vicon(vicon)

    def update_calibration(self, _):
        """
        Fetch new attitude correction parameters from the server.

        This function requires one argument due to ROS API.
        _ is a Service Request.
        """
        # get new parameters
        self.estimator.quat_corr[:] = (
            rospy.get_param(MODEL + '/' + MODEL + '/q_corr/x', 0.),
            rospy.get_param(MODEL + '/' + MODEL + '/q_corr/y', 0.),
            rospy.get_param(MODEL + '/' + MODEL + '/q_corr/z', 1.),
            rospy.get_param(MODEL + '/' + MODEL + '/q_corr/w', 0.))

        # # update the correction
        # self.quat_corr = tf.quaternion_multiply(q_update, self.quat_corr)
        log_str = "Loaded new pose calibration %s" % self.estimator.quat_corr
        rospy.loginfo(log_str)

    def get_current_state(self, _):
        """
        Calculate time difference, update time.

        This function requires a second argument due to ROS API.
        _ is a not used ros msg ( Type: Empty ).
        """
        state = StateVector()

        # Lock if the state is currently being updated, then block access
        with self.estimator.state_access_lock:
            old_time = self.estimator.time
            pos = self.estimator.pos.copy()
            vel = self.estimator.vel.copy()
            acc = self.estimator.acc.copy()
            quat = self.estimator.quat.copy()
            oemga_g = self.estimator.omega_g.copy()

        time = rospy.Time.now()
        state.header.stamp = time
        dt = time.to_time() - old_time

        state.pos = pos + dt * vel
        state.vel = vel
        state.acc = acc

        state.quat = apply_omega_to_quat(quat, oemga_g, dt)
        state.euler = tf.euler_from_quaternion(quat)

        state.omega_g = oemga_g
        state.omega_b = global_to_body(quat, oemga_g)

        print(state)

        return state

    def publish_vicon(self, vicon_msg):
        """Publish the current state information."""
        # Get the most recent VICON data
        state = StateVector()

        state.header.stamp.secs = vicon_msg.header.stamp.secs
        state.header.stamp.nsecs = vicon_msg.header.stamp.nsecs
        state.header.frame_id = vicon_msg.header.frame_id

        state.pos = self.estimator.pos
        state.vel = self.estimator.vel
        state.acc = self.estimator.acc

        state.quat = self.estimator.quat
        state.euler = self.estimator.rpy

        state.omega_g = self.estimator.omega_g
        state.omega_b = self.estimator.omega_b

        self.pub_state.publish(state)

        # For backwards compatibility...
        state_old_format = state_vector_to_data(state)

        state_old_format.header.stamp.secs = vicon_msg.header.stamp.secs
        state_old_format.header.stamp.nsecs = vicon_msg.header.stamp.nsecs
        state_old_format.header.frame_id = vicon_msg.header.frame_id

        # Publish the most recent data
        self.pub_state_old.publish(state_old_format)


if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('vicon_state_estimation')

    # read in model to be used
    model_param = rospy.search_param('model')
    if model_param:
        MODEL = rospy.get_param(model_param)
    else:
        raise EnvironmentError('No model parameter specified.')

    rospy.loginfo('Vicon model name: {0}'.format(MODEL))

    # Tuning parameters for Kalman Filter
    # increase in tau -> increase in c or d -> trust measurements less
    tau_est_trans = rospy.get_param('~tauEstTrans', 0.0001)
    tau_est_trans_dot = rospy.get_param('~tauEstTransDot', 0.007)
    tau_est_trans_dot_dot = rospy.get_param('~tauEstTransDotDot', 0.09)
    tau_est_rot = rospy.get_param('~tauEstRot', 0.005)
    tau_est_rot_dot = rospy.get_param('~tauEstRotDot', 0.07)

    # Make sure the parameters are set if default values used
    rospy.set_param('~tauEstTrans', tau_est_trans)
    rospy.set_param('~tauEstTransDot', tau_est_trans_dot)
    rospy.set_param('~tauEstTransDotDot', tau_est_trans_dot_dot)
    rospy.set_param('~tauEstRot', tau_est_rot)
    rospy.set_param('~tauEstRotDot', tau_est_rot_dot)

    # Create an instance of ViconCoordinates
    current_state = ViconCoordinates((tau_est_trans,
                                      tau_est_trans_dot,
                                      tau_est_trans_dot_dot,
                                      tau_est_rot,
                                      tau_est_rot_dot))

    # Do not exit until shutdown
    rospy.spin()
