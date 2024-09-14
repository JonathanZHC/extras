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
import time
import math

import numpy
import tf.transformations as tf
from quaternions import (apply_omega_to_quat,
                                                global_to_body)

from vicon_bridge.srv import GetState
from std_msgs.msg import Empty
from geometry_msgs.msg import TransformStamped
from vicon_bridge.msg import StateData, StateVector, Command
from message_filters import Subscriber, ApproximateTimeSynchronizer
from state_estimation import state_vector_to_data, \
    StateEstimator

# Needed to send numpy.array as a msg
from rospy.numpy_msg import numpy_msg

    
def pwm2thrust(pwm):
    """Convert pwm to thrust using a quadratic function."""
    # abc-formula coefficients for thrust to pwm conversion
    # pwm = a * thrust^2 + b * thrust + c
    a_coeff = -1.1264
    b_coeff = 2.2541
    c_coeff = 0.0209
    pwm_max = 65535.0

    pwm_scaled = pwm / pwm_max
    # solve quadratic equation using abc formula
    thrust = (-b_coeff + numpy.sqrt(b_coeff**2 - 4 * a_coeff * (c_coeff - pwm_scaled))) / (2 * a_coeff)
    return thrust


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
            Subscribed to raw data from vicon.
        sub_controller: rospy.Subscriber
            Subscribed to input signal from controller.
        pub_state: rospy.Publisher
            Publishes estimated state.
        pub_state_old: rospy.Publisher
            Publish previous state.
        sub_cal: rospy.Subscriber
            Get calibration data from parameter server.
        srv_vicon: rospy.Service
            Fetch current state of object in vicon.
    """

    def __init__(self, filter_parameters, observer_type = 'simple', model_file = None, publish_rate = 200, sim = True):
        """Initializaiton."""
        super(ViconCoordinates, self).__init__()

        # Lock for access to state
        # Makes sure the service does not conflict with normal updates
        self.estimator = StateEstimator(filter_parameters, observer_type, model_file)

        model_name = MODEL + '/' + MODEL

        # Initialize old time (for time difference calculation)
        self.time_old = 0 

        # Interface to input / output (pub / sub)
        if sim:
            # Input: Subscribe to the collected data from logfile
            self.sub_vicon = rospy.Subscriber('logdata_state', TransformStamped, self.convert_measurement_state)
            self.sub_controller = rospy.Subscriber('logdata_input', Command, self.estimate_state)
            # Output: Publish to the /estimated_state_sim topic
            self.pub_state = rospy.Publisher('estimated_state_sim', numpy_msg(StateVector), queue_size=1)
            self.pub_state_old = rospy.Publisher('current_coordinates', StateData, queue_size=1)
        else:
            # Input: Subscribe to the VICON information
            self.sub_vicon = rospy.Subscriber('/vicon/' + model_name, TransformStamped, self.convert_measurement_state)
            self.sub_controller = rospy.Subscriber('/controller_command', Command, self.estimate_state)
            # Launch a service which gets the current state (persistent connection --> some extra speed)
            self.srv_vicon = rospy.Service(MODEL + '_get_state', GetState, self.get_current_state, True)
            # Output: Publish to the /estimated_state topic
            self.pub_state = rospy.Publisher('estimated_state', numpy_msg(StateVector), queue_size=1)
            self.pub_state_old = rospy.Publisher('current_coordinates', StateData, queue_size=1)

        # Subscribe to the /new_calibration topic (updates parameters after calibration)
        self.sub_cal = rospy.Subscriber('/new_calibration', Empty, self.update_calibration)

        # Initialize the state parameters
        self.vicon = TransformStamped()
        self.time_state = 0
        self.raw_position = numpy.array([0, 0, 0])
        self.raw_quaternion = numpy.array([0, 0, 0, 1])
        
        # Initialize pubish rate
        self.publish_rate = publish_rate
        # Define upper limit for acceptable slop
        self.slop_limit = 1 / self.publish_rate

        self.estimater_started = False
        self.synchronized = False

        # Wait for vicon to actually publish
        rospy.loginfo('vicon.py: Waiting for vicon to publish {0}'
                      .format(MODEL))
        while self.sub_vicon.get_num_connections() == 0:
            rospy.sleep(1)
        rospy.loginfo('vicon.py: Got data for {0}'.format(MODEL))

        # Initialize attitude correction
        self.estimator.quat_corr[:] = (
            rospy.get_param(model_name + '/q_corr/x', 0.),
            rospy.get_param(model_name + '/q_corr/y', 0.),
            rospy.get_param(model_name + '/q_corr/z', 0.),
            rospy.get_param(model_name + '/q_corr/w', 1.))
        
        self.prev_t = None


        "----------for test----------"
        self.counter = 0
        self.delta_t_0_ave = 0
        self.delta_t_0_var = 0
        self.delta_t_1_ave = 0
        self.delta_t_1_var = 0
        self.delta_t_2_ave = 0
        self.delta_t_2_var = 0
        self.delta_t_3_ave = 0
        self.delta_t_3_var = 0
        self.delta_t_sum_ave = 0
        self.delta_t_sum_var = 0
        "----------for test----------"

    def convert_measurement_state(self, vicon):
        """Convert vicon measurements (prior states).

        Parameters
        ----------
        vicon : geometry_msgs.msg.TransformStamped

        Returns
        -------
        time : float
        position : ndarray
        quaternion : ndarray
        """
    
        #rospy.loginfo("convert_measurement_state called") # for test
        self.vicon = vicon
        
        # Record the time at which the state was determined
        self.time_state = (float(vicon.header.stamp.secs)
                + float(vicon.header.stamp.nsecs) * 1e-9)

        # Get the translational position from VICON
        self.raw_position = numpy.array([vicon.transform.translation.x,
                               vicon.transform.translation.y,
                               vicon.transform.translation.z])

        # Get the rotational position in the form of a quaternion
        self.raw_quaternion = numpy.array([vicon.transform.rotation.x,
                                 vicon.transform.rotation.y,
                                 vicon.transform.rotation.z,
                                 vicon.transform.rotation.w])
           
    def convert_measurement_input(self, controller):
        """Convert input from controller.

        Parameters
        ----------
        controller : Command (self defined)

        Returns
        -------
        time : float
        input : ndarray (CMD_ROLL, CMD_PITCH, CMD_YAW, CMD_THRUST)
        """
        
        # Record the time at which the input was determined
        seq = controller.header.seq
        time = (float(controller.header.stamp.secs)
                + float(controller.header.stamp.nsecs) * 1e-9)
        
        # Get all inputs from controller
        input = numpy.array([controller.CMD_ROLL,
                            controller.CMD_PITCH,
                            controller.CMD_YAW,
                            pwm2thrust(controller.CMD_PWM)])
        
        # After first time recieved input from controller, 
        # switch state to publish estimated value
        self.estimater_started = True
         
        return seq, time, input

    def estimate_state(self, controller):     
        """Calculate the current state based on the VICON information.

        Parameters
        ----------
        vicon : geometry_msgs.msg.TransformStamped
        controller : vicon_bridge.msg.Command (self-defined)
        """
        

        "----------for test----------"
        self.counter += 1
        t1 = time.time()
        if self.prev_t:
            delta_t_0 = t1 - self.prev_t
            self.delta_t_0_ave_old = self.delta_t_0_ave
            self.delta_t_0_ave += (delta_t_0 - self.delta_t_0_ave_old) / self.counter
            self.delta_t_0_var = (self.delta_t_0_var * (self.counter-2) + (delta_t_0 - self.delta_t_0_ave_old) * (delta_t_0 - self.delta_t_0_ave)) / (self.counter-1)
            rospy.loginfo("ave time since start of last cycle: %f. " %self.delta_t_0_ave)
            rospy.loginfo("var of time since start of last cycle: %f. " %math.sqrt(self.delta_t_0_var))
        "----------for test----------"


        # Call Callback of the input
        #seq_state, time_state, position, quaternion = self.convert_measurement_state(vicon)
        seq_input, time_input, input = self.convert_measurement_input(controller)
        time_difference = time_input - self.time_state

        # Report timestamps
        #rospy.loginfo("Received timestamps: state: %f, input: %f. " %(self.time_state, time_input))
        #rospy.loginfo("Timestamp difference: %f. " %time_difference)





        "----------for test----------" # around 0.0015s
        t2 = time.time()
        delta_t_1 = t2 - t1
        self.delta_t_1_ave_old = self.delta_t_1_ave
        self.delta_t_1_ave += (delta_t_1 - self.delta_t_1_ave_old) / self.counter
        self.delta_t_1_var = (self.delta_t_1_var * (self.counter-2) + (delta_t_1 - self.delta_t_1_ave_old) * (delta_t_1 - self.delta_t_1_ave)) / (self.counter-1)
        rospy.loginfo("ave cycle time for period 1 of estimator: %f. " %self.delta_t_1_ave)
        rospy.loginfo("var of cycle time for period 1 of estimator: %f. " %math.sqrt(self.delta_t_1_var))
        "----------for test----------"





        # Check Synchronization 
        # Interface to be used: self.synchronized
        if time_difference < self.slop_limit:
            # Compute the velocities from the current and past measurements
            self.synchronized = True
            #print("Messsages are synchronized, will publish estimated data from estimator")
        else:
            self.synchronized = False
            #raise EnvironmentError('Messsages are not synchronized.')
            #print("Warning!!!Warning!!!Warning!!!Warning!!!Messsages are not synchronized!Warning!!!Warning!!!Warning!!!Warning!!!")

        self.estimator.get_new_measurement(self.time_state, self.raw_position, self.raw_quaternion, input)
        




        "----------for test----------" # around 0.0008s
        t3 = time.time()
        delta_t_2 = t3 - t2
        self.delta_t_2_ave_old = self.delta_t_2_ave
        self.delta_t_2_ave += (delta_t_2 - self.delta_t_2_ave_old) / self.counter
        self.delta_t_2_var = (self.delta_t_2_var * (self.counter-2) + (delta_t_2 - self.delta_t_2_ave_old) * (delta_t_2 - self.delta_t_2_ave)) / (self.counter-1)
        rospy.loginfo("ave cycle time for period 2 of estimator: %f. " %self.delta_t_2_ave)
        rospy.loginfo("var of cycle time for period 2 of estimator: %f. " %math.sqrt(self.delta_t_2_var))
        "----------for test----------"




        # Use various filter / the model knowledge / the measured data to estimate the real state
        if self.estimator.observer == 'simple':
            # Predict future states (double integrator)
            # dt is not always constant, can't do it after publish_vicon()
            self.estimator.prior_update()
            # Update with measurements
            self.estimator.measurement_update()
        elif self.estimator.observer == 'EKF':
            # EKF predict & update
            self.estimator.EKF_update()
        elif self.estimator.observer == 'UKF':
            # UKF predict & update
            self.estimator.UKF_update()

        self.publish_vicon(self.vicon)





        "----------for test----------" # around 0.0008s
        t4 = time.time()
        delta_t_3 = t4 - t3
        self.delta_t_3_ave_old = self.delta_t_3_ave
        self.delta_t_3_ave += (delta_t_3 - self.delta_t_3_ave_old) / self.counter
        self.delta_t_3_var = (self.delta_t_3_var * (self.counter-2) + (delta_t_3 - self.delta_t_3_ave_old) * (delta_t_3 - self.delta_t_3_ave)) / self.counter
        rospy.loginfo("ave cycle time for period 3 of estimator: %f. " %self.delta_t_3_ave)
        rospy.loginfo("var of cycle time for period 3 of estimator: %f. " %math.sqrt(self.delta_t_3_var))
        delta_t_sum = t4 - t1
        self.delta_t_sum_ave_old = self.delta_t_sum_ave
        self.delta_t_sum_ave += (delta_t_sum - self.delta_t_sum_ave_old) / self.counter
        self.delta_t_sum_var = (self.delta_t_sum_var * (self.counter-2) + (delta_t_sum - self.delta_t_sum_ave_old) * (delta_t_sum - self.delta_t_sum_ave)) / self.counter
        rospy.loginfo("ave cycle time of estimator: %f. " %self.delta_t_sum_ave)
        rospy.loginfo("var of cycle time of estimator: %f. " %math.sqrt(self.delta_t_sum_var))
        "----------for test----------"
        self.prev_t = t1
        "----------for test----------"





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

        # Publish the estimated state
        self.pub_state.publish(state)

        # For backwards compatibility...
        state_old_format = state_vector_to_data(state)

        state_old_format.header.stamp.secs = vicon_msg.header.stamp.secs
        state_old_format.header.stamp.nsecs = vicon_msg.header.stamp.nsecs
        state_old_format.header.frame_id = vicon_msg.header.frame_id

        # Publish the most recent data
        self.pub_state_old.publish(state_old_format)


if __name__ == '__main__':

    # Define publish rate
    publish_rate = 60

    # define type of observer
    observer = 'EKF' # 'simple' / 'EKF' / 'UKF
    
    # define file path of identified model
    model_file = '/home/haocheng/Experiments/figure_8/merge_model.json'



    # Initialize the ROS node
    rospy.init_node('vicon_state_estimation')

    # read in model to be used
    model_param = rospy.search_param('model')
    if model_param:
        MODEL = rospy.get_param(model_param)
    else:
        raise EnvironmentError('No model parameter specified.')
    rospy.loginfo('Vicon model name: {0}'.format(MODEL))

    # use simulation channel or real-run channel
    sim_param = rospy.search_param('sim')
    if sim_param:
        sim = rospy.get_param(sim_param)
        if sim:
            rospy.loginfo('Channel of Vicon node: SIMULATION channel')
        else:
            rospy.loginfo('Channel of Vicon node: REAL-RUN channel')
    else:
        raise EnvironmentError('No simulation parameter specified.')

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
                                      tau_est_rot_dot), observer, model_file, publish_rate, sim)

    # Do not exit until shutdown
    rospy.spin()
