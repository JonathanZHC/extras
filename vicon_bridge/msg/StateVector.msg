# StateVector
# Data necessary to communicate the current/desired state of the drone

# Header containing time and sequence
Header header

# Linear Position [m]
float64[3] pos

# Linear Velocities [m/s]
float64[3] vel

# Linear Accelerations [m/s/s]
float64[3] acc

# Euler Angles (roll,pitch,yaw) [rad]
float64[3] euler

# Time derivatives of euler Angles (roll,pitch,yaw) [rad/s]
float64[3] euler_dot

# Angular Velocity [rad/s]
float64[3] omega_g
float64[3] omega_b

# Orientation quaternion
float64[4] quat
