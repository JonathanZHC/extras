cmake_minimum_required(VERSION 2.8.3)
project(vicon_bridge)

find_package(catkin REQUIRED COMPONENTS
    message_generation
    dynamic_reconfigure
    geometry_msgs
    roscpp
    rospy
    tf
    diagnostic_updater
    nodelet
    std_msgs
    message_filters
)

find_package(Boost REQUIRED COMPONENTS thread)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# Generate messages and services
add_message_files(FILES
    Marker.msg
    Markers.msg
    TfDistortInfo.msg
    StateData.msg
    StateVector.msg
    Command.msg
)

add_service_files(FILES
    viconCalibrateSegment.srv
    viconGrabPose.srv
    GetState.srv
)

generate_messages(DEPENDENCIES 
    geometry_msgs
    std_msgs
)

# Generate dynamic parameters
generate_dynamic_reconfigure_options(
  cfg/tf_distort.cfg
)

catkin_package(CATKIN_DEPENDS 
    dynamic_reconfigure
    geometry_msgs
    message_runtime 
    roscpp
    std_msgs
    LIBRARIES vicon_bridge_lib
    DEPENDS nodelet
)

# check for platform and use the right Vicon SDK
if(CMAKE_SIZEOF_VOID_P MATCHES "8")
    set(VICON_SDK             vicon_sdk/Vicon_SDK_1.3_Linux/64-bit/)
    set(VICON_SDK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/${VICON_SDK})
    set(VICON_SDK_LINK_DIR    ${PROJECT_SOURCE_DIR}/${VICON_SDK})
elseif(CMAKE_SIZEOF_VOID_P MATCHES "4")
    set(VICON_SDK             vicon_sdk/Vicon_SDK_1.3_Linux/32-bit/)
    set(VICON_SDK_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/${VICON_SDK})
    set(VICON_SDK_LINK_DIR    ${PROJECT_SOURCE_DIR}/${VICON_SDK})
else(CMAKE_SIZEOF_VOID_P MATCHES "8")
    message(FATAL_ERROR "unable to detect platform")
endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
find_library(VICON_SDK_LIBRARY ViconDataStreamSDK_CPP PATHS ${VICON_SDK_LINK_DIR})

include_directories(src include ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${VICON_SDK_INCLUDE_DIR})

add_library(msvc_bridge src/msvc_bridge.cpp)

add_library(vicon_bridge_lib src/vicon_bridge.cpp)
target_link_libraries(vicon_bridge_lib
    msvc_bridge
    ${VICON_SDK_LIBRARY}
    ${catkin_LIBRARIES}
)
add_dependencies(vicon_bridge_lib ${PROJECT_NAME}_generate_messages_cpp)

add_executable(vicon_bridge src/vicon_bridge_main.cpp)
target_link_libraries(vicon_bridge
    vicon_bridge_lib
    ${catkin_LIBRARIES}
)
add_dependencies(vicon_bridge ${PROJECT_NAME}_generate_messages_cpp)

add_executable(calibrate src/calibrate_segment.cpp)
target_link_libraries(calibrate
   ${catkin_LIBRARIES}
)
add_dependencies(calibrate ${PROJECT_NAME}_generate_messages_cpp)

add_executable(tf_distort src/tf_distort.cpp)
target_link_libraries(tf_distort
   ${catkin_LIBRARIES}
   ${Boost_LIBRARIES}
)
add_dependencies(tf_distort ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_gencfg)

add_executable(testclient src/ViconDataStreamSDK_CPPTest.cpp)
target_link_libraries(testclient 
    ${VICON_SDK_LIBRARY}
    ${Boost_LIBRARIES}
)
add_dependencies(testclient ${PROJECT_NAME}_generate_messages_cpp)

install(TARGETS 
    vicon_bridge_lib
    vicon_bridge
    calibrate
    tf_distort
    testclient
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(FILES vicon_bridge_nodelet.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
