#include <ros/ros.h>
#include <thread>
#include "vicon_bridge/vicon_receiver.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "vicon");
//  ViconReceiver vr;
//  ros::spin();
  ros::NodeHandle nh;
  ros::NodeHandle np("~");
  ros::AsyncSpinner aspin(1);
  aspin.start();
  ViconReceiver vr(np, nh);
 vr.startGrabbing();
  aspin.stop();
  return 0;
}