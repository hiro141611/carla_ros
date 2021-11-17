cmake_minimum_required(VERSION 3.0.2)
project(hiro)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  tf2
  tf2_ros
  pcl_ros
  sensor_msgs
)

find_package(Boost REQUIRED COMPONENTS system)

add_message_files(
  FILES
  test.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)