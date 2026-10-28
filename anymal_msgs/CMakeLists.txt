# Minimum cmake version
cmake_minimum_required(VERSION 3.5.1)

# Create project
project(anymal_msgs)

# Find catkin macros and libraries
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_generator_dds_idl REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AnymalState.msg"
  "msg/Contact.msg"
  "msg/ExtendedJointState.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_package()