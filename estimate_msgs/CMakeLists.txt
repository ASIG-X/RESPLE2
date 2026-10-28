cmake_minimum_required(VERSION 3.5)
project(estimate_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Spline.msg"
  "msg/Knot.msg"
  "msg/Estimate.msg"
  "msg/LoopClosure.msg"
  DEPENDENCIES std_msgs geometry_msgs
  LIBRARY_NAME ${PROJECT_NAME}
 )
 
ament_export_dependencies(rosidl_default_runtime)

ament_package()
