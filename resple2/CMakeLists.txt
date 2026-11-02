cmake_minimum_required(VERSION 3.5)
project(resple2)

set(_kinematics_default ON)
if(DEFINED ENV{RESPLE2_ENABLE_KINEMATICS})
  if(NOT "$ENV{RESPLE2_ENABLE_KINEMATICS}" MATCHES "^(ON|OFF)$")
    message(FATAL_ERROR "RESPLE2_ENABLE_KINEMATICS environment variable must be ON or OFF")
  endif()
  set(_kinematics_default "$ENV{RESPLE2_ENABLE_KINEMATICS}")
endif()
option(RESPLE2_ENABLE_KINEMATICS "Build legged-robot kinematic support" ${_kinematics_default})
if(RESPLE2_ENABLE_KINEMATICS)
  set(_kinematics_enabled ON)
else()
  set(_kinematics_enabled OFF)
endif()
if(DEFINED ENV{RESPLE2_ENABLE_KINEMATICS} AND
   NOT "$ENV{RESPLE2_ENABLE_KINEMATICS}" STREQUAL "${_kinematics_enabled}")
  message(FATAL_ERROR "RESPLE2_ENABLE_KINEMATICS differs between the environment and CMake cache. Pass -DRESPLE2_ENABLE_KINEMATICS=$ENV{RESPLE2_ENABLE_KINEMATICS}")
endif()
set(ENV{RESPLE2_ENABLE_KINEMATICS} "${_kinematics_enabled}")
message(STATUS "RESPLE2 kinematic support: ${_kinematics_enabled}")

set(CMAKE_BUILD_TYPE "DEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++17 -O3 -march=native -mno-avx -mno-avx2 -pthread -fopenmp -Wall -DEIGEN_INITIALIZE_MATRICES_BY_NAN")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS} -std=c++17 -O2 -g -march=native -mno-avx -mno-avx2  -pthread -fopenmp -Wall -DEIGEN_INITIALIZE_MATRICES_BY_NAN")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS} -std=c++17 -O3 -march=native -mno-avx -mno-avx2 -pthread -fopenmp -Wall -DEIGEN_INITIALIZE_MATRICES_BY_NAN")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})
get_filename_component(RESPLENU_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_definitions(-DRESPLENU_SOURCE_DIR="${RESPLENU_SOURCE_DIR}")

if(DEFINED ENV{ROS_DISTRO} AND "$ENV{ROS_DISTRO}" STREQUAL "humble")
  add_definitions(-DROS_DISTRO_HUMBLE)
else()
  add_definitions(-DROS_DISTRO_JAZZY)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(livox_interfaces REQUIRED)
find_package(livox_ros_driver REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(estimate_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_lint_auto REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(image_transport REQUIRED)
find_package(ament_index_cpp REQUIRED)
if(RESPLE2_ENABLE_KINEMATICS)
  find_package(anymal_msgs REQUIRED)
  find_package(unitree_msgs REQUIRED)
  find_package(pinocchio REQUIRED)
endif()
find_package(Boost REQUIRED COMPONENTS thread)
find_package(SuiteSparse REQUIRED MODULE)
find_package(Ceres REQUIRED)

include(FindOpenMP)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

ADD_DEFINITIONS(-DBOOST_MPL_LIMIT_LIST_SIZE=50)
ADD_DEFINITIONS(-DBOOST_MPL_LIMIT_VECTOR_SIZE=50)
ADD_DEFINITIONS(-DBOOST_VARIANT_LIMIT_TYPES=50)
SET(CFLAGS_DEPENDENCIES)
LIST(APPEND CFLAGS_DEPENDENCIES "-DBOOST_MPL_LIMIT_LIST_SIZE=50")
LIST(APPEND CFLAGS_DEPENDENCIES "-DBOOST_MPL_LIMIT_VECTOR_SIZE=50")
LIST(APPEND CFLAGS_DEPENDENCIES "-DBOOST_VARIANT_LIMIT_TYPES=50")

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_library(${PROJECT_NAME} INTERFACE)
ament_lint_auto_find_test_dependencies()
include_directories(
	include
	${EIGEN3_INCLUDE_DIR}
	${PCL_INCLUDE_DIRS}
	${SUITESPARSE_INCLUDE_DIRS})
add_executable(Mapping 
	src/Mapping.cpp 
	src/camera/equidistant_camera.cpp 
	src/camera/ImgPreprocessing.cpp 
	src/camera/pinhole_camera.cpp 	
	src/GlobalRefinement.cpp)
add_executable(RESPLE2
	src/RESPLE2.cpp
	include/voxel_map.cpp 
	src/camera/equidistant_camera.cpp
	src/camera/ImgPreprocessing.cpp 
	src/camera/pinhole_camera.cpp 
	src/VisualFrontend.cpp
	src/Estimator.cpp)

target_compile_definitions(RESPLE2 PRIVATE
  RESPLE2_ENABLE_KINEMATICS=$<BOOL:${RESPLE2_ENABLE_KINEMATICS}>)
if(RESPLE2_ENABLE_KINEMATICS)
  target_include_directories(RESPLE2 PRIVATE ${pinocchio_INCLUDE_DIRS})
  target_link_libraries(RESPLE2 ${pinocchio_LIBRARIES})
  ament_target_dependencies(RESPLE2 anymal_msgs unitree_msgs)
endif()

target_include_directories(Mapping PRIVATE /usr/local/include)
target_include_directories(RESPLE2 PRIVATE /usr/local/include)
ament_target_dependencies(${PROJECT_NAME} INTERFACE
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  std_srvs
  livox_interfaces
  livox_ros_driver
  livox_ros_driver2
  estimate_msgs
  pcl_conversions
  ament_lint_auto
  tf2_ros
  cv_bridge
  image_transport
  ament_index_cpp
)
target_link_libraries(RESPLE2
		${PROJECT_NAME}
		${PCL_LIBRARIES}
		${YAML_CPP_LIBRARIES}
		${OpenMP_LIBS}
		${OpenCV_LIBRARIES}
		${Boost_LIBRARIES}
)
target_link_libraries(Mapping
		${PROJECT_NAME}
		${PCL_LIBRARIES}
		${YAML_CPP_LIBRARIES}
		${OpenCV_LIBRARIES}
		cholmod
)
install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY cmake_modules
  DESTINATION share/${PROJECT_NAME})  
add_executable(ibtc_online
    src/ibtc/ibtc_online.cpp
    src/ibtc/ibtc.cpp
    src/camera/ImgPreprocessing.cpp
    src/camera/pinhole_camera.cpp
    src/camera/equidistant_camera.cpp)
target_include_directories(ibtc_online PRIVATE /usr/local/include)
ament_target_dependencies(ibtc_online
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  pcl_conversions
  cv_bridge
  std_srvs
  image_transport
  ament_index_cpp
  estimate_msgs
)
target_link_libraries(ibtc_online
    ${PCL_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CERES_LIBRARIES})

install(TARGETS RESPLE2
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS Mapping
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS ibtc_online
  DESTINATION lib/${PROJECT_NAME})
ament_export_dependencies(rclcpp yaml-cpp message_runtime livox_ros_driver livox_ros_driver2 livox_interfaces)
if(RESPLE2_ENABLE_KINEMATICS)
  ament_export_dependencies(anymal_msgs unitree_msgs)
endif()

ament_package()
