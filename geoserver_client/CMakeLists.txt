cmake_minimum_required(VERSION 3.5)
project(geoserver_client)

# Default to C++17
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geoserver_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Find GDAL using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(GDAL REQUIRED gdal)
if(GDAL_FOUND)
  include_directories(${GDAL_INCLUDE_DIRS})
  message(STATUS "GDAL found: ${GDAL_VERSION}")
else()
  # Fallback: try to find GDAL manually
  find_library(GDAL_LIBRARY NAMES gdal)
  find_path(GDAL_INCLUDE_DIR gdal.h PATH_SUFFIXES gdal)
  if(GDAL_LIBRARY AND GDAL_INCLUDE_DIR)
    set(GDAL_FOUND TRUE)
    include_directories(${GDAL_INCLUDE_DIR})
    message(STATUS "GDAL found (manual): ${GDAL_LIBRARY}")
  else()
    message(FATAL_ERROR "GDAL not found")
  endif()
endif()

# Find CURL
find_package(CURL REQUIRED)
if(CURL_FOUND)
  include_directories(${CURL_INCLUDE_DIRS})
  message(STATUS "CURL found")
else()
  # Fallback: try to find CURL manually
  find_library(CURL_LIBRARY NAMES curl)
  find_path(CURL_INCLUDE_DIR curl/curl.h)
  if(CURL_LIBRARY AND CURL_INCLUDE_DIR)
    set(CURL_FOUND TRUE)
    include_directories(${CURL_INCLUDE_DIR})
    set(CURL_LIBRARIES ${CURL_LIBRARY})
    message(STATUS "CURL found (manual): ${CURL_LIBRARY}")
  else()
    message(FATAL_ERROR "CURL not found")
  endif()
endif()

# Add executable
add_executable(geoserver_map_tif_node src/geoserver_client.cpp)

# Include directories
ament_target_dependencies(geoserver_map_tif_node
  rclcpp
  sensor_msgs
  geoserver_msgs
  cv_bridge
)

target_include_directories(geoserver_map_tif_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  ${GDAL_INCLUDE_DIR}
  ${CURL_INCLUDE_DIRS}
)

target_link_libraries(geoserver_map_tif_node
  ${OpenCV_LIBS}
  ${GDAL_LIBRARIES}
  ${CURL_LIBRARIES}
)

# Install executable
install(TARGETS
  geoserver_map_tif_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install resource files
install(
  FILES
    package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

