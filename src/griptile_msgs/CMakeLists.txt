cmake_minimum_required(VERSION 3.8)
project(griptile_msgs)
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SensorInfo.msg"
  "msg/ContactReference.msg"
  "srv/SetContactReference.srv"
  "srv/ManageSerial.srv"
  "msg/TactileRow44.msg"
  "msg/TactileRow44Array.msg"
  "msg/TactileGridState.msg"
  DEPENDENCIES std_msgs)
ament_export_dependencies(rosidl_default_runtime)
install(FILES LICENSE DESTINATION share/${PROJECT_NAME})

ament_package()
