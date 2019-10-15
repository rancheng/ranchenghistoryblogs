---
layout: post
title: ROS with Python2 & Python3 confliction 
---

Python2 and python3 conflict is headache. This post shares about how to solve this conflict.

#### 1. Basic Setup
 - Problem on Yaml
```python
Traceback (most recent call last):
  File "/catkin_ws/src/ros_python3_issues/src/issue_yaml.py", line 3, in <module>
    import rospy
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rospy/__init__.py", line 47, in <module>
    from std_msgs.msg import Header
  File "/opt/ros/melodic/lib/python2.7/dist-packages/std_msgs/msg/__init__.py", line 1, in <module>
    from ._Bool import *
  File "/opt/ros/melodic/lib/python2.7/dist-packages/std_msgs/msg/_Bool.py", line 5, in <module>
    import genpy
  File "/opt/ros/melodic/lib/python2.7/dist-packages/genpy/__init__.py", line 34, in <module>
    from . message import Message, SerializationError, DeserializationError, MessageException, struct_I
  File "/opt/ros/melodic/lib/python2.7/dist-packages/genpy/message.py", line 44, in <module>
    import yaml
ModuleNotFoundError: No module named 'yaml'
```
 - solve
```sh
sudo apt-get install python3-pip python3-yaml
sudo pip3 install rospkg catkin_pkg
```
#### 2. cv_bridge Issue
  - If you’ll try to use cv_bridge for OpenCV you are most likely to get the following exception:
```python
Traceback (most recent call last):
  File "/catkin_ws/src/ros_python3_issues/src/issue_cv_bridge.py", line 23, in <module>
    ros_image = bridge.cv2_to_imgmsg(numpy.asarray(empty_image), encoding="rgb8") # convert PIL image to ROS image
  File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 259, in cv2_to_imgmsg
    if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:
  File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 91, in encoding_to_cvtype2
    from cv_bridge.boost.cv_bridge_boost import getCvType
ImportError: dynamic module does not define module export function (PyInit_cv_bridge_boost)
```

    - The issue is that cv_bridge is built only for python 2.7 so our python 3 interpreter is trying to use cv_bridge for 2.7 and fails, lets built it for Python 3:
    - First, let's install some tools we’ll need for the build process
    
```sh
sudo apt-get install python-catkin-tools python3-dev python3-numpy
```
Now, create new catkin_build_ws to avoid any future problems with catkin_make(assuming you are using it) and config catkin to use your python 3(3.6 in my case) when building packages:
 - install in system python
```sh
mkdir ~/catkin_build_ws && cd ~/catkin_build_ws
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin config --install
```
 - under miniconda.
```sh
mkdir ~/catkin_build_ws && cd ~/catkin_build_ws
catkin config -DPYTHON_EXECUTABLE=/home/ran/miniconda3/bin/python3 -DPYTHON_INCLUDE_DIR=/home/ran/miniconda3/bin/python3.6m -DPYTHON_LIBRARY=/home/ran/miniconda3/lib/libpython3.6m.so
catkin config --install
```
clone official vision_opencv repo:
```sh
mkdir src
cd src
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
```
finally, let’s build and source the package:
```sh
cd ~/catkin_build_ws
catkin build cv_bridge
source install/setup.bash --extend
```
That’s it! now you can use cv_bridge from Python 3!
#### 3. rostest Issue
```python
... logging to /root/.ros/log/rostest-477a6ee2f64d-4170.log
[ROSUNIT] Outputting test results to /root/.ros/test_results/ros_python3_issues/rostest-test_test_issue_rosunit.xml
Traceback (most recent call last):
  File "/catkin_ws/src/ros_python3_issues/test/issue_rosunit.py", line 18, in <module>
    rostest.rosrun('ros_python3_issues', 'issue_rosunit', TestROSUnitIssue)
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rostest/__init__.py", line 146, in rosrun
    result = rosunit.create_xml_runner(package, test_name, result_file).run(suite)
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rosunit/xmlrunner.py", line 275, in run
    result.print_report(stream, time_taken, out_s, err_s)
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rosunit/xmlrunner.py", line 202, in print_report
    stream.write(ET.tostring(self.xml(time_taken, out, err).getroot(), encoding='utf-8', method='xml'))
TypeError: write() argument must be str, not bytes[Testcase: testissue_rosunit] ... ok
[ROSTEST]-----------------------------------------------------------------------SUMMARY
 * RESULT: SUCCESS
 * TESTS: 0
 * ERRORS: 0
 * FAILURES: 0
```

This issue was actually already solved but for some reason, it is not yet available on the apt repository so we’ll simply install it from source:

```sh
cd ~/catkin_ws/src
git clone 
https://github.com/ros/roscd ..
catkin_make_isolated --install --pkg rosunit -DCMAKE_BUILD_TYPE=Release --install-space /opt/ros/melodic
```
That’s it!
