#!/usr/bin/env python

# Import modules
import numpy as np
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from pr2_robot.srv import *


def service_call(pick_place_routine):
    object_name = String()
    pick_pose = Pose()
    test_scene_num = Int32()
    arm_name = String()
    place_pose = Pose()

    test_scene_num.data = int(1)
    object_name.data = str('biscuits')
    arm_name.data = str('right')
    pick_pose.position.x = 0.5422592759132385
    pick_pose.position.y = 0.2423889935016632
    pick_pose.position.z = 0.7055615782737732

    pick_pose.orientation.x=1
    pick_pose.orientation.y=0
    pick_pose.orientation.z=0
    pick_pose.orientation.w=0

    place_pose.position.x = 0.0
    place_pose.position.y = -0.71
    place_pose.position.z = 0.605

    place_pose.orientation.x=1
    place_pose.orientation.y=0
    place_pose.orientation.z=0
    place_pose.orientation.w=0

    resp = pick_place_routine(
        test_scene_num, object_name, arm_name, pick_pose, place_pose)
    print ("Response: ", resp.success)


if __name__ == '__main__':
    # TODO: ROS node initialization
    rospy.init_node('test_node')
    rospy.wait_for_service('pick_place_routine')
    pick_place_routine = rospy.ServiceProxy(
        'pick_place_routine', PickPlace)
    service_call(pick_place_routine)
