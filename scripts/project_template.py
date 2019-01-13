#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
import random


angle1 = Float64()
angle2 = Float64()
angle1.data = 1.57
angle2.data = -1.57
map_ready = True
mapping_stage = 0
start_time = 0
outputed = False
world_num = 3

# Helper function to get surface normals


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy(
        '/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages


def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(
        pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(
        place_pose)
    return yaml_dict

# Helper function to output to yaml file


def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber


def pcl_callback(pcl_msg):
    global map_ready
    global mapping_stage
    global start_time
    # Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    # TODO: Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    x = 0.005
    outlier_filter.set_std_dev_mul_thresh(x)
    cloud = outlier_filter.filter()
    # TODO: Voxel Grid Downsampling
    LEAF_SIZE = 0.01
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    obs_cloud = cloud_filtered = vox.filter()

    # TODO: PassThrough Filter
    # passthrough filter of obs

    passthrough = obs_cloud.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 0.62
    passthrough.set_filter_limits(axis_min, axis_max)

    obs_cloud = passthrough.filter()
    obvoidence_pub.publish(pcl_to_ros(obs_cloud))
    if map_ready:
        passthrough = cloud_filtered.make_passthrough_filter()
        filter_axis = 'z'
        passthrough.set_filter_field_name(filter_axis)
        axis_min = 0.62
        axis_max = 1.1
        passthrough.set_filter_limits(axis_min, axis_max)

        cloud_filtered = passthrough.filter()

        passthrough = cloud_filtered.make_passthrough_filter()
        filter_axis = 'y'
        passthrough.set_filter_field_name(filter_axis)
        axis_min = -0.5
        axis_max = 0.5
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud_filtered = passthrough.filter()
        # TODO: RANSAC Plane Segmentation
        seg = cloud_filtered.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)

        max_distance = 0.01
        seg.set_distance_threshold(max_distance)
        # TODO: Extract inliers and outliers
        inliers, coefficients = seg.segment()
        extracted_inliers = cloud_filtered.extract(inliers, negative=False)
        extracted_outliers = cloud_filtered.extract(inliers, negative=True)
        # TODO: Euclidean Clustering
        white_cloud = XYZRGB_to_XYZ(extracted_outliers)
        tree = white_cloud.make_kdtree()
        # Create a cluster extraction object
        ec = white_cloud.make_EuclideanClusterExtraction()
        # Set tolerances for distance threshold
        # as well as minimum and maximum cluster size (in points)
        # NOTE: These are poor choices of clustering parameters
        # Your task is to experiment and find values that work for segmenting objects.
        ec.set_ClusterTolerance(0.05)    # bigger makes more point as a p
        ec.set_MinClusterSize(0)
        ec.set_MaxClusterSize(1000)
        # Search the k-d tree for clusters
        ec.set_SearchMethod(tree)
        # Extract indices for each of the discovered clusters
        cluster_indices = ec.Extract()

        miss_list = []
        for i in range(len(cluster_indices)):
            if len(cluster_indices[i]) < 30:
                miss_list.append(i)

        # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
        # Assign a color corresponding to each segmented object in scene
        cluster_color = get_color_list(len(cluster_indices))
        color_cluster_point_list = []
        for j, indices in enumerate(cluster_indices):
            for i, indice in enumerate(indices):
                color_cluster_point_list.append([white_cloud[indice][0],
                                                 white_cloud[indice][1],
                                                 white_cloud[indice][2],
                                                 rgb_to_float(cluster_color[j])])

        # Create new cloud containing all clusters, each with unique color
        cluster_cloud = pcl.PointCloud_PointXYZRGB()
        cluster_cloud.from_list(color_cluster_point_list)
        # print(cluster_indices)
        # TODO: Convert PCL data to ROS messages
        ros_cluster_cloud = pcl_to_ros(cluster_cloud)
        # TODO: Publish ROS messages

        # Exercise-3 TODOs:

        # Classify the clusters! (loop through each detected cluster one at a time)
        detected_objects_labels = []
        detected_objects = []

        # needed to be alternate when change the world
        for index, pts_list in enumerate(cluster_indices):
            if index in miss_list:
                continue
            # Grab the points for the cluster from the extracted outliers (cloud_objects)
            pcl_cluster = extracted_outliers.extract(pts_list)
            # TODO: convert the cluster from pcl to ROS using helper function
            ros_cluster = pcl_to_ros(pcl_cluster)
            # Extract histogram features
            # TODO: complete this step just as is covered in capture_features.py

            chists = compute_color_histograms(ros_cluster, using_hsv=True)
            normals = get_normals(ros_cluster)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))

            # Make the prediction, retrieve the label for the result
            # and add it to detected_objects_labels list
            prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
            label = encoder.inverse_transform(prediction)[0]
            detected_objects_labels.append(label)

            # Publish a label into RViz
            label_pos = list(white_cloud[pts_list[0]])
            label_pos[2] += .4
            object_markers_pub.publish(make_label(label, label_pos, index))

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = ros_cluster
            detected_objects.append(do)

        rospy.loginfo('Detected {} objects: {}'.format(
            len(detected_objects_labels), detected_objects_labels))

        # Publish the list of detected objects
        # This is the output you'll need to complete the upcoming project!
        detected_objects_pub.publish(detected_objects)
        cloud_puber.publish(pcl_to_ros(extracted_outliers))
        # cloud_puber.publish(ros_cluster_cloud)
        print('point cloud published')
        detected_objects_list = detected_objects
        try:
            pr2_mover(detected_objects_list)
        except rospy.ROSInterruptException:
            pass

    # rotate PR2 to capture the collision map
    if not map_ready and mapping_stage == 0:
        start_time = rospy.get_time()
        pr2_world_joint_pub.publish(angle1)
        mapping_stage += 1

    if not map_ready and mapping_stage == 1:
        if rospy.get_time() - start_time >= 15:
            mapping_stage += 1
            pr2_world_joint_pub.publish(angle2)

    if not map_ready and mapping_stage == 2:
        if rospy.get_time() - start_time >= 45:
            mapping_stage += 1
            pr2_world_joint_pub.publish(0)

    if not map_ready and mapping_stage == 3:
        if rospy.get_time() - start_time >= 65:
            mapping_stage += 1
            map_ready = True


# function to load parameters and request PickPlace service


def pr2_mover(object_list):

    # TODO: Initialize variables
    global outputed
    global world_num
    object_name = String()
    pick_pose = Pose()
    test_scene_num = Int32()
    arm_name = String()
    place_pose = Pose()
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param = rospy.get_param('/dropbox')
    # TODO: Parse parameters into individual variables
    test_scene_num = Int32()
    test_scene_num.data = world_num
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    dict_list = []
    labels = []
    centroids = []
    for item in object_list:
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        labels.append(item.label)
        points_arr = ros_to_pcl(item.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        centroid_py_type = [np.asscalar(centroid[0]), np.asscalar(
            centroid[1]), np.asscalar(centroid[2])]
        centroids.append(centroid_py_type)
        # TODO: Create 'place_pose' for the object

        # make yaml list
        object_name.data = str(labels[-1])                   # get obj's label
        # confirm obj pick pos
        pick_pose.position.x = centroids[-1][0]
        pick_pose.position.y = centroids[-1][1]
        pick_pose.position.z = centroids[-1][2]
        pick_pose.orientation.x = 0
        pick_pose.orientation.y = 0
        pick_pose.orientation.z = 0
        pick_pose.orientation.w = 1

        # TODO: Assign the arm to be used for pick_place
        for items in object_list_param:
            if items['name'] == object_name.data:
                dropbox_group = items['group']

        if dropbox_group == dropbox_param[0]['group']:
            arm_name.data = dropbox_param[0]['name']
            place_pose.position.x = -0.1
            place_pose.position.y = 0.71
            place_pose.position.z = 0.605
            # place_pose.position = dropbox_param[0]['position']
        elif dropbox_group == dropbox_param[1]['group']:
            arm_name.data = dropbox_param[1]['name']
            place_pose.position.x = -0.1
            place_pose.position.y = -0.71
            place_pose.position.z = 0.605
            # place_pose.position = dropbox_param[1]['position']

        place_pose.orientation.x = 0
        place_pose.orientation.y = 0
        place_pose.orientation.z = 0
        place_pose.orientation.w = 1
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

        yaml_dict = make_yaml_dict(test_scene_num, arm_name,
                                   object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
        #
        """ rospy.wait_for_service('pick_place_routine')
        try:
            pick_place_routine = rospy.ServiceProxy(
                'pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name,
                                      arm_name, pick_pose, place_pose)

            print ("Response: ", resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s" % e  """
        # Wait for 'pick_place_routine' service to come up

    yaml_filename = '/home/alcatraz/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/test_output_yaml/output' + \
        str(world_num) + '.yaml'
    send_to_yaml(yaml_filename, dict_list)

    # output yaml file
    """ if outputed == False:
        yaml_filename = '/home/alcatraz/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/output_yaml/output' + \
            str(world_num) + '.yaml'
        send_to_yaml(yaml_filename, dict_list)
        print('file output successful')
        outputed = True """
    # TODO: Output your request parameters into output yaml file


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('perception')
    # TODO: Create Subscribers
    rospy.Subscriber('/pr2/world/points', PointCloud2, pcl_callback)
    # TODO: Create Publishers
    cloud_puber = rospy.Publisher('/cloud', PointCloud2, queue_size=1)
    obvoidence_pub = rospy.Publisher(
        '/pr2/3d_map/points', PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher(
        '/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher(
        '/detected_objects', DetectedObjectsArray, queue_size=1)
    pr2_world_joint_pub = rospy.Publisher(
        '/pr2/world_joint_controller/command', Float64, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(
        open('/home/alcatraz/catkin_ws/src/sensor_stick/sav/trainned.sav', 'rb'))

    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
