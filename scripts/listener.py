#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import torch
# print(torch.__version__)
import time

batch_size = 1
num_frames = 60
num_joints = 32

processed_data = torch.zeros(batch_size, num_frames, num_joints*3)

frame_count = 0
start_timestamp = time.time()

def body_tracking_callback(msg):
    
    # read the data from the MarkerArray before preprocessing
    # for marker in msg.markers:
    #     print("Marker ID:", marker.id)
    #     print("Marker Position:", marker.pose.position)
    #     print("Marker Orientation:", marker.pose.orientation)
    #     print("---------------------")
    
    global frame_count, start_timestamp

    marker_array = msg.markers

    # for marker in marker_array:
    #     print(marker.header.frame_id, marker.type, marker.color.a)

    coordinates = [[marker.pose.position.x, marker.pose.position.y, marker.pose.position.z]
                   for marker in marker_array]
    marker_tensor = torch.tensor(coordinates, dtype = torch.float32)

    # print(marker_tensor.shape)
    # print(marker_tensor)

    processed_data[:,frame_count] = marker_tensor.view(-1)
    
    frame_count +=1

    if frame_count == num_frames:

        input_data = processed_data.view(batch_size, num_frames, num_joints*3)
        # print(input_data.shape)
        print(input_data.reshape(-1,32,3)[0,:]*1000)

        elapsed_time = time.time() - start_timestamp
        print("Elapsed time:", elapsed_time)

        frame_count = 0
        processed_data.zero_()
        start_timestamp = time.time()


def listener():
    rospy.init_node('subscriber_node', anonymous=True)
    rospy.Subscriber('body_tracking_data', MarkerArray, body_tracking_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()