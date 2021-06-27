#!/usr/bin/python3
import sys
from ros import rosbag
import roslib, rospy
roslib.load_manifest('sensor_msgs')
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

TOPIC = '/camera/depth/image_rect_raw'

def CreateVideoBag(videopath="./project_1.avi", bagname="project_1.bag"):
    '''Creates a bag file with a video file'''
    bag = rosbag.Bag(bagname, 'w')
    cap = cv2.VideoCapture(videopath)
    cb = CvBridge()
    prop_fps = cap.get(cv2.CAP_PROP_FPS)
    # if prop_fps != prop_fps or prop_fps <= 1e-2:
    #     print("Warning: can't get FPS. Assuming 24.")
    #     prop_fps = 24
    ret = True
    frame_id = 0
    while(ret):
        ret, frame = cap.read()
        if not ret:
            break
        stamp = rospy.rostime.Time.from_sec(float(frame_id) / prop_fps)
        frame_id += 1
        image = cb.cv2_to_compressed_imgmsg(frame)
        image.header.stamp = stamp
        image.header.frame_id = "camera"
        bag.write(TOPIC, image, stamp)
    cap.release()
    bag.close()


if __name__ == "__main__":
    if len(sys.argv ) == 3:
        CreateVideoBag()
    else:
        print( "Usage: video2bag videofilename bagfilename")
    CreateVideoBag()


