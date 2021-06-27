#!/usr/bin/python3

import rospy 
import numpy as np 

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension

def sub_callback(sub_msg):

    rec_data = sub_msg.data
    # data_dim = sub_msg.layout
    rec_data = np.array(rec_data).reshape(2, 3, 2)
    print("sub_data:\n", rec_data)
    
rospy.init_node("test_node", anonymous=True)
publish_rate = rospy.Rate(20)
pub = rospy.Publisher("test_topic", Float32MultiArray, queue_size=10)
sub = rospy.Subscriber("test_topic", Float32MultiArray, callback=sub_callback, queue_size=5)

# scoresize = dims.prod()/float(scores.nbytes)

test_msg = Float32MultiArray()
test_msg.layout.dim = []
# test_msg.layout.dim.append(MultiArrayDimension(label="height", size=480, stride=3*640*480))
# test_msg.layout.dim.append(MultiArrayDimension(label="width", size=640, stride=3*640))
# test_msg.layout.dim.append(MultiArrayDimension(label="channel", size=3, stride=3))
test_msg.layout.dim.append(MultiArrayDimension(label="height", size=2, stride=2*3*2*2))
test_msg.layout.dim.append(MultiArrayDimension(label="width", size=3, stride=3*2*2))
test_msg.layout.dim.append(MultiArrayDimension(label="channel", size=2, stride=2*2))

while not rospy.is_shutdown():
    test_data = np.random.rand(2, 3, 2)
    test_data = np.float32(test_data)
    print("org_data:\n", test_data)
    test_msg.data = np.frombuffer(test_data.tobytes(),'float32')
    pub.publish(test_msg)
    publish_rate.sleep()

try:
    rospy.spin()
except KeyboardInterrupt:
    print("Shutting down ROS image extraction module.")