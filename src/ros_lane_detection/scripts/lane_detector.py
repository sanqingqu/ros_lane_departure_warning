#!/usr/bin/python3

import os 
import cv2 
import sys
import rospy
import torch 
import argparse
import numpy as np 
from PIL import Image
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

import torchvision.transforms as transforms
from sklearn import linear_model

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

class LaneDetector(object):
    
    def __init__(self, args) -> None:
        
        super(LaneDetector, self).__init__()
        self.verbose = args.verbose
        self.filter = args.filter
        self.model_dir = args.model_src_dir
        self.model_dataset = args.model_dataset
        self.model_path = args.model_path
        
        self.model, self.cls_num_per_lane,\
        self.griding_num, self.row_anchor = torch.hub.load(self.model_dir, self.model_dataset,\
                                    model_path=self.model_path, source="local")
    
        # gridding num are the pre-defined coloum numbers to locate lanes. 
        rospy.set_param("gridding_num", self.griding_num)#
        """
        # row anchors are a series of pre-defined coordinates in image height to detect lanes
        # the row anchors are defined according to the evaluation protocol of CULane and Tusimple
        # since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
        # you can modify these row anchors according to your training image resolution
        """
        rospy.set_param("row_anchor", self.row_anchor)
        # num_cls_per_lane is the number of row anchors
        rospy.set_param("cls_num_per_lane", self.cls_num_per_lane)
        # pre-defined lane_img_height to LaneDetection Model.
        rospy.set_param("lane_img_h", 288)
        # pre-defined lane_img width to LaneDetection Model
        rospy.set_param("lane_img_w", 800)

        self.model = self.model.cuda()
        
        # Init the ros img subscribe topic.
        self.img_sub_topic = args.front_img_topic
        self.img_flag = False # If do not receive any new img, this flag is set to False
        self.lane_pub_topic = args.lane_pub_topic
        self.lane_pre_dict = None
        
        self.cv_bridge = CvBridge()
        self.img_subscriber = rospy.Subscriber(self.img_sub_topic, CompressedImage,
                                                self.img_receive_callback, queue_size=3)

        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.lane_detect_publisher = rospy.Publisher(self.lane_pub_topic, Float32MultiArray, queue_size=5)
        self.lane_detect_msg = Float32MultiArray()
        self.lane_detect_msg.layout.dim = []
        self.lane_detect_msg.layout.dim.append(MultiArrayDimension(label="lane_col", size=201, stride=(201*18*4)*2))
        self.lane_detect_msg.layout.dim.append(MultiArrayDimension(label="lane_row", size=18, stride=(18*4)*2))
        self.lane_detect_msg.layout.dim.append(MultiArrayDimension(label="lane_type", size=4, stride=4*2))
        
    
    def img_receive_callback(self, img_msg):
        
        self.img_crop_height = int(rospy.get_param("img_crop_height"))
        self.img_rec_header = img_msg.header
        self.img_rec = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        self.img_rec = self.img_rec[self.img_crop_height:, :, :]
        
        self.img_w, self.img_h = self.img_rec.shape[1], self.img_rec.shape[0]
        self.img_flag = True
        
    def lane_detect_func(self):
        
        t1 = time_synchronized()
        img_rec_t = cv2.cvtColor(self.img_rec, cv2.COLOR_BGR2RGB)
        img_rec_t = Image.fromarray(img_rec_t)
        img_rec_t = self.img_transforms(img_rec_t).cuda().unsqueeze(0)
        model_out = self.model(img_rec_t).squeeze(0) #[w+1, r, c] # c=4, which means four type lanes.
        model_out_np = model_out.cpu().numpy()
        self.lane_detect_msg.data = np.frombuffer(model_out_np.tobytes(),'float32')
        self.lane_detect_publisher.publish(self.lane_detect_msg)
        t2 = time_synchronized()
        print("Lane Detection Inference time:{:.3f} ms".format((t2 - t1)*1000))
        
    
    # def kalman_filter(self, x_bef, p_bef, x_mes, A, Q, B, U, H, R):
        
    #     # kalman_predict(x_bef, p_bef, A, Q, B, U):
    #     # x_pre = np.dot(A, x_bef) + np.dot(B, U)
    #     x_pre = A@x_bef + B@U
    #     # p_pre = np.dot(A, np.dot(p_bef, A.T)) + Q
    #     p_pre = A@p_bef@A.T + Q

    #     # kalman_update(x_pre, p_pre, x_mes, H, R):
    #     K = p_pre@H.T/(H@p_pre@H.T + R)
    #     x_cur = x_pre + K@(x_mes - H@x_pre)
    #     p_cur = (np.eye(p_pre.shape) - K@H)@p_pre
            
    #     return x_cur, p_cur
         
    def run(self):
        publish_rate = rospy.Rate(20)
        print("Loading finished!")
        while not rospy.is_shutdown():
            
            if self.img_flag:
                self.img_flag = False
                with torch.no_grad():
                    self.lane_detect_func()
            else:
                print("img_not_ok!")

            publish_rate.sleep()
        pass
    
def main(args):
    
    rospy.init_node("lane_detector", anonymous=True)
    # pre-defined hyper-paramter to crop receive img. 
    rospy.set_param("img_crop_height", 200)
    rospy.set_param("img_lane_center", 410)
    rospy.set_param("lane_depart_thresh", 20)

    lane_model = LaneDetector(args)
    print("Loading.....")
    lane_model.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS image extraction module.")
        

if __name__ == "__main__":
    
    opts = argparse.ArgumentParser("This script is used to realize lane detection.")
    script_dir = sys.path[0]
    base_dir = os.path.dirname(script_dir)
    opts.add_argument("--model_src_dir", default=os.path.join(base_dir, "UFLD"))
    opts.add_argument("--model_path", default=os.path.join(base_dir, "ufld_weight/culane_18.pth"))
    opts.add_argument("--model_dataset", default="CULane")
    
    opts.add_argument("--front_img_topic", default="/camera/image_raw/compressed")
    opts.add_argument("--lane_pub_topic", default="/front_lane_detect")
    opts.add_argument("--verbose", action="store_true")
    opts.add_argument("--img_crop_height", default=200)
    opts.add_argument("--filter", action="store_true",
                      help="set this flag to only focus on person, car, bicycle related objects,\
                            other objects will be ignored.")
    args = opts.parse_args()
    print(args)
    main(args)