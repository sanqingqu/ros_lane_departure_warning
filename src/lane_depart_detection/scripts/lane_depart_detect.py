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

class LaneDepartDetect(object):

    def __init__(self, args) -> None:
        super(LaneDepartDetect).__init__()
        self.front_img_topic = args.front_img_topic
        self.front_lane_topic = args.front_lane_topic 
        
        self.row_anchor = rospy.get_param("row_anchor")
        self.cls_num_per_lane = rospy.get_param("cls_num_per_lane")
        self.lane_img_h = rospy.get_param("lane_img_h")
        self.lane_img_w = rospy.get_param("lane_img_h")
        self.lane_gridding_num = rospy.get_param("gridding_num")
        self.lane_depart_thresh = rospy.get_param("lane_depart_thresh")
        self.img_lane_center = rospy.get_param("img_lane_center") 
        
        self.front_sub_img = None
        self.front_lane_data = None
        
        self.cv_bridge = CvBridge()
        self.front_img_subscriber = rospy.Subscriber(self.front_img_topic, CompressedImage,\
                                                     self.front_img_callback, queue_size=5)
        
        self.front_lane_subscriber = rospy.Subscriber(self.front_lane_topic, Float32MultiArray,\
                                                     self.front_lane_callback, queue_size=5)
        
        self.img_flag = False
        self.lane_flag = False
        
    def front_img_callback(self, img_msg):
        img_crop_height = int(rospy.get_param("img_crop_height"))
        # self.img_rec_header = img_msg.header
        img_rec = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        self.img_rec = img_rec[img_crop_height:, :, :]
        
        self.img_w, self.img_h = self.img_rec.shape[1], self.img_rec.shape[0]
        self.img_flag = True
    
    def front_lane_callback(self, lane_msg):
        lane_rec_data = lane_msg.data 
        lane_rec_data = np.array(lane_rec_data).reshape(-1, self.cls_num_per_lane, 4)
        self.lane_rec_data = torch.from_numpy(lane_rec_data)
        self.lane_flag = True
        
    def lane_fit(self):
        #------------------------------------#
        #------------------------------------#
        lane_idx = torch.arange(self.lane_gridding_num) + 1
        lane_idx = lane_idx.view(-1, 1, 1)
        lane_prob = torch.softmax(self.lane_rec_data[:-1, :, :], dim=0)
        # lane_loc = torch.sum(lane_idx * lane_prob, dim=0) # [R, C]
        lane_loc = torch.argmax(lane_prob, dim=0)
        lane_loc = lane_loc / self.lane_gridding_num
        max_loc = torch.argmax(self.lane_rec_data, dim=0)
        lane_loc[max_loc == self.lane_gridding_num] = 0
        #------------------------------------#
        #------------------------------------#
        lane_loc = lane_loc.cpu().numpy()
        lane_prob = lane_prob.cpu().numpy()
        #------------------------------------#
        #------------------------------------#
        lane_col = lane_loc * self.img_w
        lane_row = self.img_h * np.array(self.row_anchor) / 288
        #------------------------------------#
        ransac_fit = linear_model.RANSACRegressor()
        
        lane_l_col = lane_col[:, 1]
        lane_l_row = lane_row[lane_l_col!=0].reshape(-1, 1)
        lane_l_col = lane_l_col[lane_l_col!=0].reshape(-1, 1)
        if len(lane_l_row) < 2:
            self.drive_region_cur = None
            return 
        ransac_fit.fit(lane_l_row, lane_l_col)
        lane_l_A = ransac_fit.estimator_.coef_[0]
        lane_l_B = ransac_fit.estimator_.intercept_
        
        lane_r_col = lane_col[:, 2]
        lane_r_row = lane_row[lane_r_col!=0].reshape(-1, 1)
        lane_r_col = lane_r_col[lane_r_col!=0].reshape(-1, 1)
        if len(lane_r_row) < 2:
            self.drive_region_cur = None
            return 
        ransac_fit.fit(lane_r_row, lane_r_col)
        
        lane_r_A = ransac_fit.estimator_.coef_[0]
        lane_r_B = ransac_fit.estimator_.intercept_
        
        row_up = int(lane_row[-1] - 125)
        row_down = int(lane_row[-1])
        left_up = int(row_up * lane_l_A + lane_l_B)
        left_down = int(row_down * lane_l_A + lane_l_B)
        right_up = int(row_up * lane_r_A + lane_r_B)
        right_down = int(row_down * lane_r_A + lane_r_B)

        self.drive_region_cur = np.array([[left_up], [right_up], [right_down], [left_down]])
        self.row_loc = np.array([[row_up], [row_up], [row_down], [row_down]])
        
        if right_up - left_up < 10:
            self.drive_region_cur = None
    
    def lane_visualize(self):
        
        try:
            self.lane_fit()
        except:
            pass
        
        if self.drive_region_cur is not None:
            """
            Lane Departure Detection！！！！
            """
            lane_fit_img = self.img_rec.copy()
            if abs(self.img_lane_center - (self.drive_region_cur[0][0] + self.drive_region_cur[1][0]) / 2) > self.lane_depart_thresh:
                # cv2.putText(self.img_rec, "Warning!", (200,450), cv2.FONT_HERSHEY_COMPLEX,5,(0,0,255),2)
                # cv2.putText(self.img_rec, 'Warning', (50,150), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 12)
                cv2.fillConvexPoly(lane_fit_img, np.hstack((self.drive_region_cur, self.row_loc)), color=(0, 0, 255))
                cv2.addWeighted(self.img_rec, 0.6, lane_fit_img, 0.4, 0, self.img_rec)
                self.drive_region_pre = self.drive_region_cur
                self.miss_count = 0
                print("Warning !!!")
            else:
                cv2.fillConvexPoly(lane_fit_img, np.hstack((self.drive_region_cur, self.row_loc)), color=(0, 255, 0))
                cv2.addWeighted(self.img_rec, 0.6, lane_fit_img, 0.4, 0, self.img_rec)
                self.drive_region_pre = self.drive_region_cur
                self.miss_count = 0
                print("SAFE !!!")
                
        elif self.drive_region_pre is not None and self.miss_count < 5:
            lane_fit_img = self.img_rec.copy()
            cv2.fillConvexPoly(lane_fit_img, np.hstack((self.drive_region_pre, self.row_loc)), color=(0, 0, 255))
            cv2.addWeighted(self.img_rec, 0.6, lane_fit_img, 0.4, 0, self.img_rec)
            self.miss_count += 1
            print("Warning !!!")
            # cv2.putText(self.img_rec, "Warning!", (200,450), cv2.FONT_HERSHEY_COMPLEX,5,(0,0,255),2)
            
        else:
            self.drive_region_pre = None
            print("miss!!!!")
        
        cv2.imshow("lane_detection", self.img_rec)
        cv2.waitKey(20)
        
    
    def run(self):
        sleep_rate = rospy.Rate(30)
        print("Modeling Preparing finished!")
        while not rospy.is_shutdown():
            
            if self.img_flag and self.lane_flag:
                self.img_flag = False
                self.lane_flag = False
                self.lane_visualize()

            sleep_rate.sleep()

def main(args):
    
    rospy.init_node("lane_departure_detection", anonymous=True)
    lane_depart_detect = LaneDepartDetect(args)
    print("Loading.....")
    lane_depart_detect.run()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS image extraction module.")
         
if __name__ == "__main__":
    
    opts = argparse.ArgumentParser("This script is used to realize lane departure detection.")

    opts.add_argument("--front_img_topic", default="/camera/image_raw/compressed")
    opts.add_argument("--front_lane_topic", default="/front_lane_detect")
    
    args = opts.parse_args()
    print(args)
    main(args)