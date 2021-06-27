#!/usr/bin/python3

import os 
import cv2 
import sys
import json 
import rospy
import torch 
import argparse
import numpy as np 
from PIL import Image
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

import torchvision.transforms as transforms
from sklearn import linear_model
# from sklearn.linear_model import LinearRegression

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
    
        self.model = self.model.cuda()
        
        # Init the ros img subscribe topic.
        self.img_sub_topic = args.img_sub_topic
        self.img_flag = False # If do not receive any new img, this flag is set to False
        self.lane_out_topic = args.lane_out_topic
        self.lane_pre_dict = None
        
        self.cv_bridge = CvBridge()
        self.img_subscriber = rospy.Subscriber(self.img_sub_topic, CompressedImage,
                                                self.img_receive_callback, queue_size=3)

        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.col_sample = np.linspace(0, 800 - 1, self.griding_num)
        self.col_sample_w = self.col_sample[1] - self.col_sample[0]
    
        self.drive_region_pre = None
        self.drive_region_cur = None
        
        self.miss_count = 0
    
        self.lane_center = 410
        self.lane_depart_thresh = 10
    
    def img_receive_callback(self, img_msg):
        self.img_rec_header = img_msg.header
        self.img_rec = self.cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        self.img_rec = self.img_rec[200:, :, :]
        self.img_w, self.img_h = self.img_rec.shape[1], self.img_rec.shape[0]
        self.img_flag = True
        
    def lane_detect_func(self):
        
        t1 = time_synchronized()
        img_rec_t = cv2.cvtColor(self.img_rec, cv2.COLOR_BGR2RGB)
        img_rec_t = Image.fromarray(img_rec_t)
        img_rec_t = self.img_transforms(img_rec_t).cuda().unsqueeze(0)
        model_out = self.model(img_rec_t).squeeze(0) #[w+1, r, c] # c=4, which means four type lanes.
        t2 = time_synchronized()

        try:
            self.lane_fit(model_out)
        except:
            pass
        if self.drive_region_cur is not None:
            lane_fit_img = self.img_rec.copy()
            cv2.fillConvexPoly(lane_fit_img, np.hstack((self.drive_region_cur, self.row_loc)), color=(0, 255, 0))
            cv2.addWeighted(self.img_rec, 0.6, lane_fit_img, 0.4, 0, self.img_rec)
            self.drive_region_pre = self.drive_region_cur
            self.miss_count = 0
            if abs(self.lane_center - (self.drive_region_cur[1] + self.drive_region_cur[0]) / 2) > self.lane_depart_thresh:
                cv2.putText(self.img_rec, "Warning!", (200,450), cv2.FONT_HERSHEY_COMPLEX,5,(0,0,255),2)
            
        elif self.drive_region_pre is not None and self.miss_count < 5:
            lane_fit_img = self.img_rec.copy()
            cv2.fillConvexPoly(lane_fit_img, np.hstack((self.drive_region_pre, self.row_loc)), color=(0, 255, 0))
            cv2.addWeighted(self.img_rec, 0.6, lane_fit_img, 0.4, 0, self.img_rec)
            self.miss_count += 1
            cv2.putText(self.img_rec, "Warning!", (200,450), cv2.FONT_HERSHEY_COMPLEX,5,(0,0,255),2)
            
        else:
            self.drive_region_pre = None
            print("miss!!!!")
            pass

        t3 = time_synchronized()
        
        print("Inference time:{:.3f} ms, Visulization time:{:.3f}".format((t2 - t1)*1000, (t3 - t2)*1000))
    
    def lane_visualize(self, model_out):
        
        lane_idx = torch.arange(self.griding_num) + 1
        lane_idx = lane_idx.view(-1, 1, 1).cuda()
        lane_prob = torch.softmax(model_out[:-1, :, :], dim=0)
        lane_loc = torch.sum(lane_idx * lane_prob, dim=0) # [R, C]
        # lane_loc = torch.argmax(lane_prob, dim=0)
        lane_loc = lane_loc / self.griding_num
        max_loc = torch.argmax(model_out, dim=0)
        lane_loc[max_loc == self.griding_num] = 0
        
        lane_loc = lane_loc.cpu().numpy()
        lane_prob = lane_prob.cpu().numpy()
        for i in range(1, 3):
            if np.sum(lane_prob[:, i]) > 3:
                for k in range(self.cls_num_per_lane):
                    if lane_loc[k, i] > 0:
                        ppp = (int(lane_loc[k, i]*self.img_w), int(self.img_h*self.row_anchor[k]/288)-1)
                        if i == 0:
                            cv2.circle(self.img_rec, ppp, 5, (255,0,0),-1)
                        if i == 1:
                            cv2.circle(self.img_rec, ppp, 5, (0,255,0),-1)
                        elif i == 2:
                            cv2.circle(self.img_rec, ppp, 5, (0,0,255),-1)
                        if i == 3:
                            cv2.circle(self.img_rec, ppp, 5, (255,255,255),-1)
    
    
    def kalman_filter(self, x_bef, p_bef, x_mes, A, Q, B, U, H, R):
        
        # kalman_predict(x_bef, p_bef, A, Q, B, U):
        # x_pre = np.dot(A, x_bef) + np.dot(B, U)
        x_pre = A@x_bef + B@U
        # p_pre = np.dot(A, np.dot(p_bef, A.T)) + Q
        p_pre = A@p_bef@A.T + Q

        # kalman_update(x_pre, p_pre, x_mes, H, R):
        K = p_pre@H.T/(H@p_pre@H.T + R)
        x_cur = x_pre + K@(x_mes - H@x_pre)
        p_cur = (np.eye(p_pre.shape) - K@H)@p_pre
            
        return x_cur, p_cur
        
    def lane_fit(self, model_out):
        
        #------------------------------------#
        #------------------------------------#
        lane_idx = torch.arange(self.griding_num) + 1
        lane_idx = lane_idx.view(-1, 1, 1).cuda()
        lane_prob = torch.softmax(model_out[:-1, :, :], dim=0)
        # lane_loc = torch.sum(lane_idx * lane_prob, dim=0) # [R, C]
        lane_loc = torch.argmax(lane_prob, dim=0)
        lane_loc = lane_loc / self.griding_num
        max_loc = torch.argmax(model_out, dim=0)
        lane_loc[max_loc == self.griding_num] = 0
        #------------------------------------#
        #------------------------------------#
        lane_loc = lane_loc.cpu().numpy()
        lane_prob = lane_prob.cpu().numpy()
        #------------------------------------#
        #------------------------------------#
        lane_col = lane_loc * self.img_w
        lane_row = self.img_h * np.array(self.row_anchor) / 288
        # lane_row = lane_row.reshape(-1, 1)
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
        # drive_region = np.array([[left_up, row_up], [right_up, row_up],
        #                 [right_down, row_down], [left_down, row_down]])
        self.drive_region_cur = np.array([[left_up], [right_up], [right_down], [left_down]])
        self.row_loc = np.array([[row_up], [row_up], [row_down], [row_down]])
        
        if right_up - left_up < 10:
            self.drive_region_cur = None
        
        # lane_fit_img = self.img_rec.copy()
        # cv2.fillConvexPoly(lane_fit_img, np.hstack((self.drive_region_cur, self.row_loc)), color=(0, 255, 0))
        # cv2.addWeighted(self.img_rec, 0.6, lane_fit_img, 0.4, 0, self.img_rec)
        # self.drive_region_pre = self.drive_region_cur
        
    def run(self):
        publish_rate = rospy.Rate(20)
        print("Loading finished!")
        while not rospy.is_shutdown():
            
            if self.img_flag:
                self.img_flag = False
                if self.verbose:
                    with torch.no_grad():
                        self.lane_detect_func()
                    cv2.imshow("img_pre", self.img_rec)
                    cv2.waitKey(20)
                else:
                    with torch.no_grad():
                        self.lane_detect_func()
            else:
                print("img_not_ok!")

            publish_rate.sleep()
        pass
    
def main(args):
    
    rospy.init_node("yolo_detector", anonymous=True)
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
    # opts.add_argument("--model_path", default=os.path.join(base_dir, "ufld_weight/tusimple_18.pth"))
    # opts.add_argument("--model_dataset", default="Tusimple")
    opts.add_argument("--img_sub_topic", default="/cam_0/usb_cam/image_raw/compressed")
    opts.add_argument("--lane_out_topic", default="/cam_0/lane_out")
    opts.add_argument("--verbose", action="store_true")
    opts.add_argument("--filter", action="store_true",
                      help="set this flag to only focus on person, car, bicycle related objects,\
                            other objects will be ignored.")
    args = opts.parse_args()
    print(args)
    main(args)