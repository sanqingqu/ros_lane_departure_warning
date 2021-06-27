#!/usr/bin/python3

import os 
import cv2 
import argparse
from tqdm import tqdm 

if __name__ == "__main__":
    #----------------------------------------------------------------
    #----------------------------------------------------------------
    
    # opts = argparse.ArgumentParser("script used to generate video based img frames")
    # opts.add_argument("--src_img_dir", default="/home/ztp/Documents/test_highway_img")
    # opts.add_argument("--des_vid_f", default="/home/ztp/Documents/test_vid.avi")
    # opts.add_argument("--fps", default=30.0)
    
    # args = opts.parse_args()

    # src_img_list = sorted(os.listdir(args.src_img_dir))
    
    # img_0 = cv2.imread(os.path.join(args.src_img_dir, src_img_list[0]))
    # img_w, img_h = img_0.shape[0], img_0.shape[1]
    
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # # vout = cv2.VideoWriter(args.des_vid_f, fourcc, 30, (img_w, img_h))
    # # vout = cv2.VideoWriter("./test.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
    # out = cv2.VideoWriter('output.avi',fourcc, 30.0, (img_w, img_h))    
    # # img_list = []
    # for img_item in tqdm(src_img_list):
    #     img = cv2.imread(os.path.join(args.src_img_dir, img_item))
    #     out.write(img)
    #     cv2.imshow("img_show", img)
    #     cv2.waitKey(33)

    # out.release()
    # cv2.destroyAllWindows()
    
    # cap = cv2.VideoCapture(0)

    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     if ret==True:
    #         frame = cv2.flip(frame,0)

    #         # write the flipped frame
    #         out.write(frame)

    #         cv2.imshow('frame',frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    # # Release everything if job is finished
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    
    import glob
    img_array = []
    height, width = 0, 0
    filename_list = glob.glob('/home/ztp/Projects/qingdao/src/driver_action_detection/Driver-Anomaly-Detection-master/dataset/val06/rec1/front_depth/*.png')
    filename_list = sorted(filename_list, key=lambda name: int(name[122:-4]))
    for filename in filename_list:
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        
        img_array.append(img)
    
    size = (width,height)
    
    out = cv2.VideoWriter('project_1.avi',cv2.VideoWriter_fourcc(*'MJPG'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
