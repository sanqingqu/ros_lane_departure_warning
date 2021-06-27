#!/usr/bin/python3

import cv2
import argparse
from joblib import Parallel, delayed

def camera_id_detect():
    valida_id = []
    for i in range(99):
        camera_input = cv2.VideoCapture(i)
        if camera_input is None or not camera_input.isOpened():
            # print("Current camera can not open!")
            pass
        else:
            
            valida_id.append(i)

    print("Valid camera ID")
    print(valida_id)
    return valida_id

def camera_show(camera_id):

    camera_input = cv2.VideoCapture(camera_id)  
    if camera_input is None:
        print("Can not open camera id: {:d}".format(camera_id))
    else:
        print("camera_brightness: {}".format(camera_input.get(10)))
        camera_input.set(10, 0.8)
        print(camera_input.get(5))
        print(camera_input.get(3))
        print(camera_input.get(4))

    while True:
        ret, img_input = camera_input.read()
        cv2.imshow("camera_id {:d}".format(camera_id), img_input)
        input_key = cv2.waitKey(1) & 0xFF
        if input_key == ord('q'):
            cv2.destroyWindow("camera_id {:d}".format(camera_id))
            break


if __name__ == "__main__":
    args = argparse.ArgumentParser("This scirpt is used to detect camera id and show camera input stream")
    args.add_argument('-c',"--camera_id", default=None, type=int)
    opts = args.parse_args()
    if opts.camera_id is not None:
        camera_show(opts.camera_id)
    else: 
        valid_list=camera_id_detect()
        print(valid_list)   
        Parallel(n_jobs=4)(delayed(camera_show)(i) for i in valid_list)

        