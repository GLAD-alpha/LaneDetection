# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# %%
train_clip_dir = '../data/TUSimple/train_set/clips'
train_data_dir = '../data/train'
test_data_dir = '../data/valid'

try:
    os.mkdir(train_data_dir)
except FileExistsError:
    pass

try:
    os.mkdir(test_data_dir)
except FileExistsError:
    pass
# %%
import json

json_dir_list = list()
train_set_root_dirt = '../data/TUSimple/train_set'

for files in os.listdir(train_set_root_dirt):
    if 'json' in files:
        json_dir_list.append(os.path.join(train_set_root_dirt, files))

cnt = 1
# open each jsons and process label with raw data dir
for label_json in json_dir_list:
    
    with open(label_json) as f:
        data = f.read()
        label_list = data.split('\n')

        # each line is the coordinate information for each image. 
        for line in label_list:
            # last line is a empty string
            if line == '':
                continue
            js = json.loads(line)
            
            # 'h_samples' is the yticks of lane coordinates
            js['h_samples'] = np.array(js['h_samples'])

            # prepare y axis
            gray_scale_image = np.zeros((720,1280,1))
            color_img = cv2.imread(os.path.join(train_set_root_dirt,js['raw_file']),cv2.IMREAD_COLOR)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

            for lane in js['lanes']:
                lane = np.array(lane)
                lane_y = js['h_samples'][np.where(lane!=-2)[0]]
                lane_x = lane[np.where(lane!=-2)[0]]

                lane_coordinates = list(zip(lane_x,lane_y))
                gray_scale_image =  cv2.polylines(gray_scale_image, [np.int32(lane_coordinates)], isClosed=False, color=(1), thickness=2)
                # color_img = cv2.polylines(color_img, [np.int32(lane_coordinates)], isClosed=False, color=(255,255,255), thickness=2)
            
            substr = js['raw_file'].split('/')
            
            with open(f'../data/train/data/{substr[-4]}_{substr[-3]}_{substr[-2]}_road.npy', 'wb') as f:
                np.save(f, color_img)
            
            with open(f'../data/train/label/{substr[-4]}_{substr[-3]}_{substr[-2]}_lane.npy', 'wb') as f:
                np.save(f, gray_scale_image)
          
            print('\r', f"image processed : {cnt}/3626", end='')
            cnt += 1