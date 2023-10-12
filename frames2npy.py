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

frame_indexes = [1,5,10,15,20]

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
            label_color_img = cv2.imread(os.path.join(train_set_root_dirt,js['raw_file']),cv2.IMREAD_COLOR)
            label_color_img = cv2.cvtColor(label_color_img, cv2.COLOR_RGB2BGR)

            train_color_imgs = list()
            # "clips/0601/1494453593562528307/20.jpg"
            for frame_idx in frame_indexes:
                file_name_split_split = js['raw_file'].split('/')
                file_name = f"{file_name_split_split[0]}/{file_name_split_split[1]}/{file_name_split_split[2]}/{frame_idx}.jpg"
                train_color_img = cv2.imread(os.path.join(train_set_root_dirt,file_name),cv2.IMREAD_COLOR)

                train_color_imgs.append(train_color_img)
            
            train_color_imgs = np.array(train_color_imgs)

            for lane in js['lanes']:
                lane = np.array(lane)
                lane_y = js['h_samples'][np.where(lane!=-2)[0]]
                lane_x = lane[np.where(lane!=-2)[0]]

                lane_coordinates = list(zip(lane_x,lane_y))
                gray_scale_image =  cv2.polylines(gray_scale_image, [np.int32(lane_coordinates)], isClosed=False, color=(1), thickness=2)
            
            substr = js['raw_file'].split('/')
            
            with open(f'../data/train/data/{substr[-4]}_{substr[-3]}_{substr[-2]}_road_clips.npy', 'wb') as f:
                np.save(f, train_color_imgs)
            
            with open(f'../data/train/label/{substr[-4]}_{substr[-3]}_{substr[-2]}_lane.npy', 'wb') as f:
                np.save(f, gray_scale_image)
          
            print('\r', f"image processed : {cnt}/3626", end='')
            cnt += 1
# %%
