# AI Hub image dataset preprocessing
import json
import random
import os
from tqdm import tqdm
import pandas as pd

data_type = 'signboard'  # signboard, htr, ocr

## Check Json File
label_dir = os.listdir(f'./test/labels/') ## pghj_kocrnn-main/

label_files = list()
for label in label_dir:
  file = json.load(open(f'./test/labels/{label}', encoding="UTF-8")) ## pghj_kocrnn-main/
  label_files.append(file)

## Separate dataset - train, validation, test
image_files = os.listdir(f'./test/images/')  ## pghj_kocrnn-main/
total = len(image_files)
print(total)
n_test = int(len(image_files))
print(n_test)


test_files = image_files[:total]
print(test_files)
## Separate image id - train, validation, test
test_img_ids = {}

for label in label_files:
  image = label['images'][0]
  if image['file_name'] in test_files:
    test_img_ids[image['file_name']] = image['file_name'][-10:-4]
#print(train_img_ids.keys())
#print(validation_img_ids)
print(test_img_ids)

## Annotations - train, validation, test 
test_annotations = {f:[] for f in test_img_ids.keys()}
test_ids_img = {test_img_ids[id_]:id_ for id_ in test_img_ids}

for idx, label in enumerate(label_files):
  image_id = (label['images'][0])['file_name'][-10:-4]
  annotation = label['annotations'][0]
  
  test_annotations[test_ids_img[image_id]].append(annotation)

## Write json files
with open(f'{data_type}_test_annotation.json', 'w') as file:
  json.dump(test_annotations, file)

## Make gt_xxx.txt files
data_root_path = f'./test/images/' ## pghj_kocrnn-main/
save_root_path = f'./test/' ## pghj_kocrnn-main/
#data_root_path = f'./data/{data_type}/images/' ## pghj_kocrnn-main/
#save_root_path = f'./deep-text-recognition-benchmark/{data_type}_data/' ## pghj_kocrnn-main/


total_annotations = json.load(open(f'./{data_type}_test_annotation.json'))
gt_file = open(f'{save_root_path}labels.txt', 'w')
for file_name in tqdm(total_annotations):
  annotations = total_annotations[file_name]
  for idx, annotation in enumerate(annotations):
    text = annotation['text']
    gt_file.write(f'{file_name}\t{text}\n')