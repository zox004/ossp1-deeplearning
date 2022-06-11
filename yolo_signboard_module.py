import os
import sys
import shutil
from numpy import full
import splitfolders
from glob import glob
from os.path import isdir

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DARKNET_FOLDER = THIS_FOLDER+'/darknet'
IMAGE_FOLDER = THIS_FOLDER+'/img'
DATASET_FOLDER = THIS_FOLDER+'/dataset'
TRAIN_FOLDER = DATASET_FOLDER+'/train'
VALID_FOLDER = DATASET_FOLDER+'/val'

def create_folder(directory):#폴더생성
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print("the file is already existing")
    except OSError:
        print('ERROR creating driectory: ' + directory)

def file_len(fname):#다크넷라벨안에 있는 건물 개수
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def copy_eng_labels_to_darknet(eng_txt_path):
    buildings=''
    with open(eng_txt_path,'r') as f:
        buildings=f.read()
    with open(DATASET_FOLDER+'/_darknet.labels','w') as f:
        f.write(buildings)
        print('_darknet.labels file:\n',buildings)

def write_obj_names(darknet_labels_path):
    try:
        shutil.copyfile(darknet_labels_path,DARKNET_FOLDER+'/data/obj.names')
    except OSError:
        print('ERROR copy _darknet.labels file to data/obj.names')

def write_obj_file(path):
    classes=file_len(path)
    with open(DARKNET_FOLDER+'/data/obj.data','w') as f:
        f.write('classes = {}\n'.format(classes))#추가된 이미지 개수
        f.write('train = {}/data/train.txt\n'.format(DARKNET_FOLDER))
        f.write('valid = {}/data/valid.txt\n'.format(DARKNET_FOLDER))
        f.write('names = {}/data/obj.names\n'.format(DARKNET_FOLDER))
        f.write('backup = {}/backup/'.format(DARKNET_FOLDER))

def split_train_valid():
    create_folder(DATASET_FOLDER)
    try:
        splitfolders.ratio(IMAGE_FOLDER,output=DATASET_FOLDER,ratio=(0.8,0.2),group_prefix=2)
        # splitfolders.ratio('데이터 경로', output='output 폴더 경로',ratio=(0.8,0.2),group_prefix=2)
        # 데이터 경로에 class별로 Directory가 존재해야 함
        print('success split train and valid dataset: '+DATASET_FOLDER)
    except:
        print("ERROR split train and valid dataset :", DATASET_FOLDER)

"""dataset/train,valid/0,aug_0"""
def write_train_path():
    train_list = os.listdir(TRAIN_FOLDER)
    full_path=[]
    a = "ABCDE"
    with open(DARKNET_FOLDER+'/data/train.txt','w') as f:
        f.write('')
    for folder in train_list:
        onedir = TRAIN_FOLDER+'/'+folder
        onedir_jpg_elem = glob(onedir+'/*.jpg')
        full_path=[os.path.join(onedir,f) for f in onedir_jpg_elem]
        with open(DARKNET_FOLDER+'/data/train.txt','a') as f:
            for jpg_path in full_path:
                f.write(jpg_path+'\n')

def write_valid_path():
    valid_list = os.listdir(VALID_FOLDER)
    full_path=[]
    with open(DARKNET_FOLDER+'/data/valid.txt','w') as f:
        f.write('')
    for folder in valid_list:
        onedir = VALID_FOLDER+'/'+folder
        onedir_jpg_elem = glob(onedir+'/*.jpg')
        full_path=[os.path.join(onedir,f) for f in onedir_jpg_elem]
        with open(DARKNET_FOLDER+'/data/valid.txt','a') as f:
            for jpg_path in full_path:
                f.write(jpg_path+'\n')

def write_config_file():
    num_classes=file_len(DATASET_FOLDER+'/_darknet.labels')
    max_batches = num_classes*2000
    steps1 = .8 * max_batches
    steps2 = .9 * max_batches
    steps_str = str(steps1)+','+str(steps2)
    num_filters = (num_classes + 5) * 3
    if os.path.exists(DARKNET_FOLDER+'/cfg/custom-yolov4-signboard-detector.cfg'):
        os.remove(DARKNET_FOLDER+'/cfg/custom-yolov4-signboard-detector.cfg')
    cfg_f=''
    with open(DARKNET_FOLDER+'/cfg/initial_cfg.txt','r') as f:
        cfg_f=f.read()
        cfg_f=cfg_f.replace('{num_classes}',str(num_classes))
        cfg_f=cfg_f.replace('{max_batches}',str(max_batches))
        cfg_f=cfg_f.replace('{steps1}',str(steps1))
        cfg_f=cfg_f.replace('{steps2}',str(steps2))
        cfg_f=cfg_f.replace('{steps_str}',str(steps_str))
        cfg_f=cfg_f.replace('{num_filters}',str(num_filters))
        print(cfg_f)

    with open(DARKNET_FOLDER+'/cfg/custom-yolov4-signboard-detector.cfg','w') as f:
        f.write(cfg_f)
        print(cfg_f)
        print('SUCCESS write custom-yolov4-signboard-detector.cfg')