import yolo_signboard_module as ysm
import os
import sys

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
DARKNET_FOLDER = THIS_FOLDER+'/darknet'
DATASET_FOLDER = THIS_FOLDER+'/dataset'
TRAIN_FOLDER = DATASET_FOLDER+'/train'
VALID_FOLDER = DATASET_FOLDER+'/val'

def train():
    ysm.create_folder(DATASET_FOLDER)
    try:
        ysm.copy_eng_labels_to_darknet(THIS_FOLDER+'/label/eng.txt')#label폴더 안에 eng 건물 리스트를 카피
        ysm.write_obj_names(DATASET_FOLDER+'/_darknet.labels')
        ysm.write_obj_file(DATASET_FOLDER+'/_darknet.labels')#다크넷 파일안에 빌딩개수 읽어서 obj파일 만듬
        ysm.split_train_valid()#트레인 밸리드 셋 분할
        ysm.write_train_path()#트레인 경로 작성
        ysm.write_valid_path()#밸리드 경로 작성
        ysm.write_config_file()#config
    except:
        print('[ERROR]learning_model.py')
        exit(0)
    
    os.system('./darknet.exe detector train {} {} {} -map'.format(DARKNET_FOLDER+'/data/obj.data',DARKNET_FOLDER+'/cfg/custom-yolov4-signboard-detector.cfg',DARKNET_FOLDER+'/yolov4-tiny.conv.29'))

if __name__ == "__main__":
	train()