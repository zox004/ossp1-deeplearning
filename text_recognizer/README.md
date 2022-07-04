# kocrnn: CRNN 기반의 한글 텍스트 인식 모델 학습
- clovaai의 deep-text-recognition-benchmark, https://github.com/apphia39/pghj_kocrnn을 참고하여 한글 인식 pretrained model을 만든다.
- 데이터셋: [AI Hub 야외 실제 촬영 한글 이미지](https://aihub.or.kr/aidata/33985), [AI Hub 한국어 글자체 이미지](https://aihub.or.kr/aidata/133)
- 자세한 과정은 [velog](https://velog.io/@apphia39/python-CRNN-%ED%95%9C%EA%B8%80-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0)를 참고하였다.
<br>

## 폴더 / 파일 설명
- 이미지파일 및 라벨링파일 등이 포함되는 폴더들은 용량이 너무 큰 관계로 레포지토리에 업데이트하지 못하였다.
- 가이드대로 따라하는 경우, 아래와 같은 폴더 구조가 만들어진다.
```
deep-text-recognition-benchmark
  |--signboard_data              
    |--train              # AIhub 간판 이미지 - train dataset 파일들을 포함
    |--test               # AIhub 간판 이미지 - test dataset 파일들을 포함
    |--validation         # AIhub 간판 이미지 - validation dataset 파일들을 포함
    |--get_images.py      # kor_dataset/aihub_data/signboard/images에서 이미지 파일들을 가져와 train, test, validation 폴더로 분리
    |--gt_train.py        # train dataset gt file
    |--gt_test.py         # test dataset gt file
    |--gt_validation.py   # validation dataset gt file
  |--demo.py              # pretrained model을 test               
  |--train.py             # model training
  |--test.py
  |--...
    
kor_dataset
  |--aihub_data
    |--signboard
      |--images                       # AIhub 간판 이미지 파일들을 포함
      |--labels 	 	# AIhub 간판 이미지들에 대한 라벨링 파일들을 포함
       
saved_models                           # Train 이후 모델이 저장되는 폴더
  |--TPS-ResNet-BiLSTM-CTC-Seed1234
    |--best_accuracy.pth               # 정확도 제일 높은 pretrained model
    |--...
    
pretrained_models                      # pretrained models를 옮겨둘 폴더
  |--kocrnn.pth
  |--...
    
test                                   # 별도로 테스트할 dataset들을 저장하는 폴더
  |--images
  |--labels.txt

```
<br>

## Dataset
AI Hub 야외 실제 촬영 한글 이미지에서 **간판 이미지**를 이용해 학습하였다.

### 1. Preprocessing
####  1-1) AI Hub  야외 실제 촬영 한글 이미지 데이터셋 가공(htr)
I. AI Hub에서 간판 데이터셋을 [다운로드](https://aihub.or.kr/aidata/33985/download) 받는다. <br>
II. 이미지 파일들은 kor_dataset/aihub_data/signboard/images/ 폴더에 저장하고, 라벨링 파일들은 kor_dataset/aihub_data/signboard/labels 폴더에 저장한다. <br>
III. aihub_dataset.py를 열어 아래와 같이 변수 값을 수정한다. <br>
```python3
data_type = 'signboard'
```
IV. AI Hub 데이터셋의 라벨링 파일 구조는 아래와 같다.<br>
```
# handwriting_data_info1.json
{
  'images': [
    {
      'id': 1,
      'width': 1601,
      'height': 1200,
      'file_name': '20201017_111654.jpg',
      'data_captured': '2020-10-26 12:20:09'
    },
  ],
  'annoations': [
    {
      'id': 1,
      'image_id': 1,
      'text': '여기가 이미지 파일에 대한 라벨이 기입된 곳입니다.',
      'bbox': [
        131,
        460,
        1290,
        479
      ]
    }
  ],
  'cropLabels': [].
  'info' : {
	'name' : '충남-2-8',
	'description' : '',
	'weather' : '맑음',
	'sun' : '낮',
	'date_created' : '2020-10-26 13:35:49'
  }
}
```
V. 아래 파일을 실행시켜 데이터 전처리를 수행한다.<br>
```
python3 aihub_dataset.py
```
이후 deep-text-recognition-benchmark/signboard_data/ 폴더에 gt_test.txt, gt_train.txt, gt_validation.txt가 생성된다.<br>

### 2. lmdb data 만들기
I. Preprocessing이 완료되면, deep-text-recognition-benchmark/signboard_data 폴더에는 gt_{xxx}.txt 파일들이 존재한다.<br>
II. get_images.py 파일을 deep-text-recognition-benchmark/htr_data 폴더로 복사한 뒤 test, train, validation 폴더를 생성한다.<br>
```
cp ./get_images.py ./deep-text-recognition-benchmark/signboard_data/
cd deep-text-recognition-benchmark/signboard_data/
mkdir train test validation
```
III. kor_dataset에 있는 이미지 파일들을 현재 폴더로 가져와서 test, train, validation 폴더로 나눠준다.<br>
```
python3 get_images.py
```
IV. 학습을 위해 lmdb data를 생성한다.<br>
```
cd ../../
python3 ./deep-text-recognition-benchmark/create_lmdb_dataset.py \
    --inputPath ./deep-text-recognition-benchmark/signboard_data/ \
    --gtFile ./deep-text-recognition-benchmark/signboard_data/gt_train.txt \
    --outputPath ./deep-text-recognition-benchmark/signboard_data_lmdb/train

python3 ./deep-text-recognition-benchmark/create_lmdb_dataset.py \
    --inputPath ./deep-text-recognition-benchmark/signboard_data/ \
    --gtFile ./deep-text-recognition-benchmark/signboard_data/gt_validation.txt \
    --outputPath ./deep-text-recognition-benchmark/signboard_data_lmdb/validation
```
<br>

## Train
### 1. 처음 학습하는 경우
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./deep-text-recognition-benchmark/train.py \
    --train_data ./deep-text-recognition-benchmark/signboard_data_lmdb/train \
    --valid_data ./deep-text-recognition-benchmark/signboard_data_lmdb/validation \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
```

### 2. 기존 모델에 추가로 학습하는 경우
학습 완료된 모델을 원하는 폴더 아래에 옮긴다.
```
cd saved_models
mv TPS-ResNet-BiLSTM-CTC-Seed1234 kocrnn
cd kocrnn
cp best_accuracy.pth ../../pretrained_models/kocrnn.pth
```
이후 해당 모델을 불러와서 그 위에 추가로 학습을 진행한다.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./deep-text-recognition-benchmark/train.py \
    --train_data ./deep-text-recognition-benchmark/signboard_data_lmdb/train \
    --valid_data ./deep-text-recognition-benchmark/signboard_data_lmdb/validation \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --saved_model ./pretrained_models/kocrnn.pth
```
<br>

> **※주의사항1**<br>
> train.py, test.py, demo.py의 --character 옵션에 학습/테스트하길 원하는 글자들을 반드시 포함시켜주어야 한다.<br>
>
> **※주의사항2**<br>
> 현재, 학습을 진행하게 되면 항상 saved_models/TPS-ResNet-BiLSTM-CTC-Seed1234 폴더 아래로 저장된다.<br>
> 따라서 새로운 학습을 진행할 때마다 모델이 overwrite되는 것을 방지하기 위해서는 <br>
> 1) --manualSeed 옵션에 매번 다른 값을 넣어주거나<br>
> 2) mv명령어를 이용해 학습이 한 번 끝날 때마다 TPS-ResNet-BiLSTM-CTC-Seed1234 폴더명을 다른 이름으로 바꿔주도록 한다.<br>

### 학습 과정
![image](https://user-images.githubusercontent.com/67676029/144251733-96410a39-9ca7-443c-80dd-f5c379cd6058.png)


<br>

## Test
### 1. gt_test.txt 파일을 이용해 test할 경우
I. deep-text-recognition-benchmark/signboard_data/test 폴더에 있는 이미지들을 테스트한다. 라벨링 파일은 gt_test.txt를 이용한다.<br>
```
# gt_test.txt
...
test/간판_가로형간판_000001.jpg 여기에
test/간판_가로형간판_000004.jpg 있는
test/간판_가로형간판_000007.jpg 라벨은
test/간판_가로형간판_000010.jpg 탭(\t)과 개행(\n)으로
test/간판_가로형간판_000026.jpg 구분합니다.
test/간판_가로형간판_000029.jpg 확장자(.jpg or .JPG)도
test/간판_가로형간판_000030.jpg 확인하세요.
...
```

II. demo.py를 아래와 같이 수정한다.<br>
```python
# 118줄, 119줄
if ".jpg" in line :
	info = line.split('.jpg\t\n')
             file_name = info[0] + '.jpg'
elif ".JPG" in line:
	info = line.split('.JPG\t\n')
	file_name = info[0] + '.JPG'
```

III. 테스트를 수행한다.<br>
```
CUDA_VISIBLE_DEVICE=0,1,2,3 python3 ./deep-text-recognition-benchmark/demo.py \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --image_folder ./deep-text-recognition-benchmark/signboard_data/test/ \
    --label_test ./deep-text-recognition-benchmark/signboard_data/gt_test.txt \
    --log_filename ./deep-text-recognition-benchmark/signboard_data/htr_test_log.txt \
    --saved_model ./pretrained_models/kocrnn.pth
```

### 2. 별도의 test 파일을 이용할 경우
I. test/images 폴더 아래 테스트할 이미지들을 넣고, test 폴더 아래 해당 테스트 이미지들에 대한 라벨링 파일(labels.txt)을 넣는다.<br>
```
# labels.txt
...
10.jpg 여기에
11.jpg 있는
12.jpg 라벨은
13.jpg 띄어쓰기로
14.jpg 구분합니다.
15.jpg 확장자도
16.jpg 확인하세요.
...
```

II. demo.py를 아래와 같이 수정한다.<br>
```python
# 118줄, 119줄
info = line.split('.jpg ')
file_name = info[0] + '.jpg'
```

III. 테스트를 수행한다.<br>
```
CUDA_VISIBLE_DEVICE=0,1,2,3 python3 ./deep-text-recognition-benchmark/demo.py \
    --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \
    --image_folder ./test/images/ \
    --label_test ./test/labels.txt \
    --log_filename ./test/test_log.txt \
    --saved_model ./pretrained_models/kocrnn.pth
```

### Test 결과 예시 (test_log.txt)
왼쪽부터 순서대로 | Image Name | Real Text | Predicted Text | Confidence Score | Character Error Rate | 이다.<br>
<img width="800" alt="20211201_230358" src="https://user-images.githubusercontent.com/67676029/144249038-016f7028-087a-4057-a282-b1f6d17d75b4.png">

