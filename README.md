# **Open Source Software Project1 - Deep Learning**

## **Project name : SINFONIZER(Signboard Information Recognizer)**

## **Members**

[강신규](https://github.com/zox004) | [이예빈](https://github.com/Yebeen-Lee) | [박세아](https://github.com/iamsseaa)

## **Dataset**

- [AI Hub](https://aihub.or.kr/)

## **Goal**

스마트폰의 카메라 어플리케이션을 이용해 실시간으로 들어오는 간판 영상에 대해, 상호명 텍스트를 뽑아내는 서비스를 개발하고자 하는 목적으로 이 프로젝트를 시작했다.
최근 다양한 비정형 문자에 대한 인식 기술 개발에 대한 관심이 증가하고 있다. 또한 영문 인식기와는 달리 한글 문자 인식기는 관련 공개 데이터셋이 부족해 관련 기술 개발에 어려움이 있다. 따라서 일상 생활에서 쉽게 접할 수 있는 상호, 간판, 도로표지판 등의 한글 문자 인식용 공개 데이터 중에서 ‘한글 상호명’ 문자 인식기를 만들어 보고자 한다.
개발한 모델은 상호명 인식 뿐만이 아니라 안전, 교통, 보안, 생활 편의 서비스 등에서 한글 문자 인식 기술을 개발하는 데 활용할 수 있으므로 확장가능성을 고려해 주제를 선정하였다.


## **Main Feature**

- 규격과 글자의 색상, 폰트가 다양한 간판 이미지에서 불필요한 문자를 제외한 상호명 텍스트만을 인식
- YOLO 모델을 이용해 간판 이미지를 입력으로 받아 상호명 텍스트가 포함된 이미지 영역을 추출하고, 해당 영역 이미지를 CRNN 모델의 입력으로 전달
- CRNN 모델의 적용 결과로 이미지 데이터였던 상호명을 텍스트 데이터로 변환해 결과로 출력
- 두 모델을 적용한 하나의 모델을 구현하며, 결과에 대해 WRA, 1-NED 평가방법으로 성능을 분석

## **Model**

### **YOLO**

이미지를 grid로 쪼갠 경계박스(bounding box)들 중에서 객체가 위치한 경계박스 찾기와 해당 객체의 class를 분류를 한 번에 진행하는 네트워크이다. 경계박스들 중 ROI(Region Of Interest) 혹은 객체 후보를 찾고 특정 threshold 값보다 작으면 지워 객체가 존재할 가능성이 높은 경계박스만을 남긴다. 그리고 각 grid sell에 대해 클래스 분류 점수를 이용하여 가장 높은 확률을 갖는 클래스로 분류한다. 마지막으로 NMS(Non-maxima suppression 비-최댓값 억제) 알고리즘으로 하나의 객체에 대한 중복된 경계박스를 제거해 객체를 탐지해낸다.

![https://user-images.githubusercontent.com/56228085/170872401-9541b3d0-92dd-41d0-8dd3-443a07a7c3f1.png](https://user-images.githubusercontent.com/56228085/170872401-9541b3d0-92dd-41d0-8dd3-443a07a7c3f1.png)

### **CRNN**

CRNN(Convolution Recurrent Neural Network) 알고리즘은 합성곱 신경망(Convolution Neural Network, CNN)와 재귀 신경망(Recurrent Neural Network, RNN)을 함께 고려한 알고리즘이다. CRNN은 주로 이미지 인식에 사용되는 CNN을 통해 이미지 내의 텍스트 특징을 추출하고, 시계열 데이터 처리에 유용한 RNN을 활용해 추출된 특징을 문자로 반환한다.

![https://user-images.githubusercontent.com/56228085/170872496-207e0517-5596-4934-86a5-4a3e562ac538.png](https://user-images.githubusercontent.com/56228085/170872496-207e0517-5596-4934-86a5-4a3e562ac538.png)

### **Process**

- step 1 : YOLO를 이용해 간판 영역에 대한 Bounding Box 검출하도록 학습시킨다. 
- step 2 : step 1의 결과로 넘어온 Bounding Box 내부에서 상호명 Text Line을 찾도록 학습시킴으로써 문자 탐지 모델을 구축한다.
- step 3 : step 2에서 Text Line을 이미지로 받은 후 CRNN 알고리즘을 통해 이미지에서 시계열 문자열인 상호명을 인식하도록 학습시킨다.

### Architecture
![image](https://user-images.githubusercontent.com/56228085/175829016-99197c53-22cf-418c-8358-1c845e5d57ec.png)

위와 같이 개발하고자 하는 모델은 자연 상의 간판 이미지에서 상호명을 추출하는 것을 최종 목표로 한다. 모델의 내부를 Black Box로 보면, 규격과 글자의 색상, 폰트가 다양한 간판 이미지를 입력으로 받아 불필요한 문자를 제외한 상호명 텍스트만을 탐지, 인식해 해당 텍스트를 출력으로 내보내는 모델이라고 할 수 있다. Black Box의 역할은 문자 탐지(Text Detection)와 문자 인식(Text Recognition)이다. Black Box의 내부 과정은 크게 YOLO와 CRNN의 두 단계로 나뉘며 각각이 문자 탐지와 문자 인식을 위한 알고리즘이다.

## 핵심 문제 정의 및 해결방안

### 핵심 문제 정의  
- YOLO와 CRNN 두 모델을 이용하여 하나의 텍스트 인식기 모델을 구현한다. 
- 영어와 특수문자 등 한글을 제외한 기호나 문자를 포함하고 있는 간판 이미지에도 상호명을 잘 추출한다. 
- 기존에 많이 제시되어온 글자 인식과는 다르게 상호명에 해당하는 “단어”를 인식해야 하므로 Sequential data를 처리하는 모델을 구축한다. 

### 본 팀이 제시한 해결 방법
- One-stage detector인 YOLO로 간판의 bounding box를 검출하고 bounding box에 있는 텍스트를 추출하는 CRNN을 합쳐, 세 단계의 학습을 통해 YOLO+CRNN Pretrained 모델을 구축한다. 
- YOLO를 이용해 간판 영역 상의 상호명 Text line 위치를 찾아내는 두번째 단계의 학습 시 영어를 한글보다 먼저 학습시킴으로써 한글 인식의 정확도를 높인다. ( 단, 특수문자를 학습시키기 위한 데이터를 아직 찾지 못해 특수문자를 인식하지 못하는 문제는 여전히 존재한다. )
- 이미지 처리에 쓰이는 CNN과 자연어 처리에 쓰이는 딥러닝 기법인 RNN을 함께 고려한 알고리즘인 CRNN 알고리즘을 사용함으로써 시계열 데이터 인식의 정확도를 높인다. 또한 테스트 방법으로 문자에 대한 정확도인 WRA와 문장 또는 단어의 정확도에 대한 1-NED방식을 모두 사용함으로써 시계열 문자 인식률을 검사하도록 한다. 

## Software Design

### Functional Requirements
- **데이터셋 라벨링** 

  학습하고자 하는 이미지 데이터셋에 대해 라벨링을 수행한다. 이미지에서 간판 영역을 찾아 좌표를 구한다.
- **간판 영역 검출 및 학습**

  간판 영역을 검출하는 YOLO 기반의 모델을 생성한다. 이미지의 라벨링 토대로 YOLO 모델을 학습시킨다.
- **Text Line 검출 및 학습** 
  
  이미지 내의 Text Line을 검출하는 YOLO 기반의 모델을 생성한다. 정확도를 높이기 위해 영어 Text Line을 선 학습 후, 한글 Text Line을 학습시킨다.
- **검출된 Text Line 중 상호명 Text Line 검출 및 이미지 크롭** 
  
  검출된 여러 Text Line 중 가장 많은 영역을 차지하는 Text Line을 상호명으로 검출한다. 해당 Text Line의 좌표를 이용해 이미지를 자른다.
- **Text 인식** 
  
  선별된 Text Line 이미지에서 Text를 인식하는 CRNN 기반의 모델을 생성한다.

## Software Architecture

### Non-Functional Requirements
- 개발 제약사항 

  python 언어로 개발, colab 사용
- 성능 

  사용자가 이미지를 입력하면 2초 이내에 텍스트를 반환하여야 한다. Test data 에 대해 90% 이상의 정확도가 되어야 한다.

### Use case Diagram
![image](https://user-images.githubusercontent.com/56228085/175829607-3e49f003-6209-4656-835c-5fe28b2b8d95.png)

### Deployment Diagram
![image](https://user-images.githubusercontent.com/56228085/175829626-8d67f146-ba26-4b49-97e2-09cbfb2b0a6c.png)

### Class Diagram
![image](https://user-images.githubusercontent.com/56228085/175829651-b242f667-ce8e-466b-8a56-4bbde10fbeb8.png)

### Sequnce Diagram
![image](https://user-images.githubusercontent.com/56228085/175829701-d3d3ff86-c2f7-41d0-bf29-d150b720cb07.png)

## 최종 산출물 구성 형태
- Application Layer
  - YOLOv4
  - CRNN
- Platform Layer
  - Tensorflow | Python Environment
  - Pytorch | Python Environment
  - OpenCV | Python Environment
- Infra Layer
  - OS : Windows 10
  - Machine: HW Machine

![image](https://user-images.githubusercontent.com/56228085/175829917-c530fb25-c40c-403c-8f9b-b03fb52f01ce.png)

## Risk Analysis & Risk Reduction Plan
### 구현 전 예상했던 어려움
간판 영역 검출에서의 어려움
: YOLO는 작은 객체에 취약하고, 다른 모델에 비해 다소 정확도가 떨어지는 문제가 발생할 것으로 예상했다. 
YOLO 마지막 단계에서 NMS(Non Maximum Suppression) 알고리즘을 적용하여 확률이 작은 간판을 제거하여 정확도를 높였다.

구현 및 학습 중 극복한 어려움
Signboard Detector
Dataset의 부족
: 간판 영역의 bounding box 좌표를 이용해 간판 영역을 검출하는 YOLO 모델을 학습시켜야 하는데, 라벨링된 간판 영역 dataset이 없다.
YOLO Mark를 이용해 직접 라벨링하여 dataset 구축
(참고 자료 : https://github.com/AlexeyAB/Yolo_mark)

하나의 이미지에서 여러 간판이 포함된 경우 정확도가 떨어짐
해당되는 이미지들을 추가로 라벨링하여 학습을 진행해 해결했다.

Textline Detector
한글과 영어 혼합 또는 영어 문자열 인식의 어려움
: 한글과 영어로 혼합된 Text Line을 인식해야하거나, 영어로 구성된 Text Line도 인식해야 하는 경우가 존재하는데, 한글 Text Line만 학습시키면 정확도 측면에서 문제가 생긴다.
영어 Text Line dataset 추가 확보했다.
영어 Text Line dataset 선학습을 진행했다.

상호명에 해당하는 Text Line 선별 알고리즘
: YOLO 기반의 Text Line Detector에서 Text Line을 모두 검출한 후, 그 중 상호명에 해당하는 Text Line을 선별해야 하는데 알고리즘이 존재하지 않는다.
간판 구성 상 상호명에 해당하는 Text Line의 영역이 가장 클 가능성을 이용해 Text Line의 경계 박스 좌표를 이용해 가장 큰 넓이를 구하는 알고리즘 구성했다.

nan loss 발생
오버플로우, 언더플로우 가능성이 있어 loss 값을 float32에서 float64로 cast하여 표현할 수 있는 loss의 비트 수를 늘렸다.
sqrt 값으로 0이 들어가 0으로 나누는 경우의 수가 생기기 때문에 loss 값에 1e-9(1nano)를 더해주었다.
overshooting 가능성이 있어 학습률을 감소시켰다.

	
Text Recognizer
상호명 영역만 잘린 간판 학습 데이터 부족
		: 직접 라벨링을 하다보니 sinfonizer의 첫번째, 두번째 단계를 거쳐 결과로서 얻게 되는 상호명 영역         이미지가 많지 않았으며, 무료로 제공되는 데이터셋이 없었다.
간판 영역 바운딩 박스 위치 라벨링이 제공되는 이미지들을 이용해 상호명 영역과 유사하게 자른 이미지를 생성하도록 파이썬 프로그램을 개발해 사용했다.



남아있는 극복 해야 할 어려움
한글 인식에 대한 어려움
	: 영문과 비교해 분류해야할 글자가 (영문 26 : 한글 900)로 약 35배 정도 차이가 나 같은 양의 데이터와 같은 학습양으로 정확도를 높이는 데에 어려움이 있다. 더불어 한글은 특성상 시계열 데이터이기 때문에 글자가 아닌 일종의 패턴으로 인식하여 검출하기 어려운 특성이 있다.
Scene text 인식의 어려움
: 이미지의 해상도, 방향, 그림자 등 여러 자연적인 요소로 인해 시스템 성능이 저하될 수 있다. 또한, 간판 이미지의 특성 상 이미지 내의 색상이 다양하고 글자의 크기와 폰트, 색상 등이 다양하기 때문에 글자 인식이 어려울 수 있다.
Text 인식 전 이미지 전처리 모듈 구성으로 개선될 것으로 예상된다.
향후 더 많은 dataset 확보를 통해 성능이 개선될 것으로 예상된다.

