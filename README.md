# **Open Source Software Project1 - Deep Learning**

## **Project name : SINFONIZER(Signboard Information Recognizer)**

## **Members**

[강신규](https://github.com/zox004) | [이예빈](https://github.com/Yebeen-Lee) | [박세아](https://github.com/iamsseaa)

## **Goal**

자연 상의 간판 이미지에서 상호명을 추출하는 것이 최종 목표

## **Main Feature**

- 규격과 글자의 색상, 폰트가 다양한 간판 이미지에서 불필요한 문자를 제외한 상호명 텍스트만을 인식
- YOLO 모델을 이용해 간판 이미지를 입력으로 받아 상호명 텍스트가 포함된 이미지 영역을 추출하고, 해당 영역 이미지를 CRNN 모델의 입력으로 전달
- CRNN 모델의 적용 결과로 이미지 데이터였던 상호명을 텍스트 데이터로 변환해 결과로 출력
- 두 모델을 적용한 하나의 모델을 구현하며, 결과에 대해 WRA, 1-NED 평가방법으로 성능을 분석

## **Dataset**

- [AI Hub](https://aihub.or.kr/)

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
