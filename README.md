# Cococam
:computer: 2019 Hanium Project 


**1. 프로젝트 소개**

1.1 하드웨어의 한계
- 최신의 모바일 디바이스와 달리 싱글 렌즈만을 탑재한 구형 및 보급형 모바일 디바이스에서는 이러한 카메라 성능 및 고품질의 이미지를 얻기 어렵습니다. 
- 본 프로젝트는 제한된 카메라 렌즈의 개수라는 하드웨어의 한계를 소프트웨어로서 극복하는 것을 목표로 하였습니다.

1.2 아웃포커스
- 사진기와 피사체 간의 거리가 가까울수록 심도가 얕아집니다. 같은 초점거리와 조리개 값을 유지하면서 피사체와의 거리와 배경의 거리를 조절함으로써 이 효과를 극대화시킬 수 있습니다. 
- 해당 프로젝트에서는 듀얼 또는 멀티 카메라를 갖는 고사양의 디바이스에서 가능하였던 아웃포커스 기능을 하드웨어가 아닌 소프트웨어를 이용한 방식으로 저사양의 디바이스에서도 서비스 할 수 있도록 합니다. 

1.3 이미지 개선
- 대개 스마트폰에서의 줌(Zoom)이라 함은 이미지의 일부분만을 확대하여 사용하는 디지털 줌 방식을 이용합니다. 예를 들어 포토샵의 이미지 리사이즈 기능을 이용하여 '크기'만을 늘려놓은 것입니다. 빛이 아닌 이미 촬영한 것에 대한 사진 중에서 적은 양의 데이터를 부풀려 놓았기 때문에 화질 저하가 발생합니다. 
- 기존 카메라에서 줌 기능 시에 발생하는 화질 저하 문제를 Image Upscailing 및 Super Resolution 방식을 통하여 개선하는 효과를 기대합니다.


**2. 주요 기능**

> 이미지 아웃포커싱 기능
- 딥러닝 Mask R-CNN 알고리즘을 통한 이미지 아웃포커싱 후처리
- 화질 개선이 이루어진 이미지에 대하여 전경-배경 분리를 통하여 디테일한 부분까지 아웃포커스 효과 적용 </br>
<img src="/Images/descriptions/outfocusing.png" width="300px" height="300px"/></img>

> 이미지 화질 개선
- 딥러닝 ESPCN 알고리즘을 통한 확대된 이미지의 화질 개선
- 카메라 Zooming 한 경우에 한하여 실행</br>
<img src="/Images/descriptions/upscailing.png" width="300px" height="150px"></img>

**3. 시스템 아키텍쳐**

<img src="/Images/system.jpeg" wwidth="300px" height="300px"></img>

**3. 디자인**

<img src="/Images/app_icon1.png" width="30%" height="30%"></img>
<img src="/Images/full_logo_pk1.png" width="50%" height="10%"/>

**4. 화면**

<img src="/Images/screenshot.jpeg" width="300px" height="150px"></img>

