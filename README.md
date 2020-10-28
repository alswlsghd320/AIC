# AI Championship : Action recogtion on MBN dataset

### 1. Preparing dataset
#### 데이터셋 : 한국데이터거래소(KDX) 제공 데이터
MBN Entertainment Youtube 채널에 존재하는 영상 데이터 정보 파일 (*.xlsx) \
약 1,000여개 Youtube 영상의 행동 카테고리, 행동 시점 레이블링 파일 (*.csv) \
30여개의 행동 카테고리를 별도 제시 \
-> MBN 영상에 적용 가능한 Kinectics 400 벤치마크 기반으로 구성 

## Installation

```
git clone https://github.com/alswlsghd320/AIC.git 
cd AIC 
conda create -f environment.yml 

# Do NOT command 'pip install apex'
git clone https://www.github.com/nvidia/apex \
cd apex \
python3 setup.py install \
```

## Demo
The following example notebooks are provided: click [here](train.ipynb) 




