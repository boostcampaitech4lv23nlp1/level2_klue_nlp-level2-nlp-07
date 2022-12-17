# 부스트캠프 4기 NLP 07조 염보라
## Members
---

김한성|염성현|이재욱|최동민|홍인희|
:-:|:-:|:-:|:-:|:-:
<img src='https://user-images.githubusercontent.com/44632158/208237676-ae158236-16a5-4436-9a81-8e0727fe6412.jpeg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/44632158/208237686-c66a4f96-1be0-41e2-9fbf-3bf738796c1b.jpeg' height=80 width=80px></img>|<img src='' height=80 width=80px></img>|<img src='' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/97818356/208237742-7901464c-c4fc-4066-8a85-1488d56e0cce.jpg' height=80 width=80px></img>|
[Github](https://github.com/datakim1201)|[Github](https://github.com/neulvo)|[Github](https://github.com/datakim1201)|[Github](https://github.com/datakim1201)|[Github](https://github.com/datakim1201)
&nbsp;

## Wrap up report
[project report 바로가기](https://github.com/boostcampaitech4lv23nlp1/level2_klue_nlp-level2-nlp-07/blob/main/NLP%20%EA%B4%80%EA%B3%84%EC%B6%94%EC%B6%9C_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(07%EC%A1%B0).pdf)

&nbsp;

# 관계 추출 과제(RE task)
## 프로젝트 수행 기간
>11/14 ~ 12/1
## 프로젝트 개요
---
>관계 추출(Relation Extraction)은 **<span style="color: #0000CD">문장의 단어(Entity)에 대한 속성과 관계를 예측**하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triplet을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

<img src ='https://user-images.githubusercontent.com/44632158/208237724-20d69cac-b5bd-4a68-bd0c-27c1481804c2.png'>

&nbsp;

## 데이터 설명
>전체 데이터에 대한 통계는 다음과 같습니다.
- train.csv: 총 32,470개
- test_data.csv: 총 7,765개 (정답 라벨 blind = 100으로 임의 표현)   

>데이터셋에 대한 자세한 설명은 아래 링크를 참조하시기 바랍니다.   
[Klue Dataset Link](https://klue-benchmark.com/)

&nbsp;
## 프로젝트 세부 내용
### 데이터
- EDA
- Preprocesisng, Re-labeling
- Data Augmentation
- Noise data experiment
### 모델
- Add Marker
- Binary Classification(Experimental)
- RECENT
- R-BERT

### 최적화
- LRFinder
- Scheduler
- Focal Loss
### 후처리
- Inference Analysis
- CoRE(Debiasing)

&nbsp; 

## 프로젝트 구조
---
```
RE Project/
│
├── code/ 
│   ├── config/
│   ├── ...
│   └── main.py
│
├── for_binary/
│   ├── config/
│   ├── data/
│   ├── ...
│   ├── run.sh
│   └── inference.py - for all setting
│
├── multi_binary/
│   ├── config/
│   ├── ...
│   ├── binary_train.sh
│   ├── multiple_train.sh
│   └── para.train.sh - for recursive learning
│
│── rBERTa/ - for further reading
│   ├── ...
│   └── model for some code check more detail..
│
├── notebooks/
│   ├── ...
│   ├── EDA.ipynb
│   ├── noise_add.ipynb
│   └── dev_prob.ipynb
│
├── .gitignore
├── README.md
│
└── thanks for comming I'm Yeombora
```
