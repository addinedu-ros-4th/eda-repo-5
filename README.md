# 뉴스데이터를 활용한 주가분석

## 1. 개요 
### 1-1) 프로젝트 목표
- 뉴스 데이터를 통해 주식 흐름에 대한 인사이트를 얻기 위하여, 뉴스에 대한 [호재/악재/중립] 감성 분석 및 주식 그래프와의 상관 관계 분석
- 마인드 맵
  
![mindmap](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/fa417f46-fe48-4d28-ad81-f18470348d32)
### 1-2) 프로젝트 기간 <2024.01.17 - 2024.01.24>
  - 이슈 생성 및 우선 순위 구분, 개발 일정 관리
  
![process (3)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/a49faacd-348f-4767-b642-44d46cccff2b)
  ![importance (1)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/c8eb2a42-7b41-4483-bcbf-729d8e1d4836)

### 1-3) 구성원 및 담당
  - 조성호(조장) : 웹 크롤링, 뉴스 요약, 데이터 전처리, 주식 데이터 수집
  - 김태형 : 웹 크롤링, 감성분석 모델 학습 및 뉴스 데이터 감성 분석
  - 이정욱 : 웹 크롤링, 감성분석 결과 및 주가 데이터 시각화, 주식 데이터 수집
  - 이지호 : 웹 크롤링, 뉴스 기사 데이터 형태소 분리, 감성분석 결과 및 데이터 시각화

### 1-4) 기술 키워드
![Untitled](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/24776159-13d1-43ed-9f08-7d27bcb54d30)
  
  
## 2. 프로젝트 결과
### 2-1) 뉴스 기사 개수와 거래량과의 관계

![samsung (2)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/e98c0347-eec5-4295-973f-c119920b3e8e)

  - 뉴스 기사 갯수가 많을 때 거래량이 상승하는 것을 볼 수 있음

### 2-2) 뉴스 감성점수와 주가와의 관계

![samsung1 (1)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/647c788a-867f-46a7-92af-5cac968051c6)

  - 뉴스 감성점수가 높으면 주가가 상승하는 것을 볼 수 있음

## 3. 기능별 구현 과정
- 프로젝트 진행 과정 : 웹 크롤링 -> 형태소 분리 -> 감성 분석 -> 데이터 시각화
![readme flow drawio (1)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/1d8fd744-cbde-4ffe-82ce-7e65ef4ed4a9)

⇒ 관심 산업 분야 및 기업에 대한 뉴스 데이터를 웹 크롤링을 통해 수집 및 DB 관리

⇒ 감성 분석 모델을 적용하여 [호재/악재/중립]에 대해 Labeling 진행하여 데이터 DB화


### 3-1) 웹 크롤링
  - 비교적 html의 패턴이 정해져 있고 공신력이 있는 기사들만 모아놨다고 판단되는 네이버 뉴스 기사들을 기준으로 크롤링
  - 주요 산업군을 검색어로 설정하여 1년 단위 기사 크롤링

### 3-2) 형태소 분리
  - 오픈소스로 제공된 여러 형태소 분리 모델 중 성능적으로 가장 우수하다고 알려진 mecab 사용

### 3-3) 감성 분석
- SKT-Brain의 오픈소스 KoBERT 한글 감성분석 pre-trained 모델 사용
- 금융 관련 한글 뉴스 말뭉치(corpus) 데이터 4,846개 전이학습(transfer learning) 진행 및 모델 생성
- 뉴스 데이터를 모델에 입력하여 predict를 통한 감성 분석 결과 데이터 생성
    1. 뉴스 데이터 news.csv 파일을 pandas dataframe으로 read
    2. 감성분석 결과 및 예측 점수에 대한 column 추가
    3. predicted_news.csv 데이터 저장
 
### 3-4) 데이터 시각화
- 레인보우로보틱스 1년치 기사 중 가장 많이 나온 키워드 분석

![Untitled (1)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/f43013e6-9441-46c2-830b-d71449eb7a04)

- 레인보우로보틱스 1년치 기사의 월당 Top 5 키워드 분석

![Untitled (2)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/b5360941-dcc2-4635-9c9e-6a934d695a4e)


- 감성 분석이 끝난 데이터 DB를 이용 월별 호재, 악재 감성 점수 추이

  - 호재는 +1, 악재는 -1로 일괄 계산
  
![rainbow_gamsung](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/d1dc69ad-2e0c-4395-88d2-6e2cfb063ca3)

  - 실제 주가 데이터와 감성분석을 토대로 예측한 경향 비교

![rainbow](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/a2a7c4b7-5fb1-4282-8f7c-f8dd14eb43fd)
  
  - 키워드 별 주가에 얼마나 영향을 끼치는지 회귀분석

![123123123 (1)](https://github.com/addinedu-ros-4th/eda-repo-5/assets/157219758/fa684e06-e4d1-4f58-8e54-17398b858235)

### 3-5) 프로젝트 진행 중 발생한 이슈 및 해결 과정
  - 웹 크롤링
      - API 활용 방안 모색 중 특정 기간 / 검색어 설정에 있어 제한사항 **有**
          
          → 네이버 뉴스 기사 크롤링 진행
          
      - 특정 검색어로 기사 검색 시 하루에 발행되는 기사 갯수가 상이
          
          → 하루 최대 100개의 기사만 크롤링 하는 것으로 설정
          
      - 웹 크롤링 중 네이버 기사의 html 형태가 다른 기사들이 존재하여 기사를 제대로 크롤링 하지 못하고 중지되는 문제 발생
          
          → “news.naver.com”이 포함되는 url만 크롤링 하도록 예외 처리
          
  - 형태소 분리
      - 오픈소스로 나와있는 불용어 사전만으로는 불용어 검출이 제대로 되지 않음
          
          → 수기로 추가하여 주었음
          
  - 감성분석
      - 하이퍼-파라미터 튜닝
          
          ```jsx
          max token length = 128
          batch size : 22
          epochs : 20
          learning rate =  5e-5
          ```
          
          - 학습 결과 : val_accuracy = 0.86
      - BERT 기반 감성 분석 모델의 input data가 최대 500 tokens 제한
      - 이 분석 하는 text 양이 제한적
          - 뉴스 데이터 요약하는 모델 사용
              
              →  중요하지 않은 문장만 요약하는 경우 발생
              
          - 뉴스 전체를 문장 별로 나눠 각 문장별로 호재/악재/중립으로 나눠 합산
              
              → 분석하는데 시간이 매우 오래 걸림
              
      1. 뉴스 데이터 news.csv 파일을 pandas dataframe으로 read
      2. 감성분석 결과 및 예측 점수에 대한 column 추가
      3. predicted_news.csv 데이터 저장

## 4. 관련 자료


## 5. 레퍼런스
  - 불용어 사전 오픈소스  
    https://github.com/stopwords-iso/stopwords-ko
  - KoBERT
    https://github.com/SKTBrain/KoBERT

