# 뉴스데이터를 활용한 주가분석

## 1. 개요 
### 1-1) 프로젝트 목표 :
- 뉴스 데이터를 통해 주식 흐름에 대한 인사이트를 얻기 위하여, 뉴스에 대한 [호재/악재/중립] 감성 분석 및 주식 그래프와의 상관 관계 분석
- 마인드 맵
  
![mindmap](https://github.com/cccsssshh/log_repository/assets/157219758/834d1d67-a073-455f-a3f5-90f8243281c2)
### 1-2) 프로젝트 기간 : 2023.01.17 - 2023.01.24
  - 이슈 생성 및 우선 순위 구분, 개발 일정 관리

  - 프로젝트 흐름 및 중요도
  
![process (3)](https://github.com/cccsssshh/log_repository/assets/157219758/666ab173-4a41-4e22-80a0-3eb7b7f8a4d6)
  ![importance (1)](https://github.com/cccsssshh/log_repository/assets/157219758/30ed229d-9511-45bc-a98d-0f22087b31a3)

### 1-3) 구성원 및 담당 :
  - 조성호(조장) : 웹 크롤링, 뉴스 요약, 데이터 전처리, 주식 데이터 수집
  - 김태형 : 웹 크롤링, 감성분석 모델 학습 및 뉴스 데이터 감성 분석
  - 이정욱 : 웹 크롤링, 감성분석 결과 및 주가 데이터 시각화, 주식 데이터 수집
  - 이지호 : 웹 크롤링, 뉴스 기사 데이터 형태소 분리, 감성분석 결과 및 데이터 시각화

### 1-4) 기술 키워드
  mysql, python, pandas, matplotlib, seaborn, FinanceDataReader, mecab, kovert
![Untitled](https://github.com/cccsssshh/log_repository/assets/157219758/b94a28bb-e5a1-47ac-982d-acdb531df91a)

  
  
## 2. 프로젝트 결과
### 2-1) 뉴스 기사 개수와 거래량과의 관계

![samsung (2)](https://github.com/cccsssshh/log_repository/assets/157219758/e33ee3b9-010e-4da1-89f0-6a6dfca14b12)

  - 뉴스 기사 갯수가 많을 때 거래량이 상승하는 것을 볼 수 있음

### 2-2) 뉴스 감성점수와 주가와의 관계

![samsung1 (1)](https://github.com/cccsssshh/log_repository/assets/157219758/08fd435f-2499-4999-82a6-d4a26cb3cbe7)

  - 뉴스 감성점수가 높으면 주가가 상승하는 것을 볼 수 있음

## 3. 기능별 구현 과정
- 프로젝트 진행 과정 : 웹 크롤링 -> 형태소 분리 -> 감성 분석 -> 데이터 시각화
![readme flow drawio (1)](https://github.com/cccsssshh/log_repository/assets/157219758/2993b2a1-e9d4-471f-8d94-9e4e7160180e)

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

![Untitled (1)](https://github.com/cccsssshh/log_repository/assets/157219758/071ec264-8b73-4ca7-99cd-83cde1ed2297)

- 레인보우로보틱스 1년치 기사의 월당 Top 5 키워드 분석

![Untitled (2)](https://github.com/cccsssshh/log_repository/assets/157219758/0235b0e3-7bee-46de-9fce-a6c356fadbe8)


- 감성 분석이 끝난 데이터 DB를 이용 월별 호재, 악재 감성 점수 추이

  - 호재는 +1, 악재는 -1로 일괄 계산
  
![rainbow_gamsung](https://github.com/cccsssshh/log_repository/assets/157219758/d1cbfc79-49ae-4875-b61b-57a3fe969e6d)

  - 실제 주가 데이터와 감성분석을 토대로 예측한 경향 비교

![rainbow](https://github.com/cccsssshh/log_repository/assets/157219758/c2049fc2-0d44-4933-a3bd-b59a967bb584)
  
  - 키워드 별 주가에 얼마나 영향을 끼치는지 회귀분석

![123123123 (1)](https://github.com/cccsssshh/log_repository/assets/157219758/9f92345c-fad7-45d0-a0cd-5671dcc2d66a)

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

