var store = [{
        "title": "MachineLearning",
        "excerpt":"머신러닝의 개념. 머신러닝 규칙을 일일이 프로그래밍하지 않아도 자동으로 데이터에서 규칙을 학습하는 알고리즘을 연구하는 분야 대표적인 머신러닝 라이브러리 : 사이킷런(Scikit-learn) 지도학습(Supervised Learning) 분류(Classification) 회귀(Regression) 추천 시스템 시각 / 음성 감지/ 인지 텍스트 분석, NLP 비지도 학습(Un-supervised Learning) 클러스터링 차원 축소 강화학습 강화 학습(Reinforcement Learning) MachineLearning Skill 머신러닝 개념 데이터 전처리 Numpy...","categories": ["MachineLearning"],
        "tags": [],
        "url": "/machinelearning/MachineLearning/",
        "teaser": null
      },{
        "title": "Numpy",
        "excerpt":"-&gt; 파이썬의 대표적인 배열(array) 라이브러리 -&gt; 벡터 및 행렬 연산에 있어 매우 편리한 기능을 제공 1. Numpy 정의 import numpy as np 2. Array 정의 및 형태 확인 type 확인 : type(~) 배열 구조 확인 : ~.shape 배열 차원 확인하기 : ~.ndim # array정의 array1 = np.array([1,2,3]) print(f\"array1 type :...","categories": ["MachineLearning"],
        "tags": [],
        "url": "/machinelearning/Numpy/",
        "teaser": null
      },{
        "title": "Pandas",
        "excerpt":"1. Pandas란? 데이터 처리와 분석을 위한 라이브러리 행과 열로 이루어진 데이터 객체를 만들어 다룰 수 있음 대용량의 데이터들을 처리하는데 매우 편리 Pandas 자료구조 시리즈(Series) : 1차원 데이터프레임(DataFrame) : 2차원 패널(Panel) : 3차원 # conda install pandas import pandas as pd 2. Pandas 다루기 1. 파일 불러오기 read_csv(‘경로’) # pd.read_csv('경로') titanic_df...","categories": ["machinelearning"],
        "tags": [],
        "url": "/machinelearning/Pandas/",
        "teaser": null
      },{
        "title": "Scikit-Learn",
        "excerpt":"1. Scikit-learn 정의 -&gt; 파이썬 머신러닝 라이브러리 중 가장 많이 사용되는 라이브러리 특징 개발을 위한 편리한 프레임워크와 API를 제공 머신러닝을 위한 다양한 알고리즘 존재 Easy conda install scikit-learn import sklearn 2. Scikit-learn의 주요 기능 &lt;img src=”/assets/images/machinelearning/scikit-Learn/scikit-Learn_1.png” width=50%&gt; 1. Example Data sklean.datasets : 사이킷런에 내장되어 예제로 제공하는 데이터 세트 datasets.load_boston( )...","categories": ["MachineLearning"],
        "tags": [],
        "url": "/machinelearning/Scikit-Learn/",
        "teaser": null
      },{
        "title": "Evaluation",
        "excerpt":"성능 평가 지표(Evaluation Metric) 분류 성능 평가 지표 정확도(Accuracy) 오차행렬(Confusion Matrix) 정밀도(Precision) 재현율(Recall) F1 Score ROC AUC 1. 정확도 함수 ```python from sklearn import preprocessing def fillna(df): df['Age'].fillna(df['Age'].mean(), inplace = True) df['Cabin'].fillna('N', inplace = True) df['Embarked'].fillna('N', inplace = True) df['Fare'].fillna(0, inplace = True) return df def drop_features(df): df.drop(['PassengerId','Name','Ticket'], axis =...","categories": ["MachineLearning"],
        "tags": [],
        "url": "/machinelearning/Evaluation/",
        "teaser": null
      }]
