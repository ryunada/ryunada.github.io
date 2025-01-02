---
title : Evaluation
categories:
    - MachineLearning
date: 2023-02-24
toc: true
toc_label: "Evaluation"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
---
# 성능 평가 지표(Evaluation Metric)

- 분류 성능 평가 지표
  - 정확도(Accuracy)
  - 오차행렬(Confusion Matrix)
  - 정밀도(Precision)
  - 재현율(Recall)
  - F1 Score
  - ROC AUC

## 1. 정확도

<img src="/assets/images/machinelearning/evaluation/evaluation_1.png" width = "50%">

<details>
    <summary> 함수 </summary>

```python
from sklearn import preprocessing

def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('N', inplace = True)
    df['Embarked'].fillna('N', inplace = True)
    df['Fare'].fillna(0, inplace = True)
    return df

def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = True)
    return df

def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df
```

</details>

  