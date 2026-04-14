---
title : "[Python] ML-Random Forest"
categories:
    - ML
date: 2026-04-04
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---

# Random Forest 개념
Random Forest는 **여러 개의 Decision Tree를 독립적으로 학습**시키고 그 결과를 다수결(분류) 또는 평균(회귀)으로 합치는 **앙상블(Ensemble)** 알고리즘

```
단일 Decision Tree의 문제:
→ 데이터 변화에 민감하고 과적합하기 쉬움

Random Forest의 해결책:
→ 다양한 트리를 만들고 결과를 합쳐 분산을 줄임
```

---

<img src = "/assets/img/ML/randomforest/random_forest.png" width = "70%" alt = "random_forest">

## 두 가지 핵심 전략

### 1. Bagging (Bootstrap Aggregating)

```
원본 데이터에서 복원 추출(Bootstrap)로 서브셋 생성
→ 각 트리가 서로 다른 데이터로 학습
→ 트리마다 다른 패턴을 학습하여 다양성 확보
```

### 2. Feature Randomness (변수 무작위 선택)

```
각 노드 분할 시 전체 변수가 아닌 일부만 랜덤하게 사용
→ 트리 간 상관관계 낮춤
→ 더 독립적인 앙상블 구성
```

---

## 언제 사용하는가

### 사용해야 할 때

| 상황                                           | 이유                                                 |
| ---------------------------------------------- | ---------------------------------------------------- |
| **빠르게 좋은 성능의 베이스라인이 필요할 때**  | 기본 파라미터만으로도 준수한 성능, 튜닝 부담 적음    |
| **결측값이 있고 전처리를 줄이고 싶을 때**      | 결측값에 비교적 강건, 스케일링 불필요                |
| **변수 중요도를 파악하고 싶을 때**             | Feature Importance가 안정적이고 신뢰도 높음          |
| **과적합이 걱정될 때**                         | Bagging으로 분산 감소, Decision Tree보다 훨씬 안정적 |
| **데이터 크기가 중간 규모 (수만 ~ 수십만 건)** | 병렬 학습으로 빠른 처리                              |

### 사용하지 말아야 할 때

| 상황                                    | 이유                                            |
| --------------------------------------- | ----------------------------------------------- |
| **실시간 예측 or 메모리가 제한적일 때** | 수백 개의 트리를 메모리에 저장 → 무거움         |
| **최고 성능이 필요할 때**               | XGBoost, LightGBM이 일반적으로 성능 우위        |
| **선형 관계가 강한 데이터**             | 선형 모델(Logistic Regression, Ridge)이 더 적합 |

### 실무 활용 사례

```
금융: 사기 거래 탐지 (불균형 데이터에서도 강건)
의료: 환자 재입원 예측
제조: 불량품 탐지, 예측 정비
마케팅: 고객 이탈 예측 (빠른 배포 필요 시)
```

## 핵심 파라미터

| 파라미터            | 설명                                 | 기본값 |
| ------------------- | ------------------------------------ | ------ |
| `n_estimators`      | 생성할 트리 수                       | 100    |
| `max_depth`         | 각 트리의 최대 깊이                  | None   |
| `max_features`      | 분할 시 사용할 변수 수 (`sqrt` 권장) | `sqrt` |
| `min_samples_split` | 노드 분할 최소 샘플 수               | 2      |
| `min_samples_leaf`  | Leaf 노드 최소 샘플 수               | 1      |

> `n_estimators`가 클수록 성능은 안정되지만 학습 시간이 늘어납니다. 보통 100~300이면 충분합니다.

---

## 실습 — Titanic 생존자 예측

### I. Library & Data Load
```python
import pandas as pd
import numpy as np

# Data Load
titanic = pd.read_csv("./Data/Titanic.csv")
titanic
```

### II. Preprocessing

#### II-I. Feature Engineering
```python
# 가족 변수 추가
titanic['FamSize'] = titanic['SibSp'] + titanic['Parch']  

# 분석에 사용할 변수만 선택
use_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked']

# 결측값 제거
titanic = titanic[use_cols].dropna(subset = ['Age'])

# 변수 형태 변경
titanic[['Survived', 'Pclass', 'Sex', 'Embarked']] = titanic[['Survived', 'Pclass', 'Sex', 'Embarked']].astype('category')
titanic['Age'] = titanic['Age'].astype('int')

# One-Hot-Encoding
titanic = pd.get_dummies(titanic, columns = ['Pclass', 'Sex', 'Embarked'], drop_first = True)
```

### II-II. Train & Test Split
```python
from sklearn.model_selection import train_test_split

y = titanic['Survived']
X = titanic.drop(['Survived'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### III. Model Train
```python
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(
    n_estimators = 200,
    max_depth = 10,
    max_features = 'sqrt',
    random_state = 0
)

RF.fit(X_train, y_train)
```

### 변수 중요도 시각화

```python
import matplotlib.pyplot as plt
import pandas as pd

importances = pd.Series(RF.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8, 5))
plt.title('Feature Importance')
plt.xlabel('중요도')
plt.tight_layout()
plt.show()
```

### OOB Score (Out-of-Bag Score)

Bagging에서 각 트리는 복원 추출로 데이터를 선택하기 때문에, 선택되지 않은 약 37%의 데이터가 생깁니다. 이를 OOB(Out-of-Bag) 샘플이라 하며, 별도의 검증셋 없이 성능을 추정할 수 있습니다.

```python
RF_oob = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    oob_score=True,   # OOB Score 활성화
    random_state=0
)
RF_oob.fit(X_train, y_train)

print(f"OOB Score : {RF_oob.oob_score_ * 100:.2f}%")
# OOB Score는 교차 검증 점수와 유사하게 해석 가능
```

```
OOB Score : 81.93%
```

### 성능 평가

```python
from sklearn.metrics import accuracy_score, confusion_matrix

pred = RF.predict(X_test)
cfx  = confusion_matrix(y_test, pred)

sensitivity = cfx[1, 1] / (cfx[1, 0] + cfx[1, 1])
specificity = cfx[0, 0] / (cfx[0, 0] + cfx[0, 1])

print(f"정확도 : {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"민감도 : {sensitivity * 100:.2f}%")
print(f"특이도 : {specificity * 100:.2f}%")
```

```
정확도 : 82.68%
민감도 : 73.68%
특이도 : 88.35%
```

---

## Bagging vs Boosting

Random Forest가 속한 Bagging과 다음 포스팅부터 다룰 Boosting을 비교합니다.

|              | Bagging (Random Forest) | Boosting (AdaBoost, XGBoost...) |
| ------------ | ----------------------- | ------------------------------- |
| 학습 방식    | 병렬                    | 순차                            |
| 각 트리 관계 | 독립적                  | 이전 결과에 의존                |
| 목적         | 분산 감소               | 편향 감소                       |
| 과적합       | 상대적으로 강함         | 노이즈에 민감할 수 있음         |
| 속도         | 빠름 (병렬 가능)        | 느림                            |

---

## 장단점

**장점**

- Decision Tree 대비 과적합에 강함
- 변수 중요도 제공 → 해석 가능
- 스케일링 불필요
- 결측값에 비교적 강건

**단점**

- 트리 수가 많아지면 메모리/시간 비용 증가
- 개별 트리보다 해석이 어려움
- Boosting 계열보다 성능이 낮을 수 있음