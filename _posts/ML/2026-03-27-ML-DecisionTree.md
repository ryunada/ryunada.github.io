---
title : "[Python] ML-Decision Tree"
categories:
    - ML
date: 2026-03-27
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---
## 개념

Decision Tree는 데이터를 **조건으로 반복적으로 분할**하여 예측하는 알고리즘입니다. 마치 스무고개처럼, 각 노드에서 특정 조건으로 데이터를 나누고 최종 Leaf 노드에서 예측값을 반환합니다.

- **분류(Classification)** 와 **회귀(Regression)** 모두 사용 가능
- 결과를 트리 구조로 시각화할 수 있어 **해석이 쉬움**
- 별도의 스케일링 불필요

---

<img src = "/assets/img/ML/decisiontree/decision_tree.png" width = "70%" alt = "decision_tree">

## 트리 분할 기준

노드를 분할할 때 **불순도(Impurity)** 를 최소화하는 방향으로 조건을 선택합니다.

| 기준                   | 설명                                    | 사용          |
| ---------------------- | --------------------------------------- | ------------- |
| **지니 계수 (Gini)**   | 무작위로 선택한 샘플이 잘못 분류될 확률 | 분류 (기본값) |
| **엔트로피 (Entropy)** | 데이터의 불확실성 정도                  | 분류          |
| **MSE**                | 평균 제곱 오차                          | 회귀          |

### Gini vs Entropy 계산 예시

노드에 Class A 40개, Class B 60개가 있다고 가정합니다.

```
p_A = 0.4,  p_B = 0.6

Gini     = 1 - (0.4² + 0.6²) = 1 - (0.16 + 0.36) = 0.48
Entropy  = -(0.4 × log₂0.4 + 0.6 × log₂0.6)
         = -(0.4 × (-1.322) + 0.6 × (-0.737))
         = 0.971

완전히 순수한 노드 (한 클래스만 존재):
  Gini    = 1 - (1² + 0²) = 0
  Entropy = -(1 × log₂1)  = 0

→ 두 기준 모두 순수할수록 0, 불순할수록 커짐
→ 실무에서 성능 차이는 거의 없음 (Gini가 계산이 빠름)
```

---

## 언제 사용하는가

### 사용해야 할 때

| 상황                                        | 이유                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| **모델 결과를 비전문가에게 설명해야 할 때** | 트리 구조로 시각화하면 "왜 이런 예측을 했는지" 직관적으로 설명 가능 |
| **전처리를 최소화하고 싶을 때**             | 스케일링 불필요, 결측값 처리 최소화                          |
| **빠른 프로토타이핑**                       | 구현이 단순하고 학습이 빠름                                  |
| **변수 중요도가 필요할 때**                 | Feature Importance로 어떤 변수가 중요한지 빠르게 파악        |
| **앙상블 모델의 구조 이해**                 | Random Forest, XGBoost의 기반 알고리즘이므로 이해 필수       |

### 사용하지 말아야 할 때

| 상황                             | 이유                                          |
| -------------------------------- | --------------------------------------------- |
| **높은 예측 성능이 최우선일 때** | 단독 사용 시 앙상블 대비 성능 낮음            |
| **데이터가 적고 변수가 많을 때** | 과적합 위험 높음                              |
| **연속적인 수치 예측 (회귀)**    | 선형 패턴 표현 불가, 외삽(extrapolation) 불가 |

### 실무 활용 사례

```
금융: 대출 승인 여부 판단 (심사 기준을 트리로 시각화)
의료: 질병 진단 보조 (의사에게 판단 근거 제공)
마케팅: 고객 세그멘테이션 1차 분석
```

## 핵심 파라미터

| 파라미터            | 설명                            | 기본값 |
| ------------------- | ------------------------------- | ------ |
| `max_depth`         | 트리의 최대 깊이                | None   |
| `min_samples_split` | 노드 분할에 필요한 최소 샘플 수 | 2      |
| `min_samples_leaf`  | Leaf 노드의 최소 샘플 수        | 1      |
| `max_features`      | 분할 시 고려할 최대 변수 수     | None   |
| `criterion`         | 분할 기준 (`gini`, `entropy`)   | `gini` |

> `max_depth`를 제한하지 않으면 과적합(Overfitting)이 발생하기 쉽습니다.

---

## 실습 — Titanic 생존자 예측

### 데이터셋

- **Titanic 데이터셋**: 891명의 승객 정보로 생존 여부를 예측
- 사용 변수: `Pclass`, `Sex`, `Age`, `FamSize`, `Fare`, `Embarked`

### 전처리

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('./Data/Titanic.csv')

# 파생변수 생성
titanic['FamSize'] = titanic['SibSp'] + titanic['Parch']

# 분석 변수 선택
use_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked']
titanic = titanic[use_cols].dropna(subset=['Age'])

# 자료형 변환
titanic[['Survived', 'Pclass', 'Sex', 'Embarked']] = \
    titanic[['Survived', 'Pclass', 'Sex', 'Embarked']].astype('category')
titanic['Age'] = titanic['Age'].astype('int')

# One-Hot Encoding
titanic = pd.get_dummies(titanic, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

y = titanic['Survived']
X = titanic.drop(['Survived'], axis=1)

# 75 : 25 분할 (스케일링 불필요)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```

### 모델 학습

```python
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features=6,
    random_state=0
)

DT.fit(X_train, y_train)
```

### 트리 시각화

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))
plot_tree(DT,
          feature_names=X.columns,
          class_names=['사망', '생존'],
          filled=True,
          max_depth=3)   # 전체 트리는 너무 크므로 3단계만 표시
plt.show()
```

### 성능 평가

```python
from sklearn.metrics import accuracy_score, confusion_matrix

pred = DT.predict(X_test)
cfx  = confusion_matrix(y_test, pred)

sensitivity = cfx[1, 1] / (cfx[1, 0] + cfx[1, 1])
specificity = cfx[0, 0] / (cfx[0, 0] + cfx[0, 1])

print(f"정확도 : {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"민감도 : {sensitivity * 100:.2f}%")
print(f"특이도 : {specificity * 100:.2f}%")
print(f"Confusion Matrix:\n{cfx}")
```

```
정확도 : 78.21%
민감도 : 65.79%
특이도 : 87.38%
Confusion Matrix:
[[90 13]
 [26 50]]
```

### Feature Importance 시각화

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(DT.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

importances.plot(kind='barh', figsize=(8, 5), color='#1565C0')
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('중요도')
plt.tight_layout()
plt.show()
```

```
Sex_male      0.412   ← 성별이 가장 중요한 변수
Fare          0.198
Age           0.175
Pclass_3      0.089
FamSize       0.076
Embarked_S    0.031
Embarked_Q    0.019
```

---

## 과적합 방지 — 가지치기 (Pruning)

`max_depth`를 변화시키며 훈련/검증 성능 변화를 확인합니다.

```python
import matplotlib.pyplot as plt

train_scores, test_scores = [], []
depths = range(1, 20)

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=0)
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.plot(depths, train_scores, label='Train')
plt.plot(depths, test_scores, label='Test')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Depth에 따른 과적합 확인')
plt.show()
```

> 훈련 점수는 계속 오르지만 검증 점수가 내려가기 시작하는 지점이 최적 `max_depth`입니다.

---

## 장단점

**장점**

- 결과 해석이 직관적 (트리 시각화 가능)
- 스케일링 불필요
- 범주형 / 연속형 변수 모두 처리 가능

**단점**

- 과적합이 발생하기 쉬움
- 데이터 변화에 민감 (분산이 높음)
- 단독 사용보다 **앙상블(Random Forest, Gradient Boost)** 로 쓰는 것이 일반적

---

> 다음 포스팅: [[ML 시리즈 #2] K-Nearest Neighbor](02_knn.md)