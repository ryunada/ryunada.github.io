---
title : "[Python] ML_Logistic Regression"
categories:
    - ML
date: 2026-05-24
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---


# Logistic Regression(로지스틱 회귀) 개념

Logistic Regression은 이름에 "회귀"가 붙지만 **분류 알고리즘**이다. 선형 결합 값을 **시그모이드(Sigmoid) 함수**로 변환해 0~1 사이의 확률로 출력하고, 임계값(기본 0.5)을 기준으로 클래스를 결정한다.

```
선형 결합(z) → 시그모이드 → 확률(0~1) → 임계값 → 클래스 예측

이진 분류의 대표 알고리즘 (다중 클래스도 지원: OvR / Softmax)
확률 해석 가능 → "이 샘플이 클래스 1일 확률: 73.2%"
선형 결정 경계 → 비선형 문제에는 Feature Engineering 필요
정규화 내장 (C 파라미터)
```

---

<p style="text-align: center;">
    <img src = "/assets/img/ML/lr/lr_framework.png" width = "70%" alt = "lr_framework">
</p>


## 핵심 수식

### 시그모이드 함수 (Sigmoid Function)

선형 결합 $z = w_0 + w_1x_1 + \cdots + w_nx_n$ 을 확률로 변환한다.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

| $z$ 값          | $\sigma(z)$ | 해석          |
| --------------- | ----------- | ------------- |
| $z \to +\infty$ | $\approx 1$ | 클래스 1 확실 |
| $z = 0$         | $0.5$       | 결정 경계     |
| $z \to -\infty$ | $\approx 0$ | 클래스 0 확실 |

### 손실 함수 (Binary Cross-Entropy)

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i)\right]$$

```
예측 확률이 실제 레이블과 가까울수록 손실 → 0
예측이 틀릴수록 손실 → ∞ (log 특성)
경사하강법(Gradient Descent)으로 가중치 업데이트
```

---

## 정규화 (Regularization)

과적합을 방지하기 위해 손실 함수에 **패널티 항**을 추가한다.

| 정규화         | 수식                                                       | 특징                                        |
| -------------- | ---------------------------------------------------------- | ------------------------------------------- |
| **L2 (Ridge)** | $\mathcal{L} + \lambda\sum w_i^2$                          | 가중치를 0에 가깝게 축소 (기본값)           |
| **L1 (Lasso)** | $\mathcal{L} + \lambda\sum \|w_i\|$                        | 일부 가중치를 정확히 0으로 → 자동 변수 선택 |
| **ElasticNet** | $\mathcal{L} + \lambda_1\sum\|w_i\| + \lambda_2\sum w_i^2$ | L1 + L2 혼합                                |

```
sklearn에서 C = 1 / λ (C가 클수록 정규화 약함 → 과적합 위험)
C가 작을수록 강한 정규화 → 과소적합 위험
```

---

## 언제 사용하는가

### 사용해야 할 때

| 상황                                  | 이유                               |
| ------------------------------------- | ---------------------------------- |
| **이진/다중 클래스 분류**             | 가장 검증된 선형 분류 알고리즘     |
| **확률 값이 필요할 때**               | `predict_proba()`로 신뢰도 반환    |
| **빠른 베이스라인**                   | 학습이 빠르고 해석이 명확          |
| **변수 중요도 파악**                  | 계수(coefficient) 크기로 직접 해석 |
| **데이터 선형 분리 가능성이 있을 때** | 선형 결정 경계로 충분한 경우       |

### 사용하지 말아야 할 때

| 상황                                | 이유                                     |
| ----------------------------------- | ---------------------------------------- |
| **비선형 결정 경계가 필요할 때**    | 선형 모델이므로 복잡한 패턴 학습 불가    |
| **변수 간 상관관계가 매우 높을 때** | 다중공선성 문제 → 계수 불안정            |
| **변수가 수천 개 이상일 때**        | L1 정규화로 해결 가능하나 Tree 계열 고려 |

### 실무 활용 사례

```
의료     : 환자의 질병 발병 확률 예측
금융     : 대출 연체 여부 예측 (신용 스코어링)
마케팅   : 이메일 클릭/전환 여부 예측
스팸 필터: 메일이 스팸일 확률 분류
```

---

## Parameters

| Parameter      | 설명                                    | Default   |
| -------------- | --------------------------------------- | --------- |
| `C`            | 정규화 강도의 역수 (클수록 정규화 약함) | `1.0`     |
| `penalty`      | 정규화 종류                             | `'l2'`    |
| `solver`       | 최적화 알고리즘                         | `'lbfgs'` |
| `max_iter`     | 최대 반복 횟수                          | `100`     |
| `multi_class`  | 다중 클래스 처리 방식                   | `'auto'`  |
| `random_state` | 재현성 시드                             | `None`    |

- `C` : 정규화 강도 — **가장 중요한 파라미터**
  - 값 변화별 효과
    - `C` 클수록 → 정규화 약함 → 훈련 데이터에 더 맞춤 → 과적합 위험
    - `C` 작을수록 → 정규화 강함 → 가중치 축소 → 과소적합 위험
  - 교차 검증으로 최적값 탐색 권장 (보통 `[0.001, 0.01, 0.1, 1, 10, 100]`)

- `penalty` : 정규화 방식
  - `'l2'` → Ridge 정규화 (기본값, 대부분의 solver 지원)
  - `'l1'` → Lasso 정규화 → `solver='liblinear'` 또는 `'saga'` 필요
  - `'elasticnet'` → `solver='saga'` 필요
  - `None` → 정규화 없음

- `solver` : 최적화 알고리즘
  - `'lbfgs'` → 중소규모, 다중 클래스에 권장 (기본값)
  - `'liblinear'` → 소규모 데이터, L1 정규화 지원
  - `'saga'` → 대규모 데이터, L1/ElasticNet 지원

---

## 실습 — Titanic 생존자 예측

### I. Library & Data Load

```python
import warnings
warnings.filterwarnings(action = 'ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
```

```python
titanic = pd.read_csv('./Data/Titanic.csv')
titanic
```

### II. Preprocessing
#### II-I. Feature Engineering

```python
titanic['FamSize'] = titanic['SibSp'] + titanic['Parch']

use_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked']
titanic = titanic[use_cols].dropna(subset = ['Age'])

titanic[['Survived', 'Pclass', 'Sex', 'Embarked']] = \
    titanic[['Survived', 'Pclass', 'Sex', 'Embarked']].astype('category')
titanic['Age'] = titanic['Age'].astype('int')

titanic = pd.get_dummies(titanic, columns = ['Pclass', 'Sex', 'Embarked'], drop_first = True)
```

#### II-II. Train & Test Split

```python
y = titanic['Survived']
X = titanic.drop(['Survived'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Logistic Regression도 스케일링 권장 (수렴 속도 개선)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
```

### III. 최적 C 탐색 (GridSearchCV)

```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

lr_cv = GridSearchCV(
    LogisticRegression(penalty = 'l2', solver = 'lbfgs', max_iter = 1000),
    param_grid,
    cv = 5,
    scoring = 'accuracy'
)
lr_cv.fit(X_train, y_train)

best_C = lr_cv.best_params_['C']
print(f"최적 C: {best_C},  CV Accuracy: {lr_cv.best_score_:.4f}")

# C에 따른 정확도 시각화
results = pd.DataFrame(lr_cv.cv_results_)
plt.semilogx(param_grid['C'], results['mean_test_score'], marker = 'o')
plt.xlabel('C (정규화 역수)')
plt.ylabel('CV Accuracy')
plt.title('C 값에 따른 교차 검증 정확도')
plt.grid(True)
plt.show()
```

<img src = "/assets/img/ML/lr/lr_c_search.png" width = "70%" alt = "lr_c_search">

### IV. Model Train

```python
LR = LogisticRegression(
    C = best_C,
    penalty = 'l2',
    solver = 'lbfgs',
    max_iter = 1000,
    random_state = 0
)
LR.fit(X_train, y_train)
```

### V. 계수 시각화 (Feature Importance)

```python
feature_names = X.columns.tolist()
coef = LR.coef_[0]

plt.figure(figsize = (8, 5))
colors = ['tomato' if c > 0 else 'steelblue' for c in coef]
plt.barh(feature_names, coef, color = colors)
plt.axvline(0, color = 'black', linewidth = 0.8)
plt.xlabel('Coefficient (계수)')
plt.title('Logistic Regression 변수 중요도')
plt.tight_layout()
plt.show()
```

<img src = "/assets/img/ML/lr/lr_coef.png" width = "70%" alt = "lr_coef">

### VI. Evaluation Score

```python
pred      = LR.predict(X_test)
pred_prob = LR.predict_proba(X_test)[:, 1]  # 생존 확률

cfx         = confusion_matrix(y_test, pred)
sensitivity = cfx[1, 1] / (cfx[1, 0] + cfx[1, 1])
specificity = cfx[0, 0] / (cfx[0, 0] + cfx[0, 1])

print(f"정확도 : {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"민감도 : {sensitivity * 100:.2f}%")
print(f"특이도 : {specificity * 100:.2f}%")
print()
print(classification_report(y_test, pred, target_names = ['사망(0)', '생존(1)']))
```

```
정확도 : 81.59%
민감도 : 72.22%
특이도 : 87.25%

              precision    recall  f1-score   support

       사망(0)       0.84      0.87      0.86       149
       생존(1)       0.77      0.72      0.74        90

    accuracy                           0.82       239
   macro avg       0.81      0.80      0.80       239
weighted avg       0.81      0.82      0.81       239
```

---

## 장단점

**장점**

- 구현이 단순하고 학습이 빠름
- 확률 값을 직접 반환 → 의사결정 임계값 조정 가능
- 계수(coefficient)로 변수 영향력 해석 가능
- 정규화 내장으로 과적합 제어 용이

**단점**

- 선형 결정 경계 → 비선형 문제에 한계
- 변수 간 독립성 가정 → 다중공선성에 취약
- 이상치에 비교적 민감
- 복잡한 Feature 관계 표현 불가

---

## 요약

| 항목                    | 내용                                       |
| ----------------------- | ------------------------------------------ |
| **알고리즘 유형**       | 선형 분류 (이진 / 다중 클래스)             |
| **핵심 아이디어**       | 선형 결합 → 시그모이드 → 확률 → 클래스     |
| **손실 함수**           | Binary Cross-Entropy                       |
| **필수 전처리**         | 스케일링 (수렴 속도 및 정규화 효과 개선)   |
| **핵심 하이퍼파라미터** | `C` (정규화 강도), `penalty` (정규화 방식) |
| **최적 C 선택**         | GridSearchCV 또는 LogisticRegressionCV     |
| **주요 강점**           | 확률 해석 가능, 빠른 학습, 계수 해석       |
| **주요 약점**           | 비선형 문제, 다중공선성                    |
| **적합한 상황**         | 이진 분류, 확률 필요, 빠른 베이스라인      |