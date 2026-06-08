---
title : "[Python] ML-XGBoost"
categories:
    - ML
date: 2026-06-08
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---

## 1. 왜 등장했는가

기존 Gradient Boosting은 정확하지만 느리고, 과적합 제어 수단이 부족했습니다.  
XGBoost는 **2차 미분(헤시안)을 이용한 정밀한 분기 탐색**과 다양한 정규화 기법으로  
속도·성능·과적합 제어를 동시에 개선해 캐글 대회를 석권했습니다. (Chen & Guestrin, 2016)

Gradient Boosting이 "방향만 알고 내려가는 등산객"이라면,  
XGBoost는 **"방향과 경사의 곡률까지 파악해 보폭을 조절하는 영리한 등산객"** 입니다.

---

## 2. 핵심 아이디어 — Gradient Boosting의 수학적 완성판

XGBoost는 본질적으로 **Gradient Boosting에 정밀도와 정규화를 더한** 알고리즘입니다.

<img src = "/assets/img/ML/xgb/xgb_idea.png" class = "mx-auto d-block" width = "100%" alt = "xgb_idea">

```
Gradient Boosting:                XGBoost:
1차 미분(기울기)만 사용            1차 + 2차 미분 사용
                                   → 더 정확한 분기 방향과 크기

정규화 없음                        L1, L2 정규화 + 트리 복잡도 패널티
                                   → 과적합 제어 수단 풍부

결측값 처리 없음                    결측값 자동 처리
                                   → 분기 방향 학습

컬럼 샘플링 없음                    subsample + colsample_bytree
                                   → 더 다양한 트리, 빠른 학습
```

---

## 3. 실제 예시로 보기 (분류 / 회귀)

### 예시 1 — 타이타닉 생존 예측 (분류)

```
훈련 데이터:
┌────────┬────────┬─────┬──────────┬──────────┐
│ 이름   │ 성별   │ 나이 │ 객실 등급 │ 생존 여부 │
├────────┼────────┼─────┼──────────┼──────────┤
│ Alice  │ 여성   │  29  │   1등석   │    ✅    │
│ Bob    │ 남성   │  31  │   3등석   │    ❌    │
│ Carol  │ 여성   │  45  │   3등석   │    ✅    │
│ Dave   │ 남성   │  12  │   2등석   │    ✅    │
│ Eve    │ 남성   │  38  │   3등석   │    ❌    │
└────────┴────────┴─────┴──────────┴──────────┘

Round 1: 초기 예측 = log(생존수/사망수) = log(3/2) ≈ 0.405
  → 1차 미분 gᵢ + 2차 미분 hᵢ 계산
  → 최적 분기: "Sex_male ≤ 0.5"
  → 리프값: -Σg / (Σh + λ)   ← λ는 L2 정규화

Round 2: 업데이트된 예측으로 새 gᵢ, hᵢ 계산
  → 다음 트리 학습

...n_estimators번 반복...
```

---

### 예시 2 — 2차 미분의 의미

```
언덕 내려가기 비유:

Gradient Boosting (1차만):
  "아래로 내려가라" (방향만)
  → 보폭을 고정 → 최솟값에 정확히 멈추기 어려움

XGBoost (1차 + 2차):
  "아래로 내려가되, 경사 완만해지면 보폭 줄여라" (방향 + 곡률)
  → 보폭도 최적화 → 더 정확하게 최솟값에 도달
```

---

## 4. 알고리즘 구성 요소

<img src = "/assets/img/ML/xgb/xgb_component.png" class = "mx-auto d-block" width = "100%" alt = "xgb_component">

| 구성 요소 | 설명 | 비유 |
|-----------|------|------|
| **gᵢ (1차 미분)** | 손실 함수의 기울기 | 어느 방향으로 틀렸나 |
| **hᵢ (2차 미분)** | 손실 함수의 곡률 | 얼마나 확실하게 틀렸나 |
| **Gain** | 분기로 얻는 손실 감소량 | 이 질문이 얼마나 도움이 되나 |
| **λ (lambda)** | L2 정규화 | 리프값에 세금 |
| **γ (gamma)** | 분기 최소 Gain 임계값 | 의미 있는 분기만 허용 |

1차와 2차 미분이 분기 탐색에서 어떻게 활용되는지 아래 그래프에서 확인할 수 있습니다.

<img src = "/assets/img/ML/xgb/xgb_gradient_hessian.png" class = "mx-auto d-block" width = "100%" alt = "xgb_gradient_hessian">

> **읽는 법:**  
> 왼쪽(1차 미분 / Gradient): 손실 함수의 기울기. 어느 방향으로 예측을 수정해야 하는지를 나타냅니다.  
> 오른쪽(2차 미분 / Hessian): 손실 함수의 곡률. 기울기 변화 속도를 나타내며, 보폭을 얼마나 크게 할지 결정합니다.  
> 두 값을 함께 사용하면 Gradient만 쓸 때보다 더 정확하게 최솟값에 도달할 수 있습니다.

---

## 5. XGBoost가 Gradient Boosting보다 나은 이유

### 5-1. 정규화 세 가지

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_j w_j^2 + \alpha\sum_j |w_j|$$

```
γ (gamma): 리프 수 패널티 → 트리를 단순하게 유지
λ (lambda): 리프값 L2 패널티 → 리프값을 작게 유지
α (alpha): 리프값 L1 패널티 → 일부 리프값을 0으로
```

---

### 5-2. Gain이 γ보다 작으면 분기 안 함

```
Gain < γ:  "이 분기로 얻는 이득이 너무 적다" → 분기 취소
           → 불필요한 가지 자동 제거 (사후 가지치기)

γ=0: 모든 분기 허용
γ=1: 손실 감소 < 1인 분기 차단
```

γ 값에 따라 트리의 복잡도가 어떻게 달라지는지 아래 그래프에서 확인할 수 있습니다.

<img src = "/assets/img/ML/xgb/xgb_gamma_complexity.png" class = "mx-auto d-block" width = "100%" alt = "xgb_gamma_complexity">

> **읽는 법:**  
> 왼쪽(γ=0): 모든 분기를 허용해 트리가 복잡해집니다. 훈련 성능은 높지만 과적합 위험이 있습니다.  
> 오른쪽(γ 증가): Gain이 γ보다 작은 분기가 차단되어 트리가 단순해집니다.  
> 최적 γ는 GridSearchCV로 탐색하지만, 일반적으로 0~1 사이의 작은 값부터 시작합니다.

---

## 6. XGBoost 장・단점

### 6-1. ✅ XGBoost 장점

```
1. 높은 예측 성능
   → 2차 미분 + 정규화 → 정확하고 안정적

2. 풍부한 정규화
   → γ, λ, α, subsample, colsample_bytree 등 다양한 과적합 제어

3. 결측값 자동 처리
   → 분기 방향 자동 학습 → 별도 imputation 불필요

4. 조기 종료 (Early Stopping)
   → 검증 성능 개선 없으면 자동 중단
```

### 6-2. ❌ XGBoost가 약한 상황

```
1. 대용량 데이터
   → LightGBM이 더 빠름 (히스토그램 방식)

2. 하이퍼파라미터 많음
   → n_estimators, learning_rate, max_depth,
      min_child_weight, subsample, colsample_bytree,
      gamma, lambda, alpha 동시 조정 필요

3. 범주형 특성
   → 인코딩 필요 (LightGBM은 직접 처리 가능)
```

#### 6-2-1. 과적합 문제

XGBoost의 주요 주의사항입니다.

훈련 AUC: 0.99, 검증 AUC: 0.82 ← 과적합 징후

**해결책 :**

```python
# 방법 1: 정규화 강화
xgb = XGBClassifier(
    gamma=1,         # 분기 최소 Gain (클수록 단순한 트리)
    reg_alpha=0.1,   # L1
    reg_lambda=1.0,  # L2
    random_state=0
)

# 방법 2: 샘플링
xgb = XGBClassifier(
    subsample=0.8,          # 행 샘플링
    colsample_bytree=0.8,   # 열 샘플링
    random_state=0
)

# 방법 3: GridSearchCV로 최적값 탐색
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':     [100, 200, 300],
    'learning_rate':    [0.01, 0.05, 0.1],
    'max_depth':        [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_cv = GridSearchCV(
    XGBClassifier(random_state=0, eval_metric='auc'),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
xgb_cv.fit(X_train, y_train)
print(f"Best params : {xgb_cv.best_params_}")
print(f"Best AUC    : {xgb_cv.best_score_:.4f}")
```

---

## 7. 한눈에 요약

| 항목 | 내용 |
| ---- | ---- |
| **알고리즘 유형** | 지도학습 / 분류 & 회귀 모두 가능 |
| **핵심 아이디어** | 2차 미분 + 다양한 정규화로 개선된 Gradient Boosting |
| **기반 모델** | 얕은 Decision Tree |
| **스케일링 필요?** | ❌ 불필요 |
| **결측값** | ✅ 자동 처리 |
| **핵심 파라미터** | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| **실전 사용** | 정확도 최우선, 캐글 대회, 중소규모 정형 데이터 |

---

## 8. 다른 알고리즘과 무엇이 다른가

### Gradient Boosting vs XGBoost vs LightGBM

| 항목 | GradientBoosting | XGBoost | LightGBM |
|------|-----------------|---------|----------|
| 미분 | 1차 | 1차+2차 | 1차+2차 |
| 트리 성장 | Level-wise | Level-wise | Leaf-wise |
| 속도 | 느림 | 보통 | 빠름 |
| 정규화 | 제한적 | 풍부 | 풍부 |
| 결측값 | ❌ | ✅ 자동 | ✅ 자동 |
| 범주형 | ❌ | ❌ | ✅ 직접 |
| 소규모 데이터 | ▲ | ✅ | ⚠️ |
| 대규모 데이터 | ❌ | ▲ | ✅ |

세 알고리즘의 포지션을 한마디로 정리하면:  
**Gradient Boosting** → 원리 이해용 교과서  
**XGBoost** → 중소규모 정형 데이터의 표준  
**LightGBM** → 대용량 데이터의 속도왕

---

## 9. 코드로 보기 — 타이타닉 생존 예측

```python
from xgboost import XGBClassifier
```

---

#### 9-1. 전처리 (짧게)

```python
import pandas as pd
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('./Data/Titanic.csv')
titanic['FamSize'] = titanic['SibSp'] + titanic['Parch']

use_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked']
titanic = titanic[use_cols].dropna(subset=['Age'])
titanic['Age'] = titanic['Age'].astype(int)
titanic = pd.get_dummies(titanic, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

y = titanic['Survived']
X = titanic.drop('Survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# ⚠️ XGBoost는 트리 기반 → 스케일링 불필요
# StandardScaler 생략
```

> **Note:** XGBoost는 트리 기반이므로 `StandardScaler`가 필요 없습니다. 결측값도 자동으로 처리합니다.

---

#### 9-2. 모델 학습

```python
xgb = XGBClassifier(
    n_estimators=300,         # 트리 수
    learning_rate=0.05,       # 각 트리 기여도
    max_depth=4,              # 트리 최대 깊이
    min_child_weight=3,       # 리프의 최소 가중치 합 (과적합 방지)
    subsample=0.8,            # 행 샘플링 비율
    colsample_bytree=0.8,     # 열 샘플링 비율
    gamma=0.1,                # 분기 최소 Gain
    reg_alpha=0.1,            # L1 정규화
    reg_lambda=1.0,           # L2 정규화
    eval_metric='auc',
    random_state=0
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=100
)
```

---

| Parameter | Default | 역할 | 과적합 방향 |
|-----------|---------|------|------------|
| `n_estimators` | `100` | 트리 수 | 클수록 과적합 ↑ |
| `learning_rate` | `0.3` | 트리 기여도 | 클수록 과적합 ↑ |
| `max_depth` | `6` | 트리 깊이 | 클수록 과적합 ↑ |
| `min_child_weight` | `1` | 리프 최소 가중치 | 작을수록 과적합 ↑ |
| `subsample` | `1.0` | 행 샘플링 | 작을수록 과적합 ↓ |
| `colsample_bytree` | `1.0` | 열 샘플링 | 작을수록 과적합 ↓ |
| `gamma` | `0` | 분기 최소 Gain | 클수록 과적합 ↓ |
| `reg_alpha` | `0` | L1 정규화 | 클수록 과적합 ↓ |
| `reg_lambda` | `1` | L2 정규화 | 클수록 과적합 ↓ |

- `n_estimators` : 트리를 몇 개 쌓을지
	- 값 변화별 효과
		- 클수록 → 더 정교한 학습, 과적합 ↑
		- Early Stopping과 함께 사용 권장
- `learning_rate` : 각 트리의 기여도 축소 계수
	- 값 변화별 효과
		- 작을수록 → 보수적 학습, `n_estimators` 늘려야 함
		- 큰 값(기본 0.3)은 빠른 수렴이지만 과적합 위험
- `max_depth` : 개별 트리의 최대 깊이
	- 값 변화별 효과
		- 기본값 6은 과적합 위험 → 3~5 권장
		- 클수록 → 복잡한 상호작용, 과적합 ↑
- `min_child_weight` : 리프 노드의 최소 가중치 합
	- 값 변화별 효과
		- 클수록 → 분기 덜 함 → 과적합 ↓
		- 작을수록 → 계속 분기 → 과적합 ↑
- `subsample` & `colsample_bytree` : 행/열 샘플링
	- 값 변화별 효과
		- 0.8 설정 → 과적합 방지 + 학습 다양성 ↑
		- 1.0 → 전체 사용
- `gamma` : 분기 최소 Gain 임계값
	- 값 변화별 효과
		- 클수록 → 의미 없는 분기 차단 → 단순한 트리
		- 0 → 모든 분기 허용

---

#### 9-3. 평가

```python
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

pred      = xgb.predict(X_test)
pred_prob = xgb.predict_proba(X_test)[:, 1]

cfx         = confusion_matrix(y_test, pred)
sensitivity = cfx[1, 1] / (cfx[1, 0] + cfx[1, 1])
specificity = cfx[0, 0] / (cfx[0, 0] + cfx[0, 1])
roc_auc     = roc_auc_score(y_test, pred_prob)

print(f"Accuracy    : {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"Sensitivity : {sensitivity * 100:.2f}%")
print(f"Specificity : {specificity * 100:.2f}%")
print(f"ROC AUC     : {roc_auc:.4f}")
print()
print(classification_report(y_test, pred, target_names=['Died (0)', 'Survived (1)']))
```

---

#### 9-4. Early Stopping 결과 확인

```python
import matplotlib.pyplot as plt

results = xgb.evals_result()

plt.figure(figsize=(8, 4))
plt.plot(results['validation_0']['auc'], color='tomato', label='Validation AUC')
plt.axvline(xgb.best_iteration, color='black', linestyle='--',
            label=f'Best iteration: {xgb.best_iteration}')
plt.xlabel('Boosting Round')
plt.ylabel('AUC')
plt.title('XGBoost Early Stopping')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

#### 9-5. 특성 중요도

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(xgb.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(7, 5))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()
```
