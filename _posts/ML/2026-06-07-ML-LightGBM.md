---
title : "[Python] ML-LightGBM"
categories:
    - ML
date: 2026-06-07
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---

## 1. 왜 등장했는가

Gradient Boosting은 강력하지만, 모든 특성의 모든 분기점을 탐색하므로 대용량 데이터에서 매우 느렸습니다.  
LightGBM은 **히스토그램 기반 분기 탐색**과 **리프 우선(Leaf-wise) 성장**으로  
기존 대비 10~100배 빠른 속도와 낮은 메모리 사용량을 달성했습니다. (Microsoft, 2017)

Gradient Boosting이 "모든 분기점을 꼼꼼히 탐색하는 학생"이라면,  
LightGBM은 **"구간별로 묶어서 빠르게 탐색하면서도 정확도는 유지하는 영리한 학생"** 입니다.

---

## 2. 핵심 아이디어 — 빠르고 정확한 Gradient Boosting

LightGBM은 본질적으로 **Gradient Boosting을 대용량 데이터에 맞게 최적화**한 알고리즘입니다.

<img src = "/assets/img/ML/lgb/lgb_idea.png" class = "mx-auto d-block" width = "100%" alt = "lgb_idea">

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

히스토그램 변환:
  나이  → 구간: [0~15], [15~30], [30~45], [45~60]
  운임  → 구간: [0~10], [10~50], [50~100], [100+]

Leaf-wise 성장:
  Round 1: 전체 분기 후 손실 감소가 가장 큰 리프 선택
  Round 2: 선택된 리프만 추가 분기
  → 불균형 트리지만 더 낮은 손실 달성
```

---

### 예시 2 — 대용량 클릭률 예측 (회귀)

```
데이터: 1억 행 × 200개 특성

기존 Gradient Boosting: 수십 시간
LightGBM:              수십 분

→ 대용량 데이터에서 사실상 유일한 실용적 선택지
```

---

## 4. 알고리즘 구성 요소

<img src = "/assets/img/ML/lgb/lgb_component.png" class = "mx-auto d-block" width = "100%" alt = "lgb_component">

| 구성 요소 | 설명 | 비유 |
|-----------|------|------|
| **히스토그램** | 연속 특성을 구간으로 나눠 메모리·속도 절약 | 성적을 1점 단위 대신 A/B/C로 분류 |
| **Leaf-wise** | 손실 최대 감소 리프만 성장 | 가장 효과 있는 공부부터 |
| **GOSS** | 중요 샘플만 추려서 학습 (행 샘플링 개선) | 어려운 문제에 집중 |
| **EFB** | 희소 특성을 묶어서 처리 (열 압축) | 비슷한 과목 합쳐서 공부 |

---

## 5. LightGBM의 핵심 기술

### 5-1. 히스토그램 기반 분기

```
기존:      Age 값 정렬 → 모든 분기점 탐색 → O(N × features)
LightGBM:  Age → 256개 구간 → 256번만 탐색 → O(bins × features)

bins=256 고정 → 데이터 크기와 무관한 탐색 비용
```

데이터가 100만 건이든 1억 건이든 탐색 비용이 동일하다는 것이 핵심입니다.  
아래 그래프는 히스토그램 압축 방식을 시각화합니다.

<img src = "/assets/img/ML/lgb/lgb_histogram.png" class = "mx-auto d-block" width = "100%" alt = "lgbm_histogram">

> **읽는 법:**  
> 왼쪽 — 기존 방식: 각 연속값마다 분기 가능성을 모두 탐색합니다. 데이터가 많을수록 탐색 횟수가 급증합니다.  
> 오른쪽 — 히스토그램 방식: 연속값을 구간(bin)으로 압축하면 탐색 횟수가 구간 수(256)로 고정됩니다.  
> 정밀도 손실이 거의 없으면서 속도가 획기적으로 향상됩니다.

---

### 5-2. Leaf-wise vs Level-wise

```
Level-wise (기존 GB):     Leaf-wise (LightGBM):
깊이 1:   ○ ○             깊이 1:       ○
깊이 2: ○ ○ ○ ○           깊이 2:     ○   ○
깊이 3: ○○○○○○○○          깊이 3: ○ ○

균형 트리, 안정적           불균형 트리, 더 정확
과적합 덜함                 과적합 주의 (num_leaves 제한 필요)
```

Level-wise와 Leaf-wise의 트리 성장 차이를 아래 그래프에서 확인할 수 있습니다.

<img src = "/assets/img/ML/lgb/lgb_leaf_vs_level.png" class = "mx-auto d-block" width = "100%" alt = "lgbm_leaf_vs_level">

> **읽는 법:**  
> 왼쪽(Level-wise): 트리가 균형 잡힌 모양으로 성장하며 모든 리프가 같은 깊이입니다. 안정적이지만 최적 분기가 아닐 수 있습니다.  
> 오른쪽(Leaf-wise): 손실 감소가 가장 큰 리프를 선택적으로 분기합니다. 불균형하지만 더 낮은 손실을 달성합니다.  
> `num_leaves`가 너무 크면 과적합이 발생하므로 주의가 필요합니다.

---

## 6. LightGBM 장・단점

### 6-1. ✅ LightGBM 장점

```
1. 매우 빠른 학습 속도
   → 히스토그램 + 병렬 처리 → 기존 GB 대비 10~100배

2. 낮은 메모리 사용
   → 히스토그램 압축 → 대용량 데이터 처리 가능

3. 높은 정확도
   → Leaf-wise 성장 → 손실 감소 최적화

4. 범주형 특성 직접 처리
   → One-hot 인코딩 없이도 학습 가능
```

### 6-2. ❌ LightGBM이 약한 상황

```
1. 소규모 데이터
   → Leaf-wise 특성상 과적합 위험 → XGBoost 또는 단순 모델 고려

2. 하이퍼파라미터에 민감
   → num_leaves, min_data_in_leaf 신중하게 조정 필요

3. 해석 어려움
   → 수백 개 트리의 앙상블 → 블랙박스
```

#### 6-2-1. Leaf-wise 과적합 문제

LightGBM의 주요 주의사항입니다.

`num_leaves=1000`, `min_data_in_leaf=1`일 때:  
훈련 AUC → 1.0에 수렴  ← 완전 과적합  
테스트 AUC → 낮음

**해결책 :**

```python
# 방법 1: num_leaves 제한 (가장 중요)
lgbm = LGBMClassifier(num_leaves=31, random_state=0)

# 방법 2: min_child_samples 늘리기
lgbm = LGBMClassifier(min_child_samples=20, random_state=0)

# 방법 3: 정규화 파라미터 추가
lgbm = LGBMClassifier(reg_alpha=0.1, reg_lambda=0.1, random_state=0)

# 방법 4: GridSearchCV로 최적값 탐색
from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_leaves':        [15, 31, 63],
    'learning_rate':     [0.01, 0.05, 0.1],
    'n_estimators':      [100, 200, 300],
    'min_child_samples': [10, 20, 30]
}
lgb_cv = GridSearchCV(
    LGBMClassifier(random_state=0, verbose=-1),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
lgb_cv.fit(X_train, y_train)
print(f"Best params : {lgb_cv.best_params_}")
print(f"Best AUC    : {lgb_cv.best_score_:.4f}")
```

---

## 7. 한눈에 요약

| 항목 | 내용 |
| ---- | ---- |
| **알고리즘 유형** | 지도학습 / 분류 & 회귀 모두 가능 |
| **핵심 아이디어** | Gradient Boosting + 히스토그램 + Leaf-wise 성장 |
| **스케일링 필요?** | ❌ 불필요 |
| **속도** | ✅ 매우 빠름 |
| **핵심 파라미터** | `num_leaves`, `learning_rate`, `n_estimators` |
| **실전 사용** | 대용량 데이터, 캐글 대회, 정확도 우선 |

---

## 8. 다른 알고리즘과 무엇이 다른가

### XGBoost vs LightGBM

```
XGBoost:                          LightGBM:
Level-wise 트리 성장               Leaf-wise 트리 성장
정확한 분기점 탐색                  히스토그램 근사 탐색
느림, 안정적                        빠름, num_leaves 주의
소~중규모 데이터                     대규모 데이터
```

| 항목 | XGBoost | LightGBM |
|------|---------|----------|
| 속도 | 보통 | 빠름 |
| 메모리 | 보통 | 낮음 |
| 과적합 제어 | 안정적 | num_leaves 주의 |
| 범주형 처리 | 인코딩 필요 | 직접 처리 가능 |
| 소규모 데이터 | ✅ | ⚠️ |
| 대규모 데이터 | ⚠️ | ✅ |

---

## 9. 코드로 보기 — 타이타닉 생존 예측

```python
from lightgbm import LGBMClassifier
import lightgbm as lgb

lgbm = LGBMClassifier(
    n_estimators = 200,       # 트리 개수
	learning_rate = 0.05,     # 학습률
	num_leaves = 31,          # 트리당 최대 리프 수
	min_child_samples = 20,   # Leaf 노드의 최소 샘플 수
	subsample = 1.0,          # 행 샘플링 비율
	colsample_bytree = 1.0,   # 변수 샘플링 비율
	random_state = 0,        
	verbose = -1
)
```
  
| Parameter           | 설명                | Default |
| ------------------- | ----------------- | ------- |
| `n_estimators`      | 트리 수              | 100     |
| `learning_rate`     | 학습률               | 0.1     |
| `num_leaves`        | 트리당 최대 리프 수 (핵심)  | 31      |
| `max_depth`         | 최대 깊이 (-1: 제한 없음) | -1      |
| `min_child_samples` | Leaf 노드의 최소 샘플 수  | 20      |
| `subsample`         | 행 샘플링 비율          | 1.0     |
| `colsample_bytree`  | 변수 샘플링 비율         | 1.0     |

- `num_leaves` : 트리당 최대 Leaf 수 
	- 값 변화별 효과
		- 클수록 → 복잡한 모델, 과적합 ↑, 학습 시간 증가
		- 작을수록 → 단순한 모델, 과적합 ↓
	- `2^max_depth`보다 작게 설정 권장
	- 소규모 데이터: 31 이하, 대규모 데이터: 64~128 시도
- `learning_rate` : 각 트리의 기여도 축소 계수
	- 값 변화별 효과
		- 클수록 → 빠른 수렴, 과적합 위험 ↑
		- 작을수록 → 안정적인 수렴, `n_estimators` 더 필요
	- 실무 권장: `0.05~0.1` + Early Stopping 조합
- `min_child_samples` : Leaf 노드에 필요한 최소 샘플 수
	- 값 변화별 효과
		- 클수록 → 분할 제한 → 과적합 ↓
		- 작을수록 → 잦은 분할 → 과적합 ↑
	- 소규모 데이터에서 과적합 방지 시 20~50으로 증가
- `subsample` / `colsample_bytree` : 행/열 샘플링 비율
	- 값 변화별 효과
		- `1.0`보다 작으면 → 무작위성 추가 → 과적합 ↓, 학습 빠름
		- 실무 권장: `0.7~0.9`
        
---

### 9-1. 전처리 (짧게)

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

# ⚠️ LightGBM은 트리 기반 → 스케일링 불필요
# StandardScaler 생략
```

> **Note:** LightGBM은 트리 기반이므로 특성 스케일에 영향을 받지 않습니다. `StandardScaler`가 필요 없습니다.

---

### 9-2. 모델 학습

```python
lgbm = LGBMClassifier(
    n_estimators = 500,        # 트리 수
    learning_rate = 0.05,      # 각 트리 기여도
    num_leaves = 31,           # 트리당 최대 리프 수 (핵심 파라미터)
    min_child_samples = 20,    # 리프 최소 샘플 수 (과적합 방지)
    reg_alpha = 0.1,           # L1 정규화
    reg_lambda = 0.1,          # L2 정규화
    random_state = 0,
    verbose = -1
)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

---

- `num_leaves` : LightGBM에서 가장 중요한 파라미터
	- 값 변화별 효과
		- 클수록 → 복잡한 모델, 과적합 ↑
		- 작을수록 → 단순한 모델, 과소적합 위험
	- `2^max_depth`보다 작게 설정 권장 (Leaf-wise 과적합 제어)
- `min_child_samples` : 리프 노드에 있어야 하는 최소 샘플 수
	- 값 변화별 효과
		- 클수록 → 분기 덜 함 → 과적합 ↓
		- 작을수록 → 계속 분기 → 과적합 ↑
- `learning_rate` & `n_estimators` : 트레이드오프 관계
	- `learning_rate` 낮추면 → `n_estimators` 늘려야 같은 성능
	- Early Stopping 사용 권장

---

### 9-3. 평가

```python
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

pred      = lgbm.predict(X_test)
pred_prob = lgbm.predict_proba(X_test)[:, 1]

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

### 9-4. 특성 중요도

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(lgbm.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(7, 5))
importances.plot(kind='barh', color='steelblue')
plt.xlabel('Feature Importance')
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.show()
```
