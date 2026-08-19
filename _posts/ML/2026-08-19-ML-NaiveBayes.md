---
title : "[Python] ML-Naive Bayes"
categories:
    - ML
date: 2026-08-19
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---

---
## 1. 왜 등장했는가

텍스트 분류처럼 특성이 수천 개에 달하는 경우, 대부분의 모델은 학습이 느리거나 차원의 저주에 빠집니다.  
Naive Bayes는 **베이즈 정리**를 기반으로, 특성들이 서로 독립이라는 단순한 가정 하나로  
계산을 극도로 단순화해 **빠르고 효율적인 확률적 분류기**를 만듭니다. (1950년대~)

---

## 2. 핵심 아이디어 — "각 특성이 독립적으로 증거를 제공한다"

Naive Bayes는 본질적으로 **의사가 증상별로 각각 독립적으로 진단하는** 방식입니다.

```
스팸 메일 분류 예시:

메일: "무료 당첨 클릭 지금"

각 단어가 독립적으로 스팸 증거를 제공:
  P(스팸 | "무료") = 0.95  ← 강한 스팸 신호
  P(스팸 | "당첨") = 0.90  ← 강한 스팸 신호
  P(스팸 | "클릭") = 0.70  ← 보통 스팸 신호
  P(스팸 | "지금") = 0.55  ← 약한 스팸 신호

최종: P(스팸) × 0.95 × 0.90 × 0.70 × 0.55  vs  P(정상) × ...

→ 스팸 확률이 더 높으면 스팸으로 분류
```

"Naive(순진한)"라고 불리는 이유는 **모든 특성이 서로 독립**이라고 가정하기 때문입니다.  
실제로는 독립이 아닌 경우가 많지만, 이 단순한 가정으로도 실전에서 놀라운 성능을 냅니다.

---

## 3. 실제 예시로 보기 (분류)

### 예시 1 — 스팸 메일 분류 (분류)

```
훈련 데이터:
┌──────────────────────┬──────────┐
│ 메일 내용            │ 레이블   │
├──────────────────────┼──────────┤
│ "무료 쿠폰 클릭"      │  스팸    │
│ "회의 일정 확인"      │  정상    │
│ "당첨 무료 지금"      │  스팸    │
│ "프로젝트 보고서"     │  정상    │
└──────────────────────┴──────────┘

P(스팸) = 2/4 = 0.5
P(정상) = 2/4 = 0.5

새 메일: "무료 회의"

P(스팸 | "무료", "회의") ∝ P(스팸) × P("무료"|스팸) × P("회의"|스팸)
                         = 0.5 × 2/6 × 0/6
                         = 0  ← 0이 되는 문제 → Laplace Smoothing 필요

P(정상 | "무료", "회의") ∝ P(정상) × P("무료"|정상) × P("회의"|정상)
                         = 0.5 × 0/6 × 1/5
                         = 0  ← 동일한 문제

Laplace Smoothing 후:
P("무료"|스팸) = (2+1)/(6+V) → 0이 되지 않도록 보정
```

---

### 예시 2 — 타이타닉 생존 예측 (분류)

```
P(생존 | 여성, 1등석, 나이=29) ∝
  P(생존) × P(여성|생존) × P(1등석|생존) × P(나이≈29|생존)

P(사망 | 여성, 1등석, 나이=29) ∝
  P(사망) × P(여성|사망) × P(1등석|사망) × P(나이≈29|사망)

→ 더 큰 확률의 클래스로 예측
```

---

## 4. 알고리즘 구성 요소

```
     ┌────────────────────────────────────────┐
     │   사전 확률 P(클래스) 계산              │  ← 훈련 데이터 클래스 비율
     └──────────────────┬─────────────────────┘
                        │
                        ▼
     ┌────────────────────────────────────────┐
     │   조건부 확률 P(특성|클래스) 계산       │  ← 특성 분포 추정
     └──────────────────┬─────────────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
    [범주형/텍스트]          [연속형]
   MultinomialNB           GaussianNB
   BernoulliNB             (정규분포 가정)
                        │
                        ▼
     ┌────────────────────────────────────────┐
     │   베이즈 정리로 사후 확률 계산          │
     │   P(클래스|특성) ∝ P(클래스)×ΠP(xᵢ|클래스)│
     └──────────────────┬─────────────────────┘
                        │
                        ▼
     ┌────────────────────────────────────────┐
     │   가장 높은 확률의 클래스 예측          │
     └────────────────────────────────────────┘
```

| 구성 요소 | 설명 | 비유 |
|-----------|------|------|
| **사전 확률 P(C)** | 클래스별 비율 | 전체 환자 중 질병 A 비율 |
| **조건부 확률 P(x\|C)** | 해당 클래스에서 특성값이 나올 확률 | 질병 A 환자 중 발열 비율 |
| **독립 가정** | 특성들이 서로 무관하다는 가정 | 증상들이 서로 연관 없다는 가정 |
| **Laplace Smoothing** | 0 확률 방지를 위한 보정 | 한 번도 없는 증상도 0% 아님 |

---

## 5. 어떻게 확률을 계산하는가

### 5-1. 베이즈 정리

$$P(C_k | x_1, ..., x_n) \propto P(C_k) \prod_{i=1}^n P(x_i | C_k)$$

```
P(클래스 | 특성들) ∝ P(클래스) × P(특성1|클래스) × P(특성2|클래스) × ...

독립 가정 덕분에 곱셈으로 단순화
→ 특성이 1000개여도 1000번의 곱셈만 필요
```

---

### 5-2. Naive Bayes 세 가지 종류

| 종류 | 적합한 데이터 | 확률 분포 가정 |
|------|-------------|-------------|
| **GaussianNB** | 연속형 수치 데이터 | 정규분포 |
| **MultinomialNB** | 텍스트 (단어 빈도) | 다항분포 |
| **BernoulliNB** | 이진 특성 (0/1) | 베르누이 분포 |

```
GaussianNB: 나이, 키, 체중 같은 연속값
  P(나이=29 | 생존) = N(μ=28.3, σ=14.2)에서의 확률 밀도

MultinomialNB: 텍스트 단어 빈도
  P("무료"=3번 | 스팸)  = 다항분포에서의 확률

BernoulliNB: 단어 존재 여부 (0/1)
  P("무료"=포함 | 스팸) = 베르누이 분포에서의 확률
```

---

### 5-3. Laplace Smoothing

$$P(x_i | C_k) = \frac{N_{ik} + \alpha}{N_k + \alpha \cdot V}$$

```
문제: 훈련 데이터에 "환불" 단어가 스팸에 0번 등장
  P("환불"|스팸) = 0/N = 0
  → 전체 곱이 0이 돼버림 (모든 확률이 0)

Laplace Smoothing (α=1):
  P("환불"|스팸) = (0+1)/(N+V)  ← 절대 0이 되지 않음
```

---

## 6. Naive Bayes 장・단점

### 6-1. ✅ Naive Bayes 장점

```
1. 매우 빠른 학습과 예측
   → 한 번 통계 계산으로 완성 → 대용량 텍스트에 이상적

2. 적은 데이터로도 작동
   → 파라미터가 적어 작은 훈련 데이터에서도 안정적

3. 고차원 데이터에 강함
   → 텍스트 분류에서 특성이 수만 개여도 효율적

4. 확률 출력
   → 클래스 소속 확률 제공
```

### 6-2. ❌ Naive Bayes가 약한 상황

```
1. 특성 간 상관관계 있을 때
   → 독립 가정 위반 → 확률 추정 왜곡
   → 예: "무료"와 "쿠폰"은 함께 등장하는 경향

2. 연속형 데이터에서 정규분포 가정 위반
   → GaussianNB는 정규분포 가정 → 비대칭 분포에 약함

3. 확률 수치 자체는 부정확
   → 클래스 순위는 맞아도 절대 확률값은 신뢰하기 어려움
```

#### 6-2-1. 독립 가정 위반 문제

Naive Bayes의 가장 큰 약점입니다.

```
텍스트: "New York Times"

독립 가정:
  P("New"|스팸) × P("York"|스팸) × P("Times"|스팸)
  → "New"와 "York"이 독립이 아님에도 독립으로 처리

실제: "New York"은 하나의 단위로 봐야 함
→ N-gram 특성이나 다른 모델 고려
```

**해결책 :**

```python
# 방법 1: var_smoothing 조정 (GaussianNB)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(var_smoothing=1e-9)

# 방법 2: alpha 조정 (MultinomialNB)
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]}
nb_cv = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='roc_auc'
)
nb_cv.fit(X_train, y_train)
print(f"Best alpha : {nb_cv.best_params_['alpha']}")
print(f"Best AUC   : {nb_cv.best_score_:.4f}")
```

---

## 7. 한눈에 요약

| 항목 | 내용 |
|------|------|
| **알고리즘 유형** | 지도학습 / 분류 |
| **핵심 아이디어** | 베이즈 정리 + 특성 독립 가정 → 빠른 확률 분류 |
| **핵심 가정** | 특성들이 서로 독립 (Naive 가정) |
| **스케일링 필요?** | ❌ 불필요 (GaussianNB) / ✅ 음수 없어야 함 (MultinomialNB) |
| **핵심 파라미터** | `alpha` (Laplace Smoothing), `var_smoothing` |
| **실전 사용** | 텍스트 분류, 스팸 필터, 실시간 분류 시스템 |

---

## 8. 다른 알고리즘과 무엇이 다른가

#### Logistic Regression vs Naive Bayes

```
Logistic Regression:                 Naive Bayes:
판별 모델 (Discriminative)            생성 모델 (Generative)
P(클래스 | 특성)을 직접 학습           P(특성 | 클래스)를 학습
→ 결정 경계를 직접 최적화             → 각 클래스의 특성 분포를 학습
특성 상관관계 고려                     독립 가정 (상관관계 무시)
느림 (반복 학습)                       빠름 (통계 계산만)
```

| 항목 | Logistic Regression | Naive Bayes |
|------|---------------------|-------------|
| 모델 유형 | 판별 모델 | 생성 모델 |
| 학습 속도 | 보통 | 매우 빠름 |
| 특성 독립 가정 | ❌ | ✅ (필수) |
| 텍스트 분류 | 좋음 | 매우 좋음 |
| 적은 데이터 | 보통 | 강함 |

---

## 9. 코드로 보기 — 타이타닉 생존 예측

```python
from sklearn.naive_bayes import GaussianNB
```

---

### 9-1. 전처리

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

# ⚠️ GaussianNB: 스케일링 불필요 (정규분포 파라미터를 특성별로 추정)
# MultinomialNB: 음수 값 불가 → MinMaxScaler 또는 CountVectorizer 사용
```

> **Note:** `GaussianNB`는 스케일링이 필요 없습니다.
> `MultinomialNB`는 음수 값을 처리하지 못하므로 `MinMaxScaler`를 사용하거나
> 텍스트 데이터에서 `CountVectorizer`를 통해 단어 빈도를 입력으로 사용합니다.

---

### 9-2. 모델 학습

```python
gnb = GaussianNB(
    var_smoothing=1e-9   # 수치 안정성을 위한 분산 평활화
)
gnb.fit(X_train, y_train)
```

---

| Parameter | Default | 역할 | 과적합 방향 |
|-----------|---------|------|------------|
| `var_smoothing` | `1e-9` | 분산에 더하는 값 (0 나눗셈 방지) | - |
| `priors` | `None` | 각 클래스의 사전 확률 (None=훈련 비율 사용) | - |

- `var_smoothing` : 수치 안정성을 위한 분산 보정값
	- 값 변화별 효과
		- 클수록 → 정규분포가 더 넓어져 부드러운 분류
		- 작을수록 → 훈련 데이터에 더 민감 (과적합 가능성)
	- 로그 스케일로 탐색: 1e-12 ~ 1e-1

#### 9-2-1. 텍스트 분류 예시 (MultinomialNB)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

# 텍스트 분류 파이프라인
text_clf = Pipeline([
    ('vect', CountVectorizer()),    # 단어 빈도 계산
    ('clf',  MultinomialNB(alpha=1.0))  # Laplace Smoothing
])
# text_clf.fit(X_text_train, y_train)
```

---

### 9-3. 평가

```python
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score
)

pred      = gnb.predict(X_test)
pred_prob = gnb.predict_proba(X_test)[:, 1]

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

### 9-4. 클래스별 특성 분포 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

feature_names = X.columns.tolist()
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.suptitle('GaussianNB — 클래스별 특성 분포 (학습된 정규분포)', fontsize=12, fontweight='bold')

for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
    x_range = np.linspace(gnb.theta_[0, i] - 3*gnb.var_[0, i]**0.5,
                          gnb.theta_[1, i] + 3*gnb.var_[1, i]**0.5, 100)
    for c, color, label in [(0, 'steelblue', 'Died'), (1, 'tomato', 'Survived')]:
        mu, sigma = gnb.theta_[c, i], gnb.var_[c, i]**0.5
        ax.plot(x_range, 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_range-mu)/sigma)**2),
                color=color, lw=2, label=label)
    ax.set_title(name, fontsize=9); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

fig.tight_layout()
plt.show()
```
