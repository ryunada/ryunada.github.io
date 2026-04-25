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

# Decision Tree 개념
Decision Tree는 데이터를 **조건으로 반복적으로 분할**하여 예측하는 알고리즘

```
각 노드에서 불순도를 최소화하는 조건으로 데이터를 나눔
→ 최종 Leaf 노드에서 예측값 반환
→ 분류(Classification) / 회귀(Regression) 모두 적용 가능
→ 별도의 스케일링 불필요
```

---

<img src = "/assets/img/ML/decisiontree/decision_tree.png" width = "70%" alt = "decision_tree">


## 트리 분할 기준

노드 분할 시 **불순도(Impurity)** 를 최소화하는 방향으로 조건을 선택

| 기준 | 설명 | 사용 |
|------|------|------|
| **지니 계수 (Gini)** | 무작위로 선택한 샘플이 잘못 분류될 확률 | 분류 (기본값) |
| **엔트로피 (Entropy)** | 데이터의 불확실성 정도 | 분류 |
| **MSE** | 평균 제곱 오차 | 회귀 |

```
Class A 40개, Class B 60개인 노드의 계산 예시:

p_A = 0.4, p_B = 0.6

Gini = 1 - (0.4² + 0.6²) = 0.48
Entropy = -(0.4 × log₂0.4 + 0.6 × log₂0.6) = 0.971

완전히 순수한 노드(한 클래스만 존재):
  Gini = 0, Entropy = 0

→ 두 기준 모두 순수할수록 0에 가까워짐
→ 실무에서 성능 차이는 거의 없음 (Gini가 계산이 빠름)
```

---

## 언제 사용하는가

### 사용해야 할 때

| 상황 | 이유 |
|------|------|
| **모델 결과를 비전문가에게 설명해야 할 때** | 트리 시각화로 예측 근거를 직관적으로 설명 가능 |
| **전처리를 최소화하고 싶을 때** | 스케일링 불필요, 결측값 처리 최소화 |
| **빠른 프로토타이핑** | 구현이 단순하고 학습이 빠름 |
| **변수 중요도가 필요할 때** | Feature Importance로 중요 변수 빠르게 파악 |
| **앙상블 모델의 구조 이해** | Random Forest, XGBoost의 기반 알고리즘이므로 이해 필수 |

### 사용하지 말아야 할 때

| 상황 | 이유 |
|------|------|
| **높은 예측 성능이 최우선일 때** | 단독 사용 시 앙상블 대비 성능 낮음 |
| **데이터가 적고 변수가 많을 때** | 과적합 위험 높음 |
| **연속적인 수치 예측 (회귀)** | 선형 패턴 표현 불가, 외삽(extrapolation) 불가 |

### 실무 활용 사례

```
금융    : 대출 승인 여부 판단 (심사 기준을 트리로 시각화)
의료    : 질병 진단 보조 (의사에게 판단 근거 제공)
마케팅  : 고객 세그멘테이션 1차 분석
```

---

## Parameters

| Parameter           | 설명                          | Default |
| ------------------- | ----------------------------- | ------- |
| `max_depth`         | 트리의 최대 깊이               | None    |
| `min_samples_split` | 노드 분할 최소 샘플 수         | 2       |
| `min_samples_leaf`  | Leaf 노드 최소 샘플 수         | 1       |
| `max_features`      | 분할 시 고려할 최대 변수 수    | None    |
| `criterion`         | 분할 기준 (`gini`, `entropy`)  | `gini`  |
| `random_state`      | 재현성 고정                    | None    |

- `max_depth` : 트리의 최대 깊이를 제한하는 파라미터
    - 값 변화별 효과
        - 클수록 → 복잡한 패턴 학습, 과적합 ↑
        - 작을수록 → 단순한 모델, 과적합 ↓
    - `None`이면 모든 Leaf가 순수해질 때까지 분할 → 과적합 위험

- `min_samples_split` : 노드를 분할하기 위해 필요한 최소 샘플 수
    - 값 변화별 효과
        - 클수록 → 분할 덜함 → 모델 단순 → 과적합 ↓
        - 작을수록 → 계속 분할 → 모델 복잡 → 과적합 ↑
    - Options
        - `int` : 최소 샘플 개수
        - `float` : 전체 데이터 대비 비율

- `min_samples_leaf` : Leaf 노드에 있어야 하는 최소 샘플 수
    - 값 변화별 효과
        - 클수록 → Leaf가 커짐 → 부드러운 모델 → 과적합 ↓
        - 작을수록 → Leaf가 작아짐 → 복잡한 모델 → 과적합 ↑

- `max_features` : 각 노드 분할 시 고려할 변수 수
    - 값 변화별 효과
        - 클수록 → 더 좋은 분할 탐색, 트리 간 유사도 ↑
        - 작을수록 → 빠른 학습, 트리 다양성 ↑
    - `None`이면 전체 변수 사용

- `criterion` : 분할 기준
    - `gini` : 계산 빠름, 실무 기본값
    - `entropy` : 정보 이론 기반, 성능 차이 거의 없음

---

## 실습 — Titanic 생존자 예측

### I. Library & Data Load
```python
# warning ignore 
import warnings
warnings.filterwarnings(action = 'ignore')

# Data Preprocessing 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Visualization 
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='AppleGothic'  # mac 한글 깨짐 현상 해결

# Model Definition
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Evaluation
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, roc_auc_score
```

```python
titanic = pd.read_csv('./Data/Titanic.csv')  # 데이터 로드
titanic
```

```
	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
...	...	...	...	...	...	...	...	...	...	...	...
886	0	2	Montvila, Rev. Juozas	male	27.0	0	0	211536	13.0000	NaN	S
887	1	1	Graham, Miss. Margaret Edith	female	19.0	0	0	112053	30.0000	B42	S
888	0	3	Johnston, Miss. Catherine Helen "Carrie"	female	NaN	1	2	W./C. 6607	23.4500	NaN	S
889	1	1	Behr, Mr. Karl Howell	male	26.0	0	0	111369	30.0000	C148	C
890	0	3	Dooley, Mr. Patrick	male	32.0	0	0	370376	7.7500	NaN	Q
891 rows × 11 columns
```

### II. Preprocessing

#### II-I. Feature Engineering
```python
# 파생변수 생성
titanic['FamSize'] = titanic['SibSp'] + titanic['Parch']

# 분석 변수 선택
use_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked']
titanic = titanic[use_cols].dropna(subset = ['Age'])  # 결측값 제거

# 자료형 변환
titanic[['Survived', 'Pclass', 'Sex', 'Embarked']] = \
    titanic[['Survived', 'Pclass', 'Sex', 'Embarked']].astype('category')  # 범주형으로 변환
titanic['Age'] = titanic['Age'].astype('int')  # 정수형으로 변환

# One-Hot Encoding
titanic = pd.get_dummies(titanic, columns = ['Pclass', 'Sex', 'Embarked'], drop_first = True)
titanic
```

```
	Survived	Age	FamSize	Fare	Pclass_2	Pclass_3	Sex_male	Embarked_Q	Embarked_S
0	0	22	1	7.2500	False	True	True	False	True
1	1	38	1	71.2833	False	False	False	False	False
2	1	26	0	7.9250	False	True	False	False	True
3	1	35	1	53.1000	False	False	False	False	True
4	0	35	0	8.0500	False	True	True	False	True
...	...	...	...	...	...	...	...	...	...
885	0	39	5	29.1250	False	True	False	True	False
886	0	27	0	13.0000	True	False	True	False	True
887	1	19	0	30.0000	False	False	False	False	True
889	1	26	0	30.0000	False	False	True	False	False
890	0	32	0	7.7500	False	True	True	True	False
714 rows × 9 columns
```

#### II-II. 수치형 변수 시각화
```python
# 수치형 변수 시각화
def numberic_plot(df, target):
    g = sns.PairGrid(df, hue = target)  # 주어진 데이터 컬럼에 대한 모든 조합을 만들어주는 빈 틀을 위한 코드        
    g.map_diag(sns.histplot)            # 삼각행렬의 중간 부분
    g.map_lower(sns.scatterplot)        # 아래 부분
    
    # 상관 계수 행렬을 구하고 상관 계수 값 표시
    corr_matrix = df.corr()
    for i, j in zip(*plt.np.triu_indices_from(g.axes, k = 1)):                                        # np.triu_indices_from : 삼각행렬의 위쪽 삼각형의 인덱스 (k = 0 : 대각 행렬 포함, 1 : 제외)
        g.axes[i, j].annotate(f"corr : {corr_matrix.iloc[i, j]:.2f}",                                 # 상관계수
                              (0.5, 0.5), xycoords = "axes fraction", ha = 'center', va = 'center',   # 중앙 정렬
                              fontsize = 12,                                                          # 글자 크기
                              color = 'black')                                                        # 글자 색  
    g.add_legend()  # 범례 표시
    plt.show()

Columns = ['Age', 'FamSize', 'Fare', 'Survived']  # 수치형 변수
numberic_plot(titanic[Columns], 'Survived')
```

<img src = "/assets/img/ML/decisiontree/dt_수치형 변수 시각화.png" width = "70%" alt = "dt_수치형 변수 시각화">


#### II-III. Train & Test Split
```python
y = titanic['Survived']
X = titanic.drop(['Survived'], axis = 1)

# Decision Tree는 스케일링 불필요
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)  # 훈련/테스트 분할 (75:25)
```

### III. Model Train
```python
DT = DecisionTreeClassifier(
    max_depth = None,       # 트리의 최대 깊이
    min_samples_split = 2,  # 노드 분할에 필요한 최소 샘플 수
    min_samples_leaf = 1,   # 말단(Leaf) 노드에 필요한 최소 샘플 수
    max_features = None,    # 각 분할을 위해 검사할 특징 수
    criterion = 'gini',     # 분할 기준
    random_state = 0        # 고정값
)

DT.fit(X_train, y_train)
```

#### III-I. Tree Visualization
```python
plt.figure(figsize = (20, 8))
plot_tree(DT,
          feature_names = X.columns,
          class_names = ['사망', '생존'],
          filled = True,
          max_depth = 3)   # 전체 트리는 너무 크므로 3단계만 표시
plt.show()
```

<img src = "/assets/img/ML/decisiontree/dt_tree visualization.png.png" width = "70%" alt = "dt_tree visualization.png">

### IV. Feature Importance Visualization
```python
importances = pd.Series(DT.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending = True)

importances.plot(kind = 'barh', figsize = (8, 5), color = '#1565C0')
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('중요도')
plt.tight_layout()
plt.show()
```

<img src = "/assets/img/ML/decisiontree/dt_feature_important.png" width = "70%" alt = "dt_feature_important">

### V. Evaluation Score

```python
pred = DT.predict(X_test)

cfx  = confusion_matrix(y_test, pred)
sensitivity = cfx[1, 1] / (cfx[1, 0] + cfx[1, 1])  # 민감도  
specificity = cfx[0, 0] / (cfx[0, 0] + cfx[0, 1])  # 특이도

print(f"정확도 : {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"민감도 : {sensitivity * 100:.2f}%")
print(f"특이도 : {specificity * 100:.2f}%")
print(f"Confusion Matrix:\n{cfx}")
```

```
정확도 : 79.89%
민감도 : 78.95%
특이도 : 80.58%
Confusion Matrix:
[[83 20]
 [16 60]]
```

### VI. ROC Curve
```python
fpr, tpr, thresholds = roc_curve(y_test, pred)

J = tpr - fpr
ix = np.argmax(J)             # 가장 큰 원소의 위치(최대값의 인덱스)
best_thresh = thresholds[ix]

#plot roc and best threshold
sens, spec = tpr[ix], 1 - fpr[ix]

# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle = '--', markersize = 0.01, color = 'black')  # 중간 기준 선
plt.plot(fpr, tpr, marker = '.', color = 'black', markersize = 0.01, label = "Ridge AUC = %.2f" % roc_auc_score(y_test, pred))
plt.scatter(fpr[ix], tpr[ix], marker = '+', s = 100, color = 'r', 
            label = f"Best threshold = {best_thresh:.3f}, \nSensitivity = {sens:.3f}, \nSpecificity = {spec:.3f}")

# axis labels
plt.title("ROC Curve")
plt.xlabel("False Positive Rate(1 - Specificity)")
plt.ylabel("True Positive Rate(Sensitivity)")
plt.legend(loc = 4)

# show the plot
plt.show()
```

<img src = "/assets/img/ML/decisiontree/idt_roc_socre.png" width = "70%" alt = "dt_roc_socre">

## 장단점

**장점**

- 결과 해석이 직관적 (트리 시각화 가능)
- 스케일링 불필요
- 범주형 / 연속형 변수 모두 처리 가능
- Feature Importance 제공

**단점**

- 과적합이 발생하기 쉬움 → `max_depth` 제한 필수
- 데이터 변화에 민감 (분산이 높음)
- 단독 사용보다 **앙상블(Random Forest, Gradient Boost)** 로 쓰는 것이 일반적
