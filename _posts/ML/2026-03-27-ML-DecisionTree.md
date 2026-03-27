---
title : "[Python] Machine Learning-Decision Tree"
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

# 수정

  [ML 알고리즘] 왜 결정 트리(Decision Tree)가 여전히 현업에서
  강력한 도구인가?

  > 작성자: Seung-Ho Ryu (Senior Data Scientist)
  > 프로젝트: Machine Learning Deep-Dive 시리즈

  머신러닝 모델을 선택할 때, 우리는 흔히 '성능'과 '설명력'
  사이에서 고민합니다. 딥러닝이 아무리 뛰어나도 "왜 이런 결과가
  나왔는가?"라는 비즈니스 측면의 질문에 답하지 못하면 실무에서는
  버려지기 일쑤입니다. 이때 가장 먼저 떠오르는 강력한 카드가 바로
  결정 트리(Decision Tree)입니다.

  ---

#  1. 직관: 데이터로 하는 '전략적 스무고개'

  결정 트리의 핵심은 "데이터를 가장 잘 가르는 질문을 순서대로
  던지는 것"입니다.

  우리가 스무고개를 할 때 첫 질문으로 "그것은 생물인가요?"라고
  묻는 이유는 무엇일까요? 바로 그 질문이 정답 후보군을 가장 많이
  날려버릴 수 있는(정보 이득이 큰) 질문이기 때문입니다. 결정
  트리는 수학적으로 이 '가장 효율적인 질문'의 순서를 설계하는
  알고리즘입니다.

  ---

#  2. 이론: 불순도를 낮추는 수학적 설계

  결정 트리는 노드를 분할할 때마다 불순도(Impurity)를 최소화하는
  방향을 택합니다.

##  2-1. 지니 불순도(Gini) vs 엔트로피(Entropy)
   * Gini: $1 - \sum (p_i)^2$. 계산이 빨라 Scikit-Learn의
     기본값으로 쓰입니다. 실무적으로 성능 차이는 미미하지만, 연산
     효율이 중요할 때 유리합니다.
   * Entropy: $-\sum p_i \log p_i$. 정보 이론에 기반하며, 지니보다
     조금 더 균형 잡힌 트리를 만드는 경향이 있으나 로그 계산으로
     인해 속도가 약간 느립니다.

  2-2. 정보 이득(Information Gain)
  부모 노드의 불순도와 자식 노드들의 불순도 합 사이의 차이를
  말합니다. 모델은 이 Information Gain을 최대화하는 지점을 찾아
  분기점(Split point)을 결정합니다.

  ---

#  3. 실무: 왜 Logistic Regression이나 SVM 대신 트리를 쓰는가?


  ┌─────────────────┬──────────────────┬────────────────────┐
  │ 비교 항목       │ Decision Tree    │ Logistic           │
  │                 │                  │ Regression / SVM   │
  ├─────────────────┼──────────────────┼────────────────────┤
  │ 데이터 스케일링 │ 필요 없음. (가장 │ 필수적임           │
  │                 │ 큰 장점)         │ (정규화/표준화     │
  │                 │                  │ 미비 시 성능 저하) │
  │ 비선형성 처리   │ 매우 뛰어남.     │ 선형적 한계.       │
  │                 │ 계단식 분할로    │ Kernel 기법 필요.  │
  │                 │ 비선형 대응.     │                    │
  │ 이상치(Outlier) │ 강건함(Robust).  │ 민감함. 이상치가   │
  │                 │ 순서만 중요함.   │ 경계선을 왜곡함.   │
  │ 해석 가능성     │ White Box. 로직  │ 확률/가중치        │
  │                 │ 추적이 가능함.   │ 기반으로 직관적    │
  │                 │                  │ 해석 난해.         │
  └─────────────────┴──────────────────┴────────────────────┘

  실무 Tip: 데이터에 범주형 변수가 많고, 스케일링을 일일이 신경
  쓰기 힘든 빠른 프로토타이핑 단계에서 결정 트리는 최선의
  선택입니다.

  ---

  4. 하이퍼파라미터: 과적합(Overfitting)과의 끝없는 전쟁

  결정 트리의 최대 약점은 "너무 똑똑해서 문제"라는 것입니다. 훈련
  데이터의 잡음(Noise)까지 다 외워버립니다.

   * max_depth: 트리의 깊이. 너무 깊으면 훈련 데이터만 완벽히
     외우는 과적합이 발생합니다. 실무에서는 3~5 정도로 시작하는
     것이 좋습니다.
   * min_samples_split: 노드를 분할하기 위한 최소 샘플 수. 이 값을
     키우면 트리가 더 이상 깊어지지 못하게 강제하여 모델을
     단순화(Regularization)합니다.
   * min_samples_leaf: 리프 노드가 되기 위한 최소 샘플 수.
     데이터가 적은 리프 노드가 생기는 것을 방지하여 일반화 성능을
     높입니다.

  ---

  5. 실전 구현 (Titanic Survival Analysis)

  <img src="https://p.ipic.vip/10p7ep.png" width="70%"
  alt="Titanic Header">

    1 import pandas as pd
    2 from sklearn.model_selection import train_test_split
    3 from sklearn.tree import DecisionTreeClassifier, plot_tree
    4 import matplotlib.pyplot as plt
    5
    6 # 1. 데이터 전처리 (현업 스타일: 파생 변수 및 원-핫 인코딩)
    7 titanic = pd.read_csv('./Data/Titanic.csv')
    8 titanic['FamSize'] = titanic['SibSp'] + titanic['Parch']
    9 X = pd.get_dummies(titanic[['Pclass', 'Sex', 'Age',
      'FamSize', 'Fare']].dropna(), drop_first=True)
   10 y = titanic.loc[X.index, 'Survived']
   11
   12 X_train, X_test, y_train, y_test = train_test_split(X, y,
      test_size=0.2, random_state=42)
   13
   14 # 2. 모델 훈련 (Pruning 적용)
   15 model = DecisionTreeClassifier(max_depth=3,
      min_samples_leaf=5, random_state=42)
   16 model.fit(X_train, y_train)
   17
   18 # 3. 시각화 (의사결정 로직 확인)
   19 plt.figure(figsize=(15, 8))
   20 plot_tree(model, feature_names=X.columns,
      class_names=['Perished', 'Survived'], filled=True)
   21 plt.show()

  <img src="https://p.ipic.vip/dmvire.png" width="70%" alt="DT
  Visualization">

  ---

#  6. 실무에서의 한계와 해결 방법

   1. 불안정성(Instability): 데이터가 살짝만 바뀌어도 트리의
      구조가 완전히 뒤집힙니다.
       * 해결: 이를 극복하기 위해 트리를 수백 개 합친 Random
         Forest나 XGBoost/LightGBM 같은 앙상블 기법이
         탄생했습니다.
   2. 데이터 편향: 특정 클래스의 샘플이 너무 많으면 트리가
      그쪽으로 치우칩니다.
       * 해결: class_weight='balanced' 파라미터를 사용하거나
         언더/오버 샘플링을 고려해야 합니다.

  ---

# 7. 실무자가 흔히 하는 실수

   * 가지치기(Pruning) 생략: 기본 파라미터로 돌리면 무조건
     과적합됩니다. max_depth 제약은 선택이 아닌 필수입니다.
   * 데이터 누수(Data Leakage): 예측 시점에는 알 수 없는 정보를
     피처로 넣는 실수(예: 생존 여부와 직결된 '구조 시간' 등)를
     주의하세요.

  ---