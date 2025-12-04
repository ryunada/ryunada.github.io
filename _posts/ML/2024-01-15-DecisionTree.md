---
title : "[Python] Decision Tree"
categories:
    - ML
date: 2024-01-15
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---

# Decision Tree

<span style = "color : blue">Create. Seung-Ho Ryu (Comment. Jung-In Seo)</span>

<details>
    <summary>Reference</summary>

- 파이썬 머신러닝 완벽 가이드 (Chapter 4-2)
- 주머니속 머신러닝 (Chapter 10-5)
- 데이터 과학자를 위한 실전 머신러닝(Chapter 6)

</details>

<img src = "/assets/img/ML/dt/DT_Information.png" width = "70%" alt = "DT_Information">

**< 실습 데이터셋 >**

- Titanic 데이터 셋 : 타이타닉 사건(1912년도) 때 타이타닉 호에 탑승했던 승객들의 정보, 생존 여부등으로 총 11개의 Feature로 이루어져 있습니다.
    - 11개의 변수 중 분석에 사용할 변수는 아래의 표에서 소개를 합니다.

<img src = "/assets/img/ML/dt/Features_Information.png" width = "70%" alt = "Features_Information">

- Flow Chart

<img src = "/assets/img/ML/dt/Flow_Chart.png" width = "70%" alt = "Flow_Chart">


```python
# Data Preprocessing 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Visualization 
import seaborn as sns
import matplotlib.pyplot as plt

# warning ignore 
import warnings
warnings.filterwarnings(action = 'ignore')

# Model Definition
from sklearn.tree import DecisionTreeClassifier

# Evaluation
from sklearn.metrics import roc_curve, accuracy_score, confusion_matrix, roc_auc_score
```

# 1. 데이터 불러오기


```python
# 데이터 불러오기
titanic = pd.read_csv('./Data/Titanic.csv')

titanic
```

|         | Survived | Pclass |                                              Name |    Sex |  Age | SibSp | Parch |           Ticket |    Fare | Cabin | Embarked |
| ------: | -------: | -----: | ------------------------------------------------: | -----: | ---: | ----: | ----: | ---------------: | ------: | ----: | -------- |
|   **0** |        0 |      3 |                           Braund, Mr. Owen Harris |   male | 22.0 |     1 |     0 |        A/5 21171 |  7.2500 |   NaN | S        |
|   **1** |        1 |      1 | Cumings, Mrs. John Bradley (Florence Briggs Th... | female | 38.0 |     1 |     0 |         PC 17599 | 71.2833 |   C85 | C        |
|   **2** |        1 |      3 |                            Heikkinen, Miss. Laina | female | 26.0 |     0 |     0 | STON/O2. 3101282 |  7.9250 |   NaN | S        |
|   **3** |        1 |      1 |      Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 |     1 |     0 |           113803 | 53.1000 |  C123 | S        |
|   **4** |        0 |      3 |                          Allen, Mr. William Henry |   male | 35.0 |     0 |     0 |           373450 |  8.0500 |   NaN | S        |
| **...** |      ... |    ... |                                               ... |    ... |  ... |   ... |   ... |              ... |     ... |   ... | ...      |
| **886** |        0 |      2 |                             Montvila, Rev. Juozas |   male | 27.0 |     0 |     0 |           211536 | 13.0000 |   NaN | S        |
| **887** |        1 |      1 |                      Graham, Miss. Margaret Edith | female | 19.0 |     0 |     0 |           112053 | 30.0000 |   B42 | S        |
| **888** |        0 |      3 |          Johnston, Miss. Catherine Helen "Carrie" | female |  NaN |     1 |     2 |       W./C. 6607 | 23.4500 |   NaN | S        |
| **889** |        1 |      1 |                             Behr, Mr. Karl Howell |   male | 26.0 |     0 |     0 |           111369 | 30.0000 |  C148 | C        |
| **890** |        0 |      3 |                               Dooley, Mr. Patrick |   male | 32.0 |     0 |     0 |           370376 |  7.7500 |   NaN | Q        |

891 rows × 11 columns

# 2. 전처리


```python
# 가족 변수 추가
titanic['FamSize'] = titanic['SibSp'] + titanic['Parch']  # FamSize = 형제 및 배우자 수 + 부모님 및 자녀 수

# 분석에 사용할 변수만 선택
Use_Columns = ['Survived', 'Pclass', 'Sex', 'Age', 'FamSize', 'Fare', 'Embarked']  
titanic = titanic[Use_Columns] 

# 결측값 제거
titanic.dropna(subset = ['Age'], axis = 0, inplace = True)

# 변수 형태 변경
titanic[['Survived', 'Pclass', 'Sex', 'Embarked']] = titanic[['Survived', 'Pclass', 'Sex', 'Embarked']].astype('category')
titanic['Age'] = titanic['Age'].astype('int')

# One-Hot-Encoding
titanic = pd.get_dummies(titanic, columns = ['Pclass', 'Sex', 'Embarked'], drop_first = True)
```

# 3. 데이터 탐색


```python
# 변수 형태
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 714 entries, 0 to 890
    Data columns (total 9 columns):
     #   Column      Non-Null Count  Dtype   
    ---  ------      --------------  -----   
     0   Survived    714 non-null    category
     1   Age         714 non-null    int64   
     2   FamSize     714 non-null    int64   
     3   Fare        714 non-null    float64 
     4   Pclass_2    714 non-null    bool    
     5   Pclass_3    714 non-null    bool    
     6   Sex_male    714 non-null    bool    
     7   Embarked_Q  714 non-null    bool    
     8   Embarked_S  714 non-null    bool    
    dtypes: bool(5), category(1), float64(1), int64(2)
    memory usage: 26.6 KB

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

<img src = "https://p.ipic.vip/iz17jv.png" alt = "numberic plot">


# 4. 데이터 분할


```python
# 생존 여부 변수를 Target으로 지정
y = titanic['Survived']

# 나머지 변수들을 예측 변수로 지정
X = titanic.drop(['Survived'], axis = 1)

# 75 : 25로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

# 5. 결정 트리 (DT ; Decision Tree)  
```Python
from sklearn.tree import DecisionTreeClassifier

# 결정 트리(Decision Tree) 모형 정의(파라미터 기본값)
DecisionTreeClassifier(
    max_depth = None,                   # 트리의 최대 깊이
    min_samples_split = 2,              # 노드 분할에 필요한 최소 샘플 수
    min_samples_leaf = 1,               # 말단(Leaf) 노드에 필요한 최소 샘플 수
    max_features = None,                # 각 분할을 위해 검사할 특징의 수
    random_state = 0                    # 고정값
)
```

<details>
    <summary>Parameters</summary>
<img src = "/assets/img/ML/dt/DT_Parameter_Information.png" width = "60%" alt = "DT_Parameter_Information">

</details>

## 5-1. 모형 정의


```python
DT = DecisionTreeClassifier(
    max_depth = None,       # 트리의 최대 깊이
    min_samples_split = 2,  # 노드 분할에 필요한 최소 샘플 수
    min_samples_leaf = 1,   # 말단(Leaf) 노드에 필요한 최소 샘플 수
    max_features = None,    # 각 분할을 위해 검사할 특징 수
    random_state = 0        # 고정값
)
```

## 5-2. 모형 훈련


```python
DT_fit = DT.fit(X_train, y_train)
DT_fit
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(random_state=0)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=0)</pre></div></div></div></div></div>



### 5-2-1. 변수 중요도


```python
feature_importance = DT.feature_importances_    # 피처 중요도 가져오기
indices = np.argsort(feature_importance)[::-1]  # 정렬
feature_names = X_train.columns                 # 피처 이름

# 피처 중요도 시각화
plt.bar(range(X_train.shape[1]), feature_importance[indices], align = "center")
plt.xticks(range(X_train.shape[1]), feature_names[indices], rotation = 45)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Decision Tree Feature Importance")
plt.show()
```

<img src = "https://p.ipic.vip/imloob.png" alt = "Feature Importance">


### 5-2-2. 트리 시각화


```python
from sklearn import tree

fig = plt.figure(figsize = (40, 20))
tree.plot_tree(DT_fit,
               feature_names = list(X_train.columns),
               class_names = ['0', '1'],
               filled = True)        
plt.show()
```

<img src = "https://p.ipic.vip/oq6uz6.png" alt = "tree plot">    


## 6-3. 모형 평가


```python
DT_pred = DT.predict(X_test)
DT_pred
```


    array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1,
           1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1,
           1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
           1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 1, 1])

### 6-3-1. Confusion Matrix

<img src = "/assets/img/ML/dt/Confusion Matrix.png" width = "50%" alt = "Confusion Matrix">


```python
DT_cfx = confusion_matrix(y_test, DT_pred)                     # Confusion Matrix(True, pred)
DT_sensitivity = DT_cfx[1, 1] / (DT_cfx[1, 0] + DT_cfx[1, 1])  # 민감도 계산
DT_specificity = DT_cfx[0, 0] / (DT_cfx[0, 0] + DT_cfx[0, 1])  # 특이도 계산

print(f"DT 정확도(accuracy) : {accuracy_score(y_test, DT_pred) * 100 :.2f}%")
print(f"DT Confusion_Matrix :\n{DT_cfx}")
print(f"DT 민감도(sensitivity) : {DT_sensitivity * 100 :.2f}%")
print(f"DT 특이도(specificity) : {DT_specificity * 100 :.2f}%")
```

    DT 정확도(accuracy) : 79.89%
    DT Confusion_Matrix :
    [[83 20]
     [16 60]]
    DT 민감도(sensitivity) : 78.95%
    DT 특이도(specificity) : 80.58%


### 6-3-2. ROC 곡선


```python
DT_pred = DT.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, DT_pred)

J = tpr - fpr
ix = np.argmax(J)             # 가장 큰 원소의 위치(최대값의 인덱스)
best_thresh = thresholds[ix]

#plot roc and best threshold
sens, spec = tpr[ix], 1 - fpr[ix]

# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle = '--', markersize = 0.01, color = 'black')  # 중간 기준 선
plt.plot(fpr, tpr, marker = '.', color = 'black', markersize = 0.01, label = "Ridge AUC = %.2f" % roc_auc_score(y_test, DT_pred))
plt.scatter(fpr[ix], tpr[ix], marker = '+', s = 100, color = 'r', 
            label = f"Best threshold = {best_thresh:.3f}, \nSensitivity = {sens:.3f}, \nSpecificity = {spec:.3f}")

# axis labels
plt.xlabel("False Positive Rate(1 - Specificity)")
plt.ylabel("True Positive Rate(Sensitivity)")
plt.legend(loc = 4)

# show the plot
plt.show()
```

<img src = "https://p.ipic.vip/rei3uf.png" alt = "ROC Plot">
    

