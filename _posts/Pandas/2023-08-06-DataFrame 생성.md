---
title : "[Python] DataFrame 생성 :: pd.DataFrame()"
categories:
    - Pandas
date: 2023-08-06
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
---

# DataFrame 생성 :: pd.DataFrame( )

<details>
    <summary>Reference</summary>
        Pandas In Action
</details>

```python
import pandas as pd

pd.DataFrame(
    data = None,           # 딕셔너리(Dictionary), Numpy ndarray, etc...
    index = None,          # 리스트(list), MultiIndex 객체
    columns = None,         # 리스트(list), MultiIndex 객체
)
```

-   Option
    -   data : 데이터
    -   index : 행 이름 설정
    -   columns : 열 이름 설정
-   중복된 행과 열 인덱스를 허용함
    -   But, 인덱스는 고유값(유니크 값)을 가지는 것이 좋음
-   기본적인 DataFrame의 index, columns옵션에 MultiIndex 객체를 넣게 되면 MultiIndex DataFrame을 만들 수 있음

```python
# 딕셔너리(Dictionary)로 데이터 프레임(DataFrame) 생성
city_data = {
    'City' : ['New York City', 'Paris', 'Barcelona', 'Rome'],
    'Country' : ['United States', 'France', 'Spain', 'Italy'],
    'Population' : pd.Series([8600000, 2141000, 5515000, 2873000])
}

cities = pd.DataFrame(city_data)
cities
```

|   | City | Country | Population |
| --: | --: | --: | --: |
| **0** | New York City | United States | 8600000 |
| **1** | Paris | France | 2141000 |
| **2** | Barcelona | Spain | 5515000 |
| **3** | Rome | Italy | 2873000 |

```python
# Numpy ndarray로 데이터프레임(DataFrame) 생성
row_labels = ['Morning', 'Afternoon',' Evening']

column_labels = (
    'Monday',
    'Tuseday',
    'Wednesday',
    'Thursday',
    'Friday'
)
random_data = np.random.randint(1,101, [3,5]) # 랜덤값 1부터 100까지 3행 5열 생성

df = pd.DataFrame(
    data = random_data,        # 데이터
    index = row_labels,        # 행 이름(index)
    columns = column_labels    # 열 이름(index)
)
df
```

|   | Monday | Tuseday | Wednesday | Thursday | Friday |
| --: | --: | --: | --: | --: | --: |
| **Morning** | 23 | 98 | 52 | 84 | 94 |
| **Afternoon** | 81 | 21 | 38 | 1 | 90 |
| **Evening** | 48 | 72 | 26 | 46 | 64 |

```python
data = [
    ['A','B+'],
    ['C+','C'],
    ['D-','A'],
]

columns = ['Schools','Cost of Living']

address = [
    ('8809 Flair Square', ' Toddside', 'IL', '37206'),
    ('9901 Austin Street', 'Toddside',' IL', '37206'),
    ('905 Hogan Quarter', 'Franklin', 'IL', '37206')
]

# MultiIndex 객체 생성
row_index = pd.MultiIndex.from_tuples(
    tuples = address,
    names = ['Street','City','State','Zip']
)

area_grades = pd.DataFrame(
    data = data, 
    index = row_index, 
    columns = columns
)
area_grades
```

|   |   |   |   | Schools | Cost of Living |
| --: | --: | --: | --: | --: | --: |
| **Street** | **City** | **State** | **Zip** |   |   |
| **8809 Flair Square** | **Toddside** | **IL** | **37206** | A | B+ |
| **9901 Austin Street** | **Toddside** | **IL** | **37206** | C+ | C |
| **905 Hogan Quarter** | **Franklin** | **IL** | **37206** | D- | A |