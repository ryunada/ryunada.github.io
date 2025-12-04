---
title : "[Python] df 차원 확인 :: df.ndim & df.shape"
categories:
    - Pandas
date: 2023-08-16
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
math: true           # 수식 설정
---
<details>
    <summary>Reference</summary>
        Pandas In Action
</details>

# 1. DataFrame 차원 확인 :: df.ndim & df.shape
```python
import pandas as pd

nba = pd.read_csv(
    './Data/nba.csv',
    parse_dates = ['Birthday']  # 'Birthday'의 데이터 유형을 날짜/시간(datetime)으로 강제 변환
)
nba
```

|         |           Name |                Team | Position |   Birthday |   Salary |
| ------: | -------------: | ------------------: | -------: | ---------: | -------: |
|   **0** |   Shake Milton |  Philadelphia 76ers |       SG | 1996-09-26 |  1445697 |
|   **1** | Christian Wood |     Detroit Pistons |       PF | 1995-09-27 |  1645357 |
|   **2** |  PJ Washington |   Charlotte Hornets |       PF | 1998-08-23 |  3831840 |
|   **3** |   Derrick Rose |     Detroit Pistons |       PG | 1988-10-04 |  7317074 |
|   **4** |  Marial Shayok |  Philadelphia 76ers |        G | 1995-07-26 |    79568 |
| **...** |            ... |                 ... |      ... |        ... |      ... |
| **445** |  Austin Rivers |     Houston Rockets |       PG | 1992-08-01 |  2174310 |
| **446** |    Harry Giles |    Sacramento Kings |       PF | 1998-04-22 |  2578800 |
| **447** |    Robin Lopez |     Milwaukee Bucks |        C | 1988-04-01 |  4767000 |
| **448** |  Collin Sexton | Cleveland Cavaliers |       PG | 1999-01-04 |  4764960 |
| **449** |    Ricky Rubio |        Phoenix Suns |       PG | 1990-10-21 | 16200000 |

450 rows × 5 columns

## 1-1. df 차원을 Int형태로 :: df.ndim

```python
df.ndim
```

- 축/배열 차원의 수를 int로 반환

```python
nba.ndim  # nba의 차원의 수
```

```
2
```

## 1-2. df 차원을 Tuple 형태로 :: df.shape

```python
df.shape
```

- DataFrame의 차원을 나타내는 튜플을 반환

```python
nba.shape # nba의 차원 형태
```

```
(450, 5)
```

