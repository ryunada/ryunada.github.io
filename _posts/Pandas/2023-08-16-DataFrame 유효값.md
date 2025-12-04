---
title : "[Python] df 행/열 별 유효값 개수 :: df.count()"
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

# df 행/열 별 유효값 개수 :: df.count()

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

```python
df.count(
    axis = 0  # 0 or 'index' : 행 | 1 or 'columns' : 열
)
```

- 각 열 또는 행에 대해 NA가 아닌 셀 계산
- Option
  - axis : 축

```python
nba.count(axis = 0)
nba.count()          # nba의 각 열당 유효한 값의 개수
```

```
Name        450
Team        450
Position    450
Birthday    450
Salary      450
dtype: int64
```

```python
nba.count(axis = 1)  # nba의 각 행당 유효한 값의 개수
```

```
0      5
1      5
2      5
3      5
4      5
      ..
445    5
446    5
447    5
448    5
449    5
Length: 450, dtype: int64
```

