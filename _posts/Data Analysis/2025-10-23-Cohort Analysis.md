---
title : "Cohort Analysis"
categories:
    - Data Analysis
date: 2025-10-23
toc: true
toc_label: "Concept"
toc_sticky: True
# toc_icon: ""
sidebar:
  nav: 'counts'
---

-   [I. Load Data](#i.-load-data)
-   [II. Data Preprocessing](#ii.-data-preprocessing)
    -   [II-I. 고객 ID (CustomerID)가 NaN인 행
        제거](#ii-i.-고객-id-customerid가-nan인-행-제거)
    -   [II-II. 구매한 날짜(월) 추출](#ii-ii.-구매한-날짜월-추출)
    -   [II-III. 고객 별 처음 구매한 날짜
        추출](#ii-iii.-고객-별-처음-구매한-날짜-추출)
    -   [II-IV. 고객이 유지된 날짜 변수
        생성](#ii-iv.-고객이-유지된-날짜-변수-생성)
-   [III. Visualization](#iii.-visualization)

-   데이터 설명
    -   실제 전자상거래(이커머스) 거래 내역을 기반으로 한 거래
        데이터(Transaction Data)로, 고객의 구매 행동 분석, RFM 세분화,
        코호트 분석, 리텐션 분석, 추천 시스템, 수요 예측 등의 연구에
        자주 사용됨
-   URL : https://archive.ics.uci.edu/dataset/352/online+retail
-   참고 Youtube : https://youtu.be/WWUG7T9ixTs?si=MJgMuXTc1W5MGSeW
-   변수 설명 |Feature|Type|Discription| |:—–:|:–:|:———-|
    |InvoiceNo|object|송장번호| |StockCode|object|각 상품(아이템)별로
    부여된 5자리 정수번호| |Description|object|상품명|
    |Quantity|int64|각 상품의 수량| |Invoice Date|datetime64|거래가
    발생한 날짜 및 시간(2010-12-01 ~ 2011-12-09)|
    |UnitPrice|float64|단위당 상품 가격(영국 파운드 기준)|
    |CustomerID|float64|각 고객에게 부여딘 5자리 정수번호|
    |Country|Country|해당 거래 고객이 거주하는 나라 이름|
-   분석 내용
    -   Cohort 분석을 통해 고객 유지율(Customer Retenntion)을 분석하고
        시각화 - Cohort Analysis : 시간 흐름에 따라 고객 집단(Cohort)의
        행동 변화나 잔존율(Retention)을 추적・분석하는 기법
    -   고객이 언제 유입되었는지 기준으로 그룹화하여 시간이 지남에 따라
        얼마나 오랬동안 활성화 상태를 유지하는지 파악하는 것을 목표

``` python
# Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
```

# I. Load Data

``` python
# Load Data
df = pd.read_excel("./Data/Cohort_1/Online Retail.xlsx")
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541909 entries, 0 to 541908
    Data columns (total 8 columns):
     #   Column       Non-Null Count   Dtype         
    ---  ------       --------------   -----         
     0   InvoiceNo    541909 non-null  object        
     1   StockCode    541909 non-null  object        
     2   Description  540455 non-null  object        
     3   Quantity     541909 non-null  int64         
     4   InvoiceDate  541909 non-null  datetime64[ns]
     5   UnitPrice    541909 non-null  float64       
     6   CustomerID   406829 non-null  float64       
     7   Country      541909 non-null  object        
    dtypes: datetime64[ns](1), float64(2), int64(1), object(4)
    memory usage: 33.1+ MB

# II. Data Preprocessing

## II-I. 고객 ID (CustomerID)가 NaN인 행 제거

``` python
df.dropna(subset = ['CustomerID'], axis = 0, inplace = True)
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">InvoiceNo</th>
<th data-quarto-table-cell-role="th">StockCode</th>
<th data-quarto-table-cell-role="th">Description</th>
<th data-quarto-table-cell-role="th">Quantity</th>
<th data-quarto-table-cell-role="th">InvoiceDate</th>
<th data-quarto-table-cell-role="th">UnitPrice</th>
<th data-quarto-table-cell-role="th">CustomerID</th>
<th data-quarto-table-cell-role="th">Country</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>536365</td>
<td>85123A</td>
<td>WHITE HANGING HEART T-LIGHT HOLDER</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>2.55</td>
<td>17850.0</td>
<td>United Kingdom</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>536365</td>
<td>71053</td>
<td>WHITE METAL LANTERN</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>536365</td>
<td>84406B</td>
<td>CREAM CUPID HEARTS COAT HANGER</td>
<td>8</td>
<td>2010-12-01 08:26:00</td>
<td>2.75</td>
<td>17850.0</td>
<td>United Kingdom</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>536365</td>
<td>84029G</td>
<td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>536365</td>
<td>84029E</td>
<td>RED WOOLLY HOTTIE WHITE HEART.</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541904</td>
<td>581587</td>
<td>22613</td>
<td>PACK OF 20 SPACEBOY NAPKINS</td>
<td>12</td>
<td>2011-12-09 12:50:00</td>
<td>0.85</td>
<td>12680.0</td>
<td>France</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541905</td>
<td>581587</td>
<td>22899</td>
<td>CHILDREN'S APRON DOLLY GIRL</td>
<td>6</td>
<td>2011-12-09 12:50:00</td>
<td>2.10</td>
<td>12680.0</td>
<td>France</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541906</td>
<td>581587</td>
<td>23254</td>
<td>CHILDRENS CUTLERY DOLLY GIRL</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541907</td>
<td>581587</td>
<td>23255</td>
<td>CHILDRENS CUTLERY CIRCUS PARADE</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541908</td>
<td>581587</td>
<td>22138</td>
<td>BAKING SET 9 PIECE RETROSPOT</td>
<td>3</td>
<td>2011-12-09 12:50:00</td>
<td>4.95</td>
<td>12680.0</td>
<td>France</td>
</tr>
</tbody>
</table>

<p>406829 rows × 8 columns</p>
</div>

-   result
    -   541909 rows에서 CutomerID가 비어있는 행이 제거되고 406829 rows만
        남음

## II-II. 구매한 날짜(월) 추출

-   기준이 월이기 때문에 day는 1로 통일한다

``` python
import datetime as dt

# function for Date(year, month, day)
def get_date(x):
    return dt.datetime(x.year, x.month, 1)

df['InvoiceMonth'] = df['InvoiceDate'].apply(get_date)
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">InvoiceNo</th>
<th data-quarto-table-cell-role="th">StockCode</th>
<th data-quarto-table-cell-role="th">Description</th>
<th data-quarto-table-cell-role="th">Quantity</th>
<th data-quarto-table-cell-role="th">InvoiceDate</th>
<th data-quarto-table-cell-role="th">UnitPrice</th>
<th data-quarto-table-cell-role="th">CustomerID</th>
<th data-quarto-table-cell-role="th">Country</th>
<th data-quarto-table-cell-role="th">InvoiceMonth</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>536365</td>
<td>85123A</td>
<td>WHITE HANGING HEART T-LIGHT HOLDER</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>2.55</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>536365</td>
<td>71053</td>
<td>WHITE METAL LANTERN</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>536365</td>
<td>84406B</td>
<td>CREAM CUPID HEARTS COAT HANGER</td>
<td>8</td>
<td>2010-12-01 08:26:00</td>
<td>2.75</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>536365</td>
<td>84029G</td>
<td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>536365</td>
<td>84029E</td>
<td>RED WOOLLY HOTTIE WHITE HEART.</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541904</td>
<td>581587</td>
<td>22613</td>
<td>PACK OF 20 SPACEBOY NAPKINS</td>
<td>12</td>
<td>2011-12-09 12:50:00</td>
<td>0.85</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541905</td>
<td>581587</td>
<td>22899</td>
<td>CHILDREN'S APRON DOLLY GIRL</td>
<td>6</td>
<td>2011-12-09 12:50:00</td>
<td>2.10</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541906</td>
<td>581587</td>
<td>23254</td>
<td>CHILDRENS CUTLERY DOLLY GIRL</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541907</td>
<td>581587</td>
<td>23255</td>
<td>CHILDRENS CUTLERY CIRCUS PARADE</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541908</td>
<td>581587</td>
<td>22138</td>
<td>BAKING SET 9 PIECE RETROSPOT</td>
<td>3</td>
<td>2011-12-09 12:50:00</td>
<td>4.95</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
</tr>
</tbody>
</table>

<p>406829 rows × 9 columns</p>
</div>

## II-III. 고객 별 처음 구매한 날짜 추출

-   Why? 처음 구매한 월을 추출하여 얼마나 고객이 오랫동안 유지되는지
    알아보기 위함
-   고객 ID (CustomerID)별 처음 구매한 날짜를 추가

``` python
df['Cohort First Month'] = df.groupby('CustomerID')['InvoiceMonth'].transform('min')
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">InvoiceNo</th>
<th data-quarto-table-cell-role="th">StockCode</th>
<th data-quarto-table-cell-role="th">Description</th>
<th data-quarto-table-cell-role="th">Quantity</th>
<th data-quarto-table-cell-role="th">InvoiceDate</th>
<th data-quarto-table-cell-role="th">UnitPrice</th>
<th data-quarto-table-cell-role="th">CustomerID</th>
<th data-quarto-table-cell-role="th">Country</th>
<th data-quarto-table-cell-role="th">InvoiceMonth</th>
<th data-quarto-table-cell-role="th">Cohort First Month</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>536365</td>
<td>85123A</td>
<td>WHITE HANGING HEART T-LIGHT HOLDER</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>2.55</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>536365</td>
<td>71053</td>
<td>WHITE METAL LANTERN</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>536365</td>
<td>84406B</td>
<td>CREAM CUPID HEARTS COAT HANGER</td>
<td>8</td>
<td>2010-12-01 08:26:00</td>
<td>2.75</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>536365</td>
<td>84029G</td>
<td>KNITTED UNION FLAG HOT WATER BOTTLE</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>536365</td>
<td>84029E</td>
<td>RED WOOLLY HOTTIE WHITE HEART.</td>
<td>6</td>
<td>2010-12-01 08:26:00</td>
<td>3.39</td>
<td>17850.0</td>
<td>United Kingdom</td>
<td>2010-12-01</td>
<td>2010-12-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541904</td>
<td>581587</td>
<td>22613</td>
<td>PACK OF 20 SPACEBOY NAPKINS</td>
<td>12</td>
<td>2011-12-09 12:50:00</td>
<td>0.85</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541905</td>
<td>581587</td>
<td>22899</td>
<td>CHILDREN'S APRON DOLLY GIRL</td>
<td>6</td>
<td>2011-12-09 12:50:00</td>
<td>2.10</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541906</td>
<td>581587</td>
<td>23254</td>
<td>CHILDRENS CUTLERY DOLLY GIRL</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541907</td>
<td>581587</td>
<td>23255</td>
<td>CHILDRENS CUTLERY CIRCUS PARADE</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541908</td>
<td>581587</td>
<td>22138</td>
<td>BAKING SET 9 PIECE RETROSPOT</td>
<td>3</td>
<td>2011-12-09 12:50:00</td>
<td>4.95</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
</tr>
</tbody>
</table>

<p>406829 rows × 10 columns</p>
</div>

## II-IV. 고객이 유지된 날짜 변수 생성

-   월 기준으로 생성
-   InoviceDate와 Cohort First Month의 차이는 고객이 유지된 날짜이다.

``` python
# Function 날짜의 year, month, day 추출 
def get_date_elements(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day

# 적용
Invoice_year, Invoice_month, _ = get_date_elements(df, 'InvoiceMonth')
Cohort_year, Cohort_month, _ = get_date_elements(df, 'Cohort First Month')
```

``` python
# 사람들이 처음 구매 후 활성화된 기간
year_diff = Invoice_year - Cohort_year
month_diff = Invoice_month - Cohort_month
df['Cohort Retention Period'] = year_diff * 12 + month_diff + 1
df.tail()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">InvoiceNo</th>
<th data-quarto-table-cell-role="th">StockCode</th>
<th data-quarto-table-cell-role="th">Description</th>
<th data-quarto-table-cell-role="th">Quantity</th>
<th data-quarto-table-cell-role="th">InvoiceDate</th>
<th data-quarto-table-cell-role="th">UnitPrice</th>
<th data-quarto-table-cell-role="th">CustomerID</th>
<th data-quarto-table-cell-role="th">Country</th>
<th data-quarto-table-cell-role="th">InvoiceMonth</th>
<th data-quarto-table-cell-role="th">Cohort First Month</th>
<th data-quarto-table-cell-role="th">Cohort Retention Period</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">541904</td>
<td>581587</td>
<td>22613</td>
<td>PACK OF 20 SPACEBOY NAPKINS</td>
<td>12</td>
<td>2011-12-09 12:50:00</td>
<td>0.85</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
<td>5</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541905</td>
<td>581587</td>
<td>22899</td>
<td>CHILDREN'S APRON DOLLY GIRL</td>
<td>6</td>
<td>2011-12-09 12:50:00</td>
<td>2.10</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
<td>5</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541906</td>
<td>581587</td>
<td>23254</td>
<td>CHILDRENS CUTLERY DOLLY GIRL</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
<td>5</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541907</td>
<td>581587</td>
<td>23255</td>
<td>CHILDRENS CUTLERY CIRCUS PARADE</td>
<td>4</td>
<td>2011-12-09 12:50:00</td>
<td>4.15</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
<td>5</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">541908</td>
<td>581587</td>
<td>22138</td>
<td>BAKING SET 9 PIECE RETROSPOT</td>
<td>3</td>
<td>2011-12-09 12:50:00</td>
<td>4.95</td>
<td>12680.0</td>
<td>France</td>
<td>2011-12-01</td>
<td>2011-08-01</td>
<td>5</td>
</tr>
</tbody>
</table>

</div>

\[!Caution\] - Cohort Retention Period에 +1을 하는 이유 - 코호트
분석에서는 “고객이 코호트에 처음 유입된 달을 기준으로 몇 개월이
경과했는가”를 계산합니다. - 그런데, 고객이 유입된 같은 달에 다시
구매했다면 경과 개월 수는 0개월이 되어버립니다. - 분석 관점에서는 이
’0개월차’를 고객 활동의 첫 번째 달(Month 1) 로 간주하는 것이
자연스럽습니다. - 따라서, 0개월차 → 1개월차로 맞추기 위해 Cohort
Retention Period에 +1을 더해줍니다.

# III. Visualization

``` python
# 실제 활동한 고유 고객의 수를 계산
cohort_data = df.groupby(['Cohort First Month', 'Cohort Retention Period'])['CustomerID'].apply(pd.Series.nunique).reset_index()
cohort_data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Cohort First Month</th>
<th data-quarto-table-cell-role="th">Cohort Retention Period</th>
<th data-quarto-table-cell-role="th">CustomerID</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">0</td>
<td>2010-12-01</td>
<td>1</td>
<td>948</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">1</td>
<td>2010-12-01</td>
<td>2</td>
<td>362</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2</td>
<td>2010-12-01</td>
<td>3</td>
<td>317</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">3</td>
<td>2010-12-01</td>
<td>4</td>
<td>367</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">4</td>
<td>2010-12-01</td>
<td>5</td>
<td>341</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">...</td>
<td>...</td>
<td>...</td>
<td>...</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">86</td>
<td>2011-10-01</td>
<td>2</td>
<td>93</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">87</td>
<td>2011-10-01</td>
<td>3</td>
<td>46</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">88</td>
<td>2011-11-01</td>
<td>1</td>
<td>321</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">89</td>
<td>2011-11-01</td>
<td>2</td>
<td>43</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">90</td>
<td>2011-12-01</td>
<td>1</td>
<td>41</td>
</tr>
</tbody>
</table>

<p>91 rows × 3 columns</p>
</div>

-   Caution
    -   실제 활동한 고유 고객의 수를 계산하는 것은 고객 한 명이 해당
        기간 동안 10번 구매했더라도, 그 고객은 1명으로만 집계됨

``` python
# create a pivot table
cohort_table = cohort_data.pivot(index = 'Cohort First Month', columns = ['Cohort Retention Period'], values = 'CustomerID')
cohort_table
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th">Cohort Retention Period</th>
<th data-quarto-table-cell-role="th">1</th>
<th data-quarto-table-cell-role="th">2</th>
<th data-quarto-table-cell-role="th">3</th>
<th data-quarto-table-cell-role="th">4</th>
<th data-quarto-table-cell-role="th">5</th>
<th data-quarto-table-cell-role="th">6</th>
<th data-quarto-table-cell-role="th">7</th>
<th data-quarto-table-cell-role="th">8</th>
<th data-quarto-table-cell-role="th">9</th>
<th data-quarto-table-cell-role="th">10</th>
<th data-quarto-table-cell-role="th">11</th>
<th data-quarto-table-cell-role="th">12</th>
<th data-quarto-table-cell-role="th">13</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">Cohort First Month</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">2010-12-01</td>
<td>948.0</td>
<td>362.0</td>
<td>317.0</td>
<td>367.0</td>
<td>341.0</td>
<td>376.0</td>
<td>360.0</td>
<td>336.0</td>
<td>336.0</td>
<td>374.0</td>
<td>354.0</td>
<td>474.0</td>
<td>260.0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-01-01</td>
<td>421.0</td>
<td>101.0</td>
<td>119.0</td>
<td>102.0</td>
<td>138.0</td>
<td>126.0</td>
<td>110.0</td>
<td>108.0</td>
<td>131.0</td>
<td>146.0</td>
<td>155.0</td>
<td>63.0</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-02-01</td>
<td>380.0</td>
<td>94.0</td>
<td>73.0</td>
<td>106.0</td>
<td>102.0</td>
<td>94.0</td>
<td>97.0</td>
<td>107.0</td>
<td>98.0</td>
<td>119.0</td>
<td>35.0</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-03-01</td>
<td>440.0</td>
<td>84.0</td>
<td>112.0</td>
<td>96.0</td>
<td>102.0</td>
<td>78.0</td>
<td>116.0</td>
<td>105.0</td>
<td>127.0</td>
<td>39.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-04-01</td>
<td>299.0</td>
<td>68.0</td>
<td>66.0</td>
<td>63.0</td>
<td>62.0</td>
<td>71.0</td>
<td>69.0</td>
<td>78.0</td>
<td>25.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-05-01</td>
<td>279.0</td>
<td>66.0</td>
<td>48.0</td>
<td>48.0</td>
<td>60.0</td>
<td>68.0</td>
<td>74.0</td>
<td>29.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-06-01</td>
<td>235.0</td>
<td>49.0</td>
<td>44.0</td>
<td>64.0</td>
<td>58.0</td>
<td>79.0</td>
<td>24.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-07-01</td>
<td>191.0</td>
<td>40.0</td>
<td>39.0</td>
<td>44.0</td>
<td>52.0</td>
<td>22.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-08-01</td>
<td>167.0</td>
<td>42.0</td>
<td>42.0</td>
<td>42.0</td>
<td>23.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-09-01</td>
<td>298.0</td>
<td>89.0</td>
<td>97.0</td>
<td>36.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-10-01</td>
<td>352.0</td>
<td>93.0</td>
<td>46.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-11-01</td>
<td>321.0</td>
<td>43.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2011-12-01</td>
<td>41.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
</tbody>
</table>

</div>

``` python
# change index 형태 변경
cohort_table.index = cohort_table.index.strftime("%B %Y")
cohort_table
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th">Cohort Retention Period</th>
<th data-quarto-table-cell-role="th">1</th>
<th data-quarto-table-cell-role="th">2</th>
<th data-quarto-table-cell-role="th">3</th>
<th data-quarto-table-cell-role="th">4</th>
<th data-quarto-table-cell-role="th">5</th>
<th data-quarto-table-cell-role="th">6</th>
<th data-quarto-table-cell-role="th">7</th>
<th data-quarto-table-cell-role="th">8</th>
<th data-quarto-table-cell-role="th">9</th>
<th data-quarto-table-cell-role="th">10</th>
<th data-quarto-table-cell-role="th">11</th>
<th data-quarto-table-cell-role="th">12</th>
<th data-quarto-table-cell-role="th">13</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">Cohort First Month</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">December 2010</td>
<td>948.0</td>
<td>362.0</td>
<td>317.0</td>
<td>367.0</td>
<td>341.0</td>
<td>376.0</td>
<td>360.0</td>
<td>336.0</td>
<td>336.0</td>
<td>374.0</td>
<td>354.0</td>
<td>474.0</td>
<td>260.0</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">January 2011</td>
<td>421.0</td>
<td>101.0</td>
<td>119.0</td>
<td>102.0</td>
<td>138.0</td>
<td>126.0</td>
<td>110.0</td>
<td>108.0</td>
<td>131.0</td>
<td>146.0</td>
<td>155.0</td>
<td>63.0</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">February 2011</td>
<td>380.0</td>
<td>94.0</td>
<td>73.0</td>
<td>106.0</td>
<td>102.0</td>
<td>94.0</td>
<td>97.0</td>
<td>107.0</td>
<td>98.0</td>
<td>119.0</td>
<td>35.0</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">March 2011</td>
<td>440.0</td>
<td>84.0</td>
<td>112.0</td>
<td>96.0</td>
<td>102.0</td>
<td>78.0</td>
<td>116.0</td>
<td>105.0</td>
<td>127.0</td>
<td>39.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">April 2011</td>
<td>299.0</td>
<td>68.0</td>
<td>66.0</td>
<td>63.0</td>
<td>62.0</td>
<td>71.0</td>
<td>69.0</td>
<td>78.0</td>
<td>25.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">May 2011</td>
<td>279.0</td>
<td>66.0</td>
<td>48.0</td>
<td>48.0</td>
<td>60.0</td>
<td>68.0</td>
<td>74.0</td>
<td>29.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">June 2011</td>
<td>235.0</td>
<td>49.0</td>
<td>44.0</td>
<td>64.0</td>
<td>58.0</td>
<td>79.0</td>
<td>24.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">July 2011</td>
<td>191.0</td>
<td>40.0</td>
<td>39.0</td>
<td>44.0</td>
<td>52.0</td>
<td>22.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">August 2011</td>
<td>167.0</td>
<td>42.0</td>
<td>42.0</td>
<td>42.0</td>
<td>23.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">September 2011</td>
<td>298.0</td>
<td>89.0</td>
<td>97.0</td>
<td>36.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">October 2011</td>
<td>352.0</td>
<td>93.0</td>
<td>46.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">November 2011</td>
<td>321.0</td>
<td>43.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">December 2011</td>
<td>41.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
</tbody>
</table>

</div>

``` python
# Visualize our results in heatmap
plt.figure(figsize = (21, 10))
plt.title("Cohort")
sns.heatmap(cohort_table, annot = True, cmap = "Blues")
plt.show()
```

![](Cohort%20Analysis_files/figure-markdown_strict/cell-12-output-1.png)

``` python
# cohort table for percentage
# 고객 유지율 = 특정 경과 기간의 잔존 고객수 / 코호트 유입 시점의 총 고객 수
new_cohort_table = cohort_table.divide(cohort_table.iloc[:, 0], axis = 0)
new_cohort_table
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th">Cohort Retention Period</th>
<th data-quarto-table-cell-role="th">1</th>
<th data-quarto-table-cell-role="th">2</th>
<th data-quarto-table-cell-role="th">3</th>
<th data-quarto-table-cell-role="th">4</th>
<th data-quarto-table-cell-role="th">5</th>
<th data-quarto-table-cell-role="th">6</th>
<th data-quarto-table-cell-role="th">7</th>
<th data-quarto-table-cell-role="th">8</th>
<th data-quarto-table-cell-role="th">9</th>
<th data-quarto-table-cell-role="th">10</th>
<th data-quarto-table-cell-role="th">11</th>
<th data-quarto-table-cell-role="th">12</th>
<th data-quarto-table-cell-role="th">13</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">Cohort First Month</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">December 2010</td>
<td>1.0</td>
<td>0.381857</td>
<td>0.334388</td>
<td>0.387131</td>
<td>0.359705</td>
<td>0.396624</td>
<td>0.379747</td>
<td>0.354430</td>
<td>0.354430</td>
<td>0.394515</td>
<td>0.373418</td>
<td>0.500000</td>
<td>0.274262</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">January 2011</td>
<td>1.0</td>
<td>0.239905</td>
<td>0.282660</td>
<td>0.242280</td>
<td>0.327791</td>
<td>0.299287</td>
<td>0.261283</td>
<td>0.256532</td>
<td>0.311164</td>
<td>0.346793</td>
<td>0.368171</td>
<td>0.149644</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">February 2011</td>
<td>1.0</td>
<td>0.247368</td>
<td>0.192105</td>
<td>0.278947</td>
<td>0.268421</td>
<td>0.247368</td>
<td>0.255263</td>
<td>0.281579</td>
<td>0.257895</td>
<td>0.313158</td>
<td>0.092105</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">March 2011</td>
<td>1.0</td>
<td>0.190909</td>
<td>0.254545</td>
<td>0.218182</td>
<td>0.231818</td>
<td>0.177273</td>
<td>0.263636</td>
<td>0.238636</td>
<td>0.288636</td>
<td>0.088636</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">April 2011</td>
<td>1.0</td>
<td>0.227425</td>
<td>0.220736</td>
<td>0.210702</td>
<td>0.207358</td>
<td>0.237458</td>
<td>0.230769</td>
<td>0.260870</td>
<td>0.083612</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">May 2011</td>
<td>1.0</td>
<td>0.236559</td>
<td>0.172043</td>
<td>0.172043</td>
<td>0.215054</td>
<td>0.243728</td>
<td>0.265233</td>
<td>0.103943</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">June 2011</td>
<td>1.0</td>
<td>0.208511</td>
<td>0.187234</td>
<td>0.272340</td>
<td>0.246809</td>
<td>0.336170</td>
<td>0.102128</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">July 2011</td>
<td>1.0</td>
<td>0.209424</td>
<td>0.204188</td>
<td>0.230366</td>
<td>0.272251</td>
<td>0.115183</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">August 2011</td>
<td>1.0</td>
<td>0.251497</td>
<td>0.251497</td>
<td>0.251497</td>
<td>0.137725</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">September 2011</td>
<td>1.0</td>
<td>0.298658</td>
<td>0.325503</td>
<td>0.120805</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">October 2011</td>
<td>1.0</td>
<td>0.264205</td>
<td>0.130682</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">November 2011</td>
<td>1.0</td>
<td>0.133956</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">December 2011</td>
<td>1.0</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
<td>NaN</td>
</tr>
</tbody>
</table>

</div>

``` python
# create a percentages
plt.figure(figsize = (21, 10))
plt.title("Cohort")
sns.heatmap(new_cohort_table, annot = True, cmap = "Blues", fmt = '.0%')
plt.show()
```

![](Cohort%20Analysis_files/figure-markdown_strict/cell-14-output-1.png)
