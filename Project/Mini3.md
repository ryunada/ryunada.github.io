```python
import os
os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Python/Mini_Project/Mini_Project3[EDA_RowData]')
print(os.getcwd())
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    Input In [1], in <cell line: 2>()
          1 import os
    ----> 2 os.chdir('/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/Python/Mini_Project/Mini_Project3[EDA_RowData]')
          3 print(os.getcwd())
    

    FileNotFoundError: [WinError 3] 지정된 경로를 찾을 수 없습니다: '/Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook\xa0Air/2022/Bigdata/Python/Mini_Project/Mini_Project3[EDA_RowData]'



```python
import pandas as pd
import numpy as np
import chart_studio.plotly as py 
import cufflinks as cf 
cf.go_offline(connected=True)
import plotly.graph_objects as go
```

## 데이터 불러오기


```python
Category = pd.read_csv('data/DM_Category.csv',encoding = 'utf-8-sig')  # 품목 이름
City = pd.read_csv('data/DM_City.csv', encoding = 'utf-8-sig')
County = pd.read_csv('data/DM_County.csv', encoding = 'utf-8-sig') # 미국의 군은 주 아래에 있는 지방 행정구역이다.
ItemInfo = pd.read_csv('data/DM_ItemInfo.csv', encoding = 'utf-8-sig') # 품목 번호와 연관된 품목 설명
Stores = pd.read_csv('data/DM_Stores.csv', encoding = 'utf-8-sig')
Vendor = pd.read_csv('data/DM_Vendor.csv', encoding = 'utf-8-sig')
Sales = pd.read_csv('data/FT_Sales.csv', encoding = 'utf-8-sig')
US_County = pd.read_csv('data/uscounties.csv',encoding = 'utf-8-sig')  # 주이름 데이터 
```

## 데이터 확인

### Category 데이터 확인


```python
Category.info()
```


```python
Category.isnull().sum()
```

### City 데이터 확인


```python
City.info()
```


```python
City.isnull().sum()
```

### County 데이터 확인 
- County : 미국 군( 주 밑에단위 ) 총 99개


```python
County.info()
```


```python
County.isnull().sum()
```

### ItemInfo 데이터 확인


```python
ItemInfo.info()
```


```python
ItemInfo.isnull().sum()
```


```python
ItemInfo = ItemInfo.dropna() # 데이터가 많으므로 결측값 제거
```


```python
ItemInfo.isnull().sum()
```

### Stores 데이터 확인
- Store Name : 판매 가게명
- Zip Code : 우편 번호
- Store Location : 가게 위치


```python
Stores.info()
```


```python
Stores.isnull().sum()
```


```python
Stores = Stores.dropna()  # 데이터가 많으므로 제거
```


```python
Stores.isnull().sum()
```

### Vendor 데이터 확인
- Vendor : 공급업체


```python
Vendor.info()
```


```python
Vendor.isnull().sum()
```

### Sales 데이터 확인
- Bottles Sold : 병 판매량
- Volume Sold (Liters) : 주류 판매량 (단위 : 리터)
- Volume Sold (Gallons) : 주류 판매량 (단위 : 갤런) - 1Liter = 0.2641722~Gallons


```python
Sales.info()
```


```python
Sales.isnull().sum()
```


```python
# ItemInfo의 Category_ID에 해당하는 Vendor_ID를 통해서 Sales의 Category_ID를 채워 줄 수 있지만 시간상 그냥 제거한다.
Sales = Sales.dropna()  
```


```python
Sales.isnull().sum()
```

---

# 데이터 관계

![%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-08-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.28.35.png](attachment:%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-08-19%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.28.35.png)

### Sales의 데이터가 엄청 커 커널이 죽기 때문에 필요한 데이터만 출력
2022년 데이터 | Date의 형태가 object이므로 to_datetime()으로 변경  
형태도 Y-m-d로 변경


```python
Sales['Date'] = pd.to_datetime(Sales['Date'], format = '%Y-%m-%d')
```

### 2021년 데이터만 추출


```python
Sales_t1 = Sales.loc[ Sales['Date'].dt.year== 2021]
```


```python
Sales_t1
```

## 데이터 결합
- Sales_t1와    
    - ItemInfo, | Cateogry_ID로 연결  
    - Category, | Category_ID  
    - Vendor,   | Vendor_ID  
    - Country,  | City_ID  
    - City,     | County_ID  
    - Stores    | Store_ID  

## Sales_t1, ItemInfo 결합


```python
Sales_ItemInfo = pd.merge(Sales_t1, ItemInfo)
```


```python
Sales_Category = pd.merge(Sales_t1, Category, on = 'Category_ID')
```


```python
Sales_Vendor = pd.merge(Sales_t1, Vendor, on = 'Vendor_ID')
```


```python
Sales_City = pd.merge(Sales_t1, City, on = 'City_ID')
```


```python
Sales_County = pd.merge(Sales_t1, County, on  = 'County_ID')
```


```python
Sales_Stores = pd.merge(Sales_t1, Stores)
```


```python
Sales_t1
```

---

---

## 카테고리별 판매량(단위 : 병) 상위 15
=> 막대를 병으로


```python
Sales_Category_E1 = Sales_Category[['Date','Category','Bottles Sold']]
```


```python
test1 = Sales_Category_E1.groupby('Category')['Bottles Sold'].sum().sort_values(ascending = False).head(15)
```


```python
test1.iplot(kind = 'bar', 
            title = '카테고리별 판매량(단위 : 병) 상위 15위')
```


```python
r1 = test1.to_frame()
```


```python
r1
```


```python
fig = go.Figure()
colors = ['#D2AF85',]*len(r1.index)
colors[0] = '#DDA44F'
fig.add_trace(
    go.Bar(
        x = r1.index,    # x축
        y = r1['Bottles Sold'],    # y축
        text = r1['Bottles Sold'],
        textposition = 'auto', # outside, inside, auto, none
        
        marker_color = colors  ###### 차트 막대 색
    )
)

fig.update_layout(
    {
        # ppt 제목과 겹치기 때문에 생략
#         'title':{
#             'text': '<b>카테고리별 판매량(단위 : 병) 상위 15위</b>',
#             'x':0.5,     # title의 x축 위치
#             'y':0.9,     # title의 y축 위치
#             'font':{
#                 'size': 20,
#                 'color':'#DDA44F'
#             }
#         }
        'xaxis':{
            'title':'Category',# X축 제목
            'color' :'white',
            'showticklabels': True, # X축 값의 제목
            "tickfont": {
                "size": 10,
                'color':'white'
            }
        },
        'yaxis':{
            'title' : 'Bottles Sold',
            'color' : 'white'
        },
        'paper_bgcolor':'#0F0E0A', # 차트 바깥쪽 배경
        'plot_bgcolor':'#0F0E0A'   # 차트 안쪽 배경
    }
)

fig.add_annotation(
    x = 'AMERICAN VODKA',
    y = 6500000,
    text = '<b>2021 Best Vodka</b>',
    showarrow = False,
    bgcolor = 'white',
    font = dict(
        size = 10,
        color = '#0F0E0A'  ###### 변경
    ),
)
```


```python
test9 = Sales_Category_E1.groupby('Category')['Bottles Sold'].sum().sort_values(ascending = False).tail(15)
```


```python
r3 = test9.to_frame()
```


```python
fig = go.Figure()
colors = ['#D2AF85',]*len(r1.index)
colors[0] = '#DDA44F'
fig.add_trace(
    go.Bar(
        x = r3.index,    # x축
        y = r3['Bottles Sold'],    # y축
        text = r3['Bottles Sold'],
        textposition = 'auto', # outside, inside, auto, none
        
        marker_color = colors  ###### 차트 막대 색
    )
)

fig.update_layout(
    {
        # ppt 제목과 겹치기 때문에 생략
#         'title':{
#             'text': '<b>카테고리별 판매량(단위 : 병) 상위 15위</b>',
#             'x':0.5,     # title의 x축 위치
#             'y':0.9,     # title의 y축 위치
#             'font':{
#                 'size': 20,
#                 'color':'#DDA44F'
#             }
#         }
        'xaxis':{
            'title':'Category',# X축 제목
            'color' :'white',
            'showticklabels': True, # X축 값의 제목
            "tickfont": {
                "size": 10,
                'color':'white'
            }
        },
        'yaxis':{
            'title' : 'Bottles Sold',
            'color' : 'white'
        },
        'paper_bgcolor':'#0F0E0A', # 차트 바깥쪽 배경
        'plot_bgcolor':'#0F0E0A'   # 차트 안쪽 배경
    }
)

fig.add_annotation(
    x = 'IMPORTED WHISKIES',
    y = 10000,
    text = '<b>2021 Worst Vodka</b>',
    showarrow = False,
    bgcolor = 'white',
    font = dict(
        size = 10,
        color = '#0F0E0A'  ###### 변경
    ),
)
```

## 가장 많은 판매량을 가진 AMERICAN VODKA 월별


```python
# 가장 많은 판매량을 가진 AMERICAN VODKA 날짜별
test2 = Sales_Category_E1.loc[Sales_Category_E1['Category'] == 'AMERICAN VODKA']
```


```python
test2 = test2.set_index('Date')
```


```python
test2
```


```python
test3 = test2.groupby(pd.Grouper(freq = 'M'))['Category'].count()
```


```python
test3.iplot(kind = 'line',
           title = '판매량이 가장 높은 AMERICAN VODKA의 월별 판매량 추이')
```


```python
r2 =test3.to_frame()
```


```python
r2
```


```python
fig2 = go.Figure()

fig2.add_trace(
    go.Line(
        x = r2.index,
        y = r2['Category'],
        line = dict(color='#D2AF85'),
        marker_color = '#DDA44F'
    )
)

fig2.update_layout(
{
#     # ppt제목과 겹쳐 제목 생략
#     'totle':{
#         'text': '<b>판매량이 가장 높은 AMERICAN VODKA의 월별 판매량 추이</b>',
#         'x':0.5,
#         'y':0.9,
#         'font':{
#             'size':20,
#             'color':'#DDA44F'
#         }
#     }
    'xaxis':{
        'title':'Date',# X축 제목
        'color' :'white',
        'showticklabels': True, # X축 값의 제목
        'dtick':'M1',   # 전체 x축 표기
        "tickfont": {
            "size": 10,
            'color':'white'
        }
    },
    'yaxis':{
        'title':'Bottle Sold',
        'color':'white'
    },
    'paper_bgcolor':'#0F0E0A', # 차트 바깥쪽 배경
    'plot_bgcolor':'#0F0E0A'   # 차트 안쪽 배경
    
    }
)
fig2.add_annotation(
    x = '2021-06-30',
    y = 37000,
    text = '<b>2021 Best Month</b>',
    showarrow = False,
    bgcolor = 'white',
    font = dict(
        size = 10,
        color = '#0F0E0A'  ###### 변경
    ),
)
fig2.add_annotation(
    x = '2021-03-01',
    y = 29000,
    text = '<b>2021 Worst Month</b>',
    showarrow = False,
    bgcolor = 'white',
    font = dict(
        size = 10,
        color = '#0F0E0A'  ###### 변경
    ),
)
```


```python
r2
```

---


```python
Sales_County_Category = pd.merge(Sales_County, Category, on = 'Category_ID')
```


```python
Sales_County_Category_E1 = Sales_County_Category[['County','Bottles Sold']]
```


```python
test5 = Sales_County_Category_E1.groupby('County').sum()
```


```python
test5.iplot(kind = 'line')
```


```python
fig3 = go.Figure()

fig3.add_trace(
    go.Line(
        x = test5.index,
        y = test5['Bottles Sold'],
        line = dict(color='#D2AF85'),
        marker_color = '#DDA44F'
    )
)

fig3.update_layout(
{
#     # ppt제목과 겹쳐 제목 생략
#     'totle':{
#         'text': '<b>판매량이 가장 높은 AMERICAN VODKA의 월별 판매량 추이</b>',
#         'x':0.5,
#         'y':0.9,
#         'font':{
#             'size':20,
#             'color':'#DDA44F'
#         }
#     }
    'xaxis':{
        'title':'County',# X축 제목
        'color' :'white',
        'showticklabels': True, # X축 값의 제목
        'showgrid':False,
#         'dtick':'M1',   # 전체 x축 표기
        "tickfont": {
            "size": 10,
            'color':'white'
        }
    },
    'yaxis':{
        'title':'Bottle Sold',
        'color':'white'
    },
    'paper_bgcolor':'#0F0E0A', # 차트 바깥쪽 배경
    'plot_bgcolor':'#0F0E0A'   # 차트 안쪽 배경
    
    }
)
fig3.add_annotation(
    x = 'POLK',
    y = 7000000,
    text = '<b>1st County</b>',
    showarrow = False,
    bgcolor = 'white',
    font = dict(
        size = 10,
        color = '#0F0E0A'  ###### 변경
    ),
)
fig3.add_annotation(
    x = 'LINN',
    y = 3500000,
    text = '<b>2st County</b>',
    showarrow = False,
    bgcolor = 'white',
    font = dict(
        size = 10,
        color = '#0F0E0A'  ###### 변경
    ),
)
```


```python
test5
```


```python
test6 = test5.reset_index()
```


```python
test6
```


```python
US_County['County'] = US_County['County'].str.upper()
```


```python
ryu = pd.merge(test6, US_County,how = 'left', on = 'County')
```


```python
ryu = ryu.loc[ryu['state_name']=='Iowa']
```


```python
ryu = ryu[['County','Bottles Sold','lat','lng']]
```


```python
import plotly.express as px
# px.set_mapbox_access_token(open(".mapbox_token").read())
df = ryu
fig = px.scatter_mapbox(df, lat="lat", lon="lng",  color="Bottles Sold", size="Bottles Sold",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
fig.show()
```


```python
County
```
