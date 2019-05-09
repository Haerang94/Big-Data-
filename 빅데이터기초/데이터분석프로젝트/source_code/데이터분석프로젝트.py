#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


#2017년도 119 활동실적 통계를 불러온다, 필요한 모든 데이터들을 통합하기 위해 모든 데이터의 연도는 2017년도로 통일했다 
#전체자료를 불러왔는 데 필요없는 행들이 보이는 것을 알 수 있다 
performance= pd.read_excel('../data/119활동실적.xls',
                           encoding='utf-8')
performance.head()


# In[5]:


#필요없는 부분의 행을 제외하고 엑셀파일에서 4번째 줄부터 불러온다 (header사용)
performance= pd.read_excel('../data/119활동실적.xls',
                           header=3,
                           encoding='utf-8')
performance.head()


# In[6]:


#나타난 각 합계를 알아볼 수 없으므로 알아볼 수 있도록 합계 컬럼의 이름을 알맞게 수정한다 (rename) 
performance.rename(columns={performance.columns[1]:'구별',
                          performance.columns[2]:'구조활동총계',
                           performance.columns[3]:'출동건수총계',
                           performance.columns[8]:'구조인원총계'},inplace=True)
performance.head()


# In[7]:


performance_sum= pd.read_excel('../data/119활동실적.xls',
                           header=3,
                           parse_cols='A,B,C,D,I',
                           encoding='utf-8')
performance_sum.head()


# In[8]:


#총계컬럼들만 따로 쉽게 비교하고 싶으므로 performance_sum을 새로 만든다 
#처리사건총계는 출동사건총계의 부분집합이고(포함됨), 이 통계에서는 총 출동건수 중 구조인원에 해당하는 기준이 명확하지 않다 
performance_sum.rename(columns={performance_sum.columns[1]:'구별',
                          performance_sum.columns[2]:'출동건수총계',
                           performance_sum.columns[3]:'처리사건총계',
                           performance_sum.columns[4]:'구조인원총계'},inplace=True)
performance_sum.head()


# In[9]:


#출동건수총계를 기준으로 내림차순으로 정렬한다; 강남구, 송파구, 서초구 순으로 출동건수가 많다는 것을 알 수 있다 
performance_sum.sort_values(by='출동건수총계',ascending=False).head(5)


# In[16]:


#서울시 구별 인구수를 확인하기 위해 서울시 구별 인구통계 엑셀파일을 불러온다
pop_seoul=pd.read_excel('../data/서울시구별인구통계.xls',encoding='utf-8')
pop_seoul.head()


# In[18]:


# 필요없는 행을 제외하기 위해 3번째 행부터 불러온다 
pop_seoul=pd.read_excel('../data/서울시구별인구통계.xls',
                        header=2,
                        parse_cols='B,D,G,J,N',
                        encoding='utf-8')
pop_seoul.head()


# In[19]:


#각 컬럼의 이름을 알아보기 쉽게 수정한다 
pop_seoul.rename(columns={pop_seoul.columns[0]:'구별',
                          pop_seoul.columns[1]:'인구수',
                          pop_seoul.columns[2]:'한국인',
                          pop_seoul.columns[3]:'외국인',
                          pop_seoul.columns[4]:'고령자'}, inplace=True)
                         
pop_seoul.head()


# In[20]:


#한국인비율, 고령자비율, 외국인비율 컬럼을 추가해준다 
pop_seoul['고령자비율']=pop_seoul['고령자']/pop_seoul['인구수']*100
pop_seoul['한국인비율']=pop_seoul['한국인']/pop_seoul['인구수']*100
pop_seoul['외국인비율']=pop_seoul['외국인']/pop_seoul['인구수']*100
pop_seoul.head()


# In[23]:


#구별 합계가 필요하지 서울시 전체 합계는 필요없으므로 첫번째 행은 삭제한다
pop_seoul.drop([0],inplace=True)
pop_seoul.head()


# In[26]:


pop_seoul.sort_values(by='인구수',ascending=False).head()


# In[27]:


# 구별 119출동현황과 구별 인구수 현황 데이터를 합친다
data_result=pd.merge(performance_sum,pop_seoul,on='구별')
data_result.head()


# In[30]:


#그래프를 그리기 위해 index를 구 이름으로 설정한다 
data_result.set_index('구별',inplace=True)
data_result.head()


# In[31]:


import numpy as np


# In[32]:


#상관관계를 비교해본다 -0.5 정도로 보통의 음의 상관관계를 보인다
np.corrcoef(data_result['고령자비율'],data_result['출동건수총계'])


# In[33]:


np.corrcoef(data_result['외국인비율'],data_result['출동건수총계'])


# In[34]:


#인구수와는 어느 정도 양의 상관관계를 나타내고 있는 것을 볼 수 있다 
np.corrcoef(data_result['인구수'],data_result['출동건수총계'])


# In[40]:


#그래프를 그리는 모듈 임포트하기 
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline #그래프의 결과를 출력세션에 나타나게 한다')


# In[42]:


import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus']=False

if platform.system()=='Darwin':
    rc('font',family='AppleGothic')
elif platform.system()=='Windows':
    path="c:/Windows/Fonts/malgun.ttf"
    font_name=font_manager.FontProperties(fname=path).get_name()
    rc('font',family=font_name)
else:
    print('Unknown system... sorry~~~')


# In[43]:


data_result.head()


# In[44]:


data_result['출동건수총계'].plot(kind='barh',grid=True,figsize=(10,10))
plt.show()


# In[45]:


# 보기 쉽게 하기위해 데이터를 정렬해서 그래프 다시 그리기
data_result['출동건수총계'].sort_values().plot(kind='barh',grid=True,figsize=(10,10))
plt.show()
#강남구가 월등히 다른 구보다 출동횟수가 높다. 인구수가 많이 때문에..??


# In[47]:


# 인구 대비 출동횟수총계 그래프 확인
data_result['출동건수비율']=data_result['출동건수총계']/ data_result['인구수']*100
data_result['출동건수비율'].sort_values().plot(kind='barh',grid=True,figsize=(10,10))
plt.show()
#인구수 대비 출동횟수는 종로구와 중구가 월등히 높은 것을 알 수 있다. 강남구도 인구수 대비 3위로 높은 위치를 차지한다. 


# In[49]:


#산포도로 나타내보자
plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'],data_result['출동건수총계'],s=50)
plt.xlabel('인구수')
plt.ylabel('출동건수')
plt.grid()
plt.show()


# In[50]:


#앞에서 인구수와 출동건수가 양의 상관관계가 있는 것을 확인해보았다 대표 직선을 그리자
fp1=np.polyfit(data_result['인구수'],data_result['출동건수총계'],1)
fp1


# In[51]:


f1=np.poly1d(fp1) #y축 데이터
fx=np.linspace(100000,700000,100)#y축 데이터


# In[70]:


plt.figure(figsize=(10,10))
plt.scatter(data_result['인구수'],data_result['출동건수총계'],s=50)
plt.plot(fx,f1(fx),ls='dashed',lw=3,color='r')
plt.xlabel('인구수')
plt.ylabel('출동건수')
plt.grid()
plt.show()


# In[193]:


#직선에서 벗어나는 오차컬럼을 구해서 추가하고 내림차순 정렬
fp1=np.polyfit(data_result['인구수'],data_result['출동건수총계'],1)
f1=np.poly1d(fp1)
fx=np.linspace(100000,700000,100)
data_result['오차']=np.abs(data_result['출동건수총계']/100-f1(data_result['인구수'])/100)
df_sort=data_result.sort_values(by='오차',ascending=False)
df_sort.head()
                         


# In[194]:


plt.figure(figsize=(10,6))
plt.scatter(data_result['인구수'],data_result['출동건수총계'],
           c=data_result['오차'],s=40)
plt.plot(fx,f1(fx),ls='dashed',lw=3,color='g')

for n in range(10):
    plt.text(df_sort['인구수'][n]*1.02,df_sort['출동건수총계'][n]*0.98,
            df_sort.index[n],fontsize=10)
    
    plt.xlabel('인구수')
    plt.ylabel('인구당출동건수비율')
    plt.colorbar()
    plt.grid()
    plt.show()


# df_sort.head()

# In[198]:


from sklearn import preprocessing
col=['출동건수총계','인구수']

x=df_sort[col].values
min_max_scaler=preprocessing.MinMaxScaler()

x_scaled=min_max_scaler.fit_transform(x.astype(float))
events_norm=pd.DataFrame(x_scaled,
                         columns=col,
                         index=df_sort.index)
events_norm.head()


# In[ ]:




