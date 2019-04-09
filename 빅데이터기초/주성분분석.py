
# coding: utf-8

# In[ ]:


#가톨릭대학교 201521514 컴퓨터정보공학부 최해랑 


# In[ ]:


#PCA를 이용해서 유방암 데이터 셋 시각화하기 
#유방암 데이터의 음성, 양성 클래스에 대해 각 특성의 히스토그램 


# In[23]:


# coding: utf-8

from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

import numpy as np

import mglearn



cancer = load_breast_cancer()



fig,axes = plt.subplots(15,2,figsize=(10,20))

malignant = cancer.data[cancer.target==0]

benign = cancer.data[cancer.target==1]



ax = axes.ravel()

for i in range(30):

    _,bins = np.histogram(cancer.data[:,i],bins=50)

    ax[i].hist(malignant[:,i],bins=bins, color=mglearn.cm3(0),alpha=.5)

    ax[i].hist(benign[:,i],bins=bins, color=mglearn.cm3(2),alpha=.5)

    ax[i].set_title(cancer.feature_names[i])

    ax[i].set_yticks(())

ax[0].set_xlabel("attr size")

ax[0].set_ylabel("frequency")

ax[0].legend(["neg","pos"],loc="best")

fig.tight_layout()

plt.show()


# In[ ]:


#두 개의 주성분을 2차원 공간에 표현 


# In[26]:


# coding: utf-8

import matplotlib.pyplot as plt

import mglearn

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler



cancer = load_breast_cancer()



# StandardScaler를 사용해 각 틍성의 분산이 1이 되도록 스케일 조정

standard_scaler = StandardScaler()

standard_scaler.fit(cancer.data)



X_scaled = standard_scaler.transform(cancer.data)



# PCA 객체를 생성하고 fit메서드를 호출해 주성분을 찾고, transform 메서드를 호출해 데이터를 회전시키고 차원을 축소한다.

# 기본값일때 PCA는 데이터를 회전만 시키고 모든 주성분을 유지한다.

# 데이터의 차원을 줄이려면 PCA 객체를 지정하면 된다.



from sklearn.decomposition import PCA



# 데이터 첫 2개의 성분만 유지한다.

pca = PCA(n_components=2)



# PCA 모델 만들기

pca.fit(X_scaled)



# 처음 두개의 주성분을 사용해 데이터 변환

X_pca = pca.transform(X_scaled)



print("원본 데이터 형태 : {}".format(str(X_scaled.shape)))

print("축소된 데이터 형태 : {}".format(str(X_pca.shape)))

# 원본 데이터 형태 : (569, 30)

# 축소된 데이터 형태 : (569, 2)



# 두개의 주성분을 그래프로 나타내자.

plt.figure(figsize=(8,8))

mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)

plt.legend(["neg","pos"],loc="best")

plt.gca().set_aspect("equal")

plt.xlabel("1st attr")

plt.ylabel("2nd attr")

plt.show()


# In[ ]:


#최적의 가중값으로 나타낸 주성분 


# In[17]:


# coding: utf-8

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



cancer = load_breast_cancer()

# StandardScaler를 사용해 각 틍성의 분산이 1이 되도록 스케일 조정

standard_scaler = StandardScaler()

standard_scaler.fit(cancer.data)



X_scaled = standard_scaler.transform(cancer.data)



# 데이터 첫 2개의 성분만 유지한다.

pca = PCA(n_components=2)



# PCA 모델 만들기

pca.fit(X_scaled)



# 처음 두새의 주성분을 사용해 데이터 변환

X_pca = pca.transform(X_scaled)



# 히트맵 시각화 하기

plt.matshow(pca.components_, cmap="viridis")

plt.yticks([0, 1], ["comp 1", "comp 2"])

plt.colorbar()

plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')

plt.xlabel("attr")

plt.ylabel("principle comp")

plt.show()


# In[ ]:


#첫번째 주성분 (첫번째 가로 줄)의 모든특성은 부호가 같다.

이 말은 모든 특성 사이에 공통의 상호관계가 있다는 뜻이다.

따라서 한 특성의 값이 커지면 다른 값들도 높아질 것이다.



두번째 주성분의 특성은 부호가 섞여있다.

따라서 2번째 주성분의 축이 가지는 의미는 파악하기 힘들다.

