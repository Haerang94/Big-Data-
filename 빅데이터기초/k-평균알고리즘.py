
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


size = 200
x = np.random.normal(0, 2, size)
y = np.random.normal(-0.5, 3, size)


# In[5]:


plt.figure(figsize=(8,6))
plt.scatter(x, y)
plt.xlim(-8, 8)
plt.ylim(-10, 10);


# In[11]:


def get_distance(p1, p2):
    """
    input : 두 데이터 포인트의 위치
    output : 두 포인트 사이의 직선 거리 (Euclidean Distance)
    """
    z = np.array(p1)-np.array(p2)
    return np.sqrt(np.dot(z, z))


def k_points(n):
    """
    input : n = 총 만들 그룹의 수 (또는 k)
    output : n 수의 그룹 리스트

    여기서 'points' 는 각 그룹에 속하게 되는 데이터 포인트를 나타냅니다.
    'prev' 는 그룹의 전 위치, 'curr' 는 현 위치.
    각 그룹은 리스트 내 dictionary 로 구성합니다.
    """
    return [{'prev':None, 'curr':(np.random.randint(-4, 4, 1)[0],np.random.randint(-4, 4, 1)[0]), 'points':[]} for _ in range(n)]


def update_position():
    """
    output : 각 그룹이 갖고 있는 현 데이터 dictionary

    전위치를 현위치로 업데이트 후 현위치를 데이터 포인트의 중앙으로 업데이트 시킵니다.
    모든 정보 업데이트 후 각 그룹의 point 를 초기화 합니다. 다음 iteration 때 다른 point 의 값을 넣기 때문.
    """
    dat = {}
    for i, point in enumerate(p):
        point['prev'] = point['curr']
        point['curr'] = np.mean(point['points'], axis=0) if len(point['points']) > 0 else point['prev']
        dat[i] = np.array(point['points'])
        point['points'] = []
    return dat
def done_moving():
    """
    output : True 또는 False

    또 다른 iteration 이 필요한지 확인합니다. 만약 전위치와 현위치가 다르면 false, 각 그룹의 위치가 그대로면 true.
    """
    for point in p:
        if np.any(point['curr'] != point['prev']):
            return False
    return True


def assign_points(x, y):
    """
    input : 데이터 포인트의 위치

    각 데이터 포인트의 위치와 모든 그룹의 위치를 계산 후 가장 가까이 있는 그룹으로 업데이트 합니다.
    """
    for x1, y1 in zip(x, y):

        dist = [get_distance(point['curr'], [x1,y1]) for point in p]

        # Get the closest kth point from current x,y
        closest_point = dist.index(min(dist))
        p[closest_point]['points'].append([x1,y1])


# In[13]:


p = k_points(5)
colors = ['red', 'blue', 'green', 'black', 'yellow']
plt.figure(figsize=(8,6))
plt.xlim(-8, 8)
plt.ylim(-10, 10)

assign_points(x, y)
points = update_position()

for i in range(len(p)):
    plt.scatter(p[i]['curr'][0], p[i]['curr'][1], s=110, c=colors[i])

    # 가끔 어느 특정 그룹은 모든 데이터 포인트 들로부터 다른 그룹보다 멀리 떨어져
    # points 리스트에 아무 데이터도 없을 수가 있기 때문에 if 로 확인 후 그래프로 그려줍니다.
    if len(points[i]) > 0:
        plt.scatter(points[i][:,0], points[i][:,1], c=colors[i], alpha=0.4)


# In[14]:


i = 1
fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlim(-8, 8)
ax.set_ylim(-10, 10)

while not done_moving():

    assign_points(x, y)
    points = update_position()

    for j in range(len(p)):
        ax.scatter(p[j]['curr'][0], p[j]['curr'][1], s=110, c=colors[j])
        if len(points[j]) > 0:
            ax.scatter(points[j][:,0], points[j][:,1], c=colors[j], alpha=0.4)

    i += 1
    plt.pause(0.8)
    # 전 그래프 초기화
    ax.cla()
    
print("Done at {}th iteration".format(i))

# 그래프가 필요 없으면 밑의 코드를
# i = 1

# while not done_moving():

#     assign_points(x, y)
#     update_position()
#     i += 1

# print("{} 번째 loop 에서 끝".format(i))


# In[ ]:




