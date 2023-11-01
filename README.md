# bbang-hyung-5
# 비지도학습 (Unsupervised Learning)
앞에서 배웠던 지도학습 (Supervised Learning) 방법과 달리 입력 데이터에 대한 정답(라벨)이 주어지지 않습니다.

지도학습은 데이터에 대한 정답값을 예측하는 것이 목표였다면, 비지도학습은 데이터의 차원을 축소하거나 비슷한 것끼리 군집화(개체들을 비슷한 것끼리 그룹을 나누는 것)하는 것이 목표가 됩니다.

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/1f0aca22-d405-43ba-b962-682f13d1f7b9)
# PCA (Priciple Component Analysis)
현재 가장 많이 사용하는 차원 축소 기법

- Feature 갯수를 줄인다
- 주요 특징들을 추출한다
- (데이터가 줄어드니) 계산 비용이 감소한다
- 전반적인 경향을 사람이 이해하기 쉽다.
## Iris 데이터셋을 사용한 PCA
![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/246fb173-66cd-4c95-8ae7-97b15370fcf7)

### Iris 데이터셋 로드
```python
from sklearn.datasets import load_iris
import pandas as pd
// iris 데이터셋 패키지와 판다스 패키지를 가져온다.

iris = load_iris()
// iris 데이터 안에 iris 데이터를 넣는다.

df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']
// 판다스 데이터 프레임을 사용하여 데이터 셋 로드를 한다.

df.head()
// 상위 5개 폴거를 뽑아온다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/7c81d6c1-0d08-4a31-a3b0-f05b80bcd3ac)
### PCA 수행
```python
from sklearn.decomposition import PCA
// PCA 패키지를 가져온다.

pca = PCA(n_components=2)
// 특징을 2개로 만들 수 있는 PCA를 생성한다.

data = pca.fit_transform(df.drop(columns=['target']))
// 데이터를 특징 2개로 만들고 비지도 학습이기 때문에 타겟 값을 삭제 시킨다.

data[:5]
// 0~4번째 데이터를 불러온다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/1b356d7a-cbba-44c5-bfa3-18a26a8a9061)
### PCA 시각화
```python
import matplotlib.pyplot as plt
import seaborn as sns
// seaborn 패키지와 matplotlib 패키지를 가져온다.

plt.figure(figsize=(16, 8))
// 가로 16 세로 8 사이즈 차트를 만든다.

sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=df['target'])
// 특징 2개로 정리한 산점도 차트를 그린다.

plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/5b8b6f7f-9260-469b-89f0-a34583a2f46c)
## PCA 데이터로 학습
### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(data, df[['target']], test_size=0.2, random_state=2021)
// 검증 데이터와 훈련 데이터로 분할 한다.

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
// 훈련데이터와 검증데이터가 무슨 형태로 들어가 있는지 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/46b528a7-2ed7-4107-a618-52996446b933)
### SVM 학습/검증
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
// SVC 모델과 accuracy_score 패키지를 가져온다.

model = SVC()
// SVC 모델을 정의한다.

model.fit(x_train, y_train['target'])
// 모델을 훈련시킨다.

y_pred = model.predict(x_val)
// 정답값을 예측한다.

accuracy_score(y_val, y_pred) * 100
// 예측한 값을 정답 값과 비교하여 정확도를 확인한다.
```
feature가 많을 수록 정확도가 오르지만 PCA를 통해서 feature 값을 2로 낮췄기 때문에 예전 데이터보다 1% 정도 정확도가 떨어졌다.(기능은 빠름) 

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/c5bef46e-6f3d-454a-ab24-c028d52ebbab)
## MNIST 데이터셋을 사용한 PCA
이미지 데이터에서도 사용이 가능하다

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/db2c657f-1882-4a62-a705-42864d4b67e8)
### MNIST 데이터셋 로드
```python
from sklearn.datasets import load_digits
// load_digits 데이터 셋 패키지를 가져온다.

digits = load_digits()
// digits 데이터에 load_digits 데이터를 넣는다.

data = digits['data']
target = digits['target']
// 데이터는 digits 데이터로 정의하고 target은 digits target으로 정의한다.
```
### 정규화
```python
from sklearn.preprocessing import MinMaxScaler
// MinMaxScaler 정규화 패키지를 가져온다.

scaler = MinMaxScaler()
// scaler 데이터에 MinMaxScaler 객체를 생성한다.

scaled = scaler.fit_transform(data)
// scaled 데이터 안에 정규화시킨 데이터 값을 넣는다.

scaled[0]
// 0번째 데이터를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/ae125928-90c4-42dd-a98f-e2eca004c1e3)
### PCA 수행
```python
from sklearn.decomposition import PCA
// PCA 패키지를 가져온다.

pca = PCA(n_components=10)
// 특징을 10개를 만들 수 있는 PCA를 생성한다.

data_pca = pca.fit_transform(scaled)
// data_pca에 특징을 10개로 변환시킨 scaled 데이터를 넣는다.

data_pca[0]
// 0번째 데이터를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/a58aec81-db92-4778-b45b-a62852d4c215)
### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(data_pca, target, test_size=0.2, random_state=2021)
// 검증 데이터와 훈련 데이터로 분할한다.

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
// 검증 데이터와 훈련 데이터의 형태를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/b4cda88c-a315-493f-b0ec-85741c7f95cf)
### SVM 학습/검증

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
// SVC 모델과 accuracy_score 패키지를 가져온다.

model = SVC()
// 모델 정의

model.fit(x_train, y_train)
// 모델 훈련

y_pred = model.predict(x_val)
// 정답값예측

accuracy_score(y_val, y_pred) * 100
// 정답 값과 예측 값을 비교하여 정확도를 확인한다.
```
이 예제도 1.2% 정도 정확도가 낮아진 것을 볼 수 있다.

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/e54a6941-1cc9-43ff-bb71-d9d608e3731d)

# 머신러닝의 정체
[심화] 현재 우리가 머신러닝이라고 부르는 것의 의미를 상상해보자
## 사람의 얼굴을 구분하는 인공지능은 어떻게 만들까

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/0e2972a9-9002-4b37-afd4-61d3c12357eb)
## 다시 MNIST로
우리는 64개의 Feature를 가진 손글씨 숫자 이미지를 PCA 기법을 사용해서 10개의 Feature롤 축소시켰다

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/fb5eeeec-d2c6-4eea-aeab-8e9fb35b48b4)
## PCA 3차원 그래프
만약 차원을 축소(64차원 -> 3차원)시켜서 3개의 Feature를 더 잘 뽑아낼 수 있다면 계산이 더 쉬워지지 않을까?

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
// load_digits 데이터 셋과 정규화 패키지, PCA 패키지, plotly.express 패키지를 가져온다.

digits = load_digits()
// 데이터를 넣어준다.

data = digits['data']
target = digits['target']
// 데이터 셋 시켜준다.

scaler = MinMaxScaler()
// 데이터 정규화 패키지 생성.

scaled = scaler.fit_transform(data)
// 데이터를 정규화시켜 넣는다.

pca = PCA(n_components=3)
// 특징 3개로 만들 수 있는 PCA를 넣는다.

data_pca = pca.fit_transform(scaled)
// 특징을 3개로 변환 시켜 데이터를 넣는다.

fig = px.scatter_3d(x=data_pca[:, 0], y=data_pca[:, 1], z=data_pca[:, 2], color=target, opacity=0.7)
// plotly.express 패키지를 통해 3차원 산점도 차트를 그린다.
fig.show()
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/53e44b20-7e0a-48db-8dd9-a8c293018df7)
## 사람 얼굴의 경우
만얄 손글씨 숫자 10개를 사람 얼굴 사진이라고 생각한다면?

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/600403a5-1c41-4b93-81c2-cb6f23c67c6b)

- O : 저장된 오바마 사진
- T : 저장된 트럼프 사진
- I : 새로 들어온 입력 사진(아직 누군지 모름)

1. 각각의 사진에서 차원 축소를 하여 3차원 만들고 (x,y,z) 좌표를 추출한다.
2. 각각의 사진과 새로 들어온 입력 사진(I)과의 거리(Distanc)를 계산한다.
3. 거리가 가장 가까운 사진으로부터 "오바마"라는 결과를 예측한다.

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/8461ff3a-b8a8-435a-a3e7-67eb760932e3)

3차원에서 거리를 구하는 방법

(x1,y1,z1)-(x2,y2,z2) 제곱의 루트를 씌어주면 된다.

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/c48e757a-6b8c-4088-90b8-1fe61cbd2dbd)

머신러닝이란 Big 데이터를 n차원 공간에 한 점으로 표시하는 것
# K-Means Clustering
가장 유명한 군집화 알고리즘

- 중점 Centroid을 기준으로 가까운 포인트들을 군집화
- 원하는 갯수 (K)로 군집화
- 연산 속도가 빠른 편

다양한 군집화 알고리즘 살펴보기(가로 군집화 알고리즘)

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/3f3db8e1-b199-4ae9-9397-6cb8be3aecc0)
## Iris 데이터셋을 이용한 실습
### 데이터셋 로드
target은 빼고 우리가 정답을 모른다고 가정한다.

```python
from sklearn.datasets import load_iris
import pandas as pd
// iris 데이터 셋 패키지와 판다스 패키지를 가져온다

iris = load_iris()
// 데이터를 넣는다.

df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
// 데이터 셋을 한다.(타겟값 안넣음)

df.head()
// 상위 5개 데이터 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/a8080cd9-3ac5-4de1-abc1-a1ee4b18cf16)

```python
iris['target']
// iris 타겟 값이 저장되어 있는것을 보여준다. 
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/02c9acf8-c24e-4178-990c-b446448a85a2)
### PCA
왜 PCA를 하는 것일까?

나중에 군집화 그래프를 만들기 위해서
```python
from sklearn.decomposition import PCA
// PCA 패키지를 가져온다.

pca = PCA(n_components=2)
// 특징을 2개로 만들 수 있는 PCA를 생성

data = pca.fit_transform(df)
// 특징을 2개로 변환시켜 넣는다.

data[:5]
// 0~4번째 데이터를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/14809d8f-d4fd-453d-828e-f260538320a6)
### 모델 정의, 학습
```python
from sklearn.cluster import KMeans
// KMeans 모델을 가져온다.

model = KMeans(n_clusters=3 [k와 동일])
// 3개의 군집으로 나누는 KMeans 모델을 정의한다.

model.fit(data)
// 모델 훈련

model.labels_
// 분류된 데이터가 어떤 라벨을 가지고 있는지 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/42a3505a-e32c-41bc-b022-6c9a2b6765cf)
### 군집화 결과 시각화
```python
import matplotlib.pyplot as plt
import seaborn as sns
// matplotlib 패키지와 seaborn 패키지를 가져온다.

colors = ['red', 'green', 'blue']
// 빨강 초록 파랑 색을 사용하여 구분한다.

plt.figure(figsize=(16, 10))
sns.scatterplot(
    x=data[:, 0],
    y=data[:, 1],
    hue=model.labels_,
    palette=colors,
    alpha=0.5,
)
// pca 결과를 라벨별로 색으로 그린다.

sns.scatterplot(
    x=model.cluster_centers_[:, 0],
    y=model.cluster_centers_[:, 1],
    hue=[0, 1, 2],
    palette=colors,
    s=300,
    legend=False
)
// 군집의 중심점에 300 사이즈 색깔 점을 찍는다.
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/5c758247-a6ec-4b45-84b0-59143c0fd1dc)

정답값 차트와 비교

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 8))
sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=iris['target'])
plt.show()
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/77b2d1e2-62b9-4eb0-b1f3-f8818766a842)

꽤 군집화가 잘되었다. (중심점의 거리가 가까우면 구분이 어렵다.)

정확도를 올리고 싶으면 feature의 갯수를 늘리거나 pca를 쓰지 말아야된다.
## 군집화 평가 지표
실루엣 점수 Silhouette Score

군집화가 얼마나 잘되었는지를 정량적으로 평가하는 지표

- a(i): 응집도 (Cohesion)
- b(i): 분리도 (separation)

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/6c2abef4-6107-4af8-801f-05d855bf05c3)

실루엣 점수

- 0: 군집화가 잘 되지 않았다
 - 군집 밀도가 낮고(응집도 ↑), 군집간 거리가 가깝다(분리도 ↓)
- 1: 군집화가 잘 되었다
 - 군집 밀도가 높고(응집도 ↓), 군집간 거리가 멀다(분리도 ↑)

```python
from sklearn.metrics import silhouette_score
// silhouette_score 패키지를 가져온다.

silhouette_score(data, model.labels_)
// 군집화의 결과를 평가해준다. (정답값을 넣지않는다)
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/dcd793ba-344b-468b-b8e3-177863a94a09)

군집화가 어느정도 잘 된걸 볼 수 있다.
## MNIST에서 K-Means 실습
### 데이터셋 로드
```python
from sklearn.datasets import load_digits
// load_digits 데이터셋 패키지를 가져온다.

digits = load_digits()
// 데이터를 넣는다.

data = digits['data']
target = digits['target']
// 데이터셋을 한다.
```
### 정규화
```python
from sklearn.preprocessing import MinMaxScaler
// 정규화 패키지를 가져온다.

scaler = MinMaxScaler()
// 정규화를 할 수 있는 객체를 생성한다.

scaled = scaler.fit_transform(data)
// 정규화를 시켜준 데이터를 넣어준다.

scaled[0]
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/9fbc8771-39a3-4a71-9222-bc9c7118f3be)

### PCA 수행
```python
from sklearn.decomposition import PCA
// pca 패키지를 가져온다.

pca = PCA(n_components=10)
// 특징 10개를 만들 수 있는 pca를 생성한다.

data_pca = pca.fit_transform(scaled)
// 특징을 10개로 변환시킨 데이터를 넣는다.

data_pca[0]
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/e1f6fc93-9ca8-4548-b9d8-239a3a9584ee)

### 모델 정의, 학습
```python
from sklearn.cluster import KMeans
// KMeans 모델을 가져온다.

model = KMeans(n_clusters=10)
// 10개의 군집으로 나누는 KMeans 모델을 정의한다.

model.fit(data_pca)
// 모델 훈련

model.labels_
// 분류된 데이터가 어떤 라벨을 가지고 있는지 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/05005543-709c-4130-a5b9-8aa0c981b837)
### 실루엣 검증
```python
from sklearn.metrics import silhouette_score
// silhouette_score 패키지를 가져온다.

silhouette_score(data_pca, model.labels_)
// 군집화의 결과를 평가해준다.
```

![image](https://github.com/hsy0511/bbang-hyung-5/assets/104752580/0379224d-427f-470c-8d28-6e82d44ef972)
