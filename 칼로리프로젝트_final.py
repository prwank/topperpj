데이콘 Basic 칼로리 소모량 예측 프로젝트

#%%

## import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from random import randint, uniform
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import warnings

## 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 ggplot 스타일 사용
plt.style.use('ggplot')


## 그래프에서 마이너스 폰트 깨지는 문제 해결
mpl.rcParams['axes.unicode_minus'] = False

## warning 문구 삭제
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% 데이터

## 1. 데이터 불러오기
# train 데이터
train_data = pd.read_csv('C:/Data/project/calories/train.csv')
train_data.info()

# test 데이터
test_data = pd.read_csv('C:/Data/project/calories/test.csv')
test_data.info()

# submission
sub = pd.read_csv('C:/Data/project/calories/sample_submission.csv')
sub.info()


## 2. 데이터 확인
# 데이터 모양
train_data.shape
test_data.shape

# 데이터 정보
train_data.info()
test_data.info()

# 데이터 상위 값, 하위 값
train_data.head()
train_data.tail()
test_data.head()

# 결측치 확인
train_data.isnull().sum()
test_data.isnull().sum()


## 3. feature 내용
1.  Exercise_Duration         운동시간(분) : 1. ~ 30.0
2.  Body_Temperature(F)       체온 : 98.8 ~ 106.7 사이 45개 온도
3.  BPM                       심박수 : 69.0 ~ 128.0 사이 56개 심박수
4.  Height(Feet)              키(피트) : 4.0 ~ 7.0 사이 4개의 값
5.  Height(Remainder_Inches)  키(피트 계산 후 더해야 할 키) : 0. ~ 12. 사이 13개의 값
6.  Weight(lb)                몸무게(파운드) : 79.4 ~ 291.0 사이 88개의 값
7.  Weight_Status             체중 상태 : Normal Weight, Overweight, Obese
8.  Gender                    성별 : F, M
9.  Age                       나이 : 20 ~ 79 사이 60개 값
10. Calories_Burned           칼로리 소모량(목표 예측값) : 1.0 ~ 300.0 사이 270개의 값



#%% 데이터 변환


## 체중상태, 성별 인코딩

# 1. labelEncoding
le = LabelEncoder()

# train_data
train_data['Weight_Status']
Ws_le = le.fit_transform(train_data['Weight_Status']) 
train_data['Weight_Status'] = Ws_le

train_data['Gender']
gender_le = le.fit_transform(train_data['Gender']) 
le.classes_ # F:0, M:1
train_data['Gender'] = gender_le


# test_data
test_data['Weight_Status']
Ws_le = le.fit_transform(test_data['Weight_Status']) 
test_data['Weight_Status'] = Ws_le

test_data['Gender']
gender_le = le.fit_transform(test_data['Gender']) 
le.classes_ # F:0, M:1
test_data['Gender'] = gender_le


# 2. onehotEncoding
ohe = OneHotEncoder()

# train_data
train_data['Weight_Status']
Ws_ohe = ohe.fit_transform(train_data[['Weight_Status']]) 
Ws_ohe.toarray()
Ws_ohe = pd.DataFrame(Ws_ohe.toarray().astype(int))
Ws_ohe = Ws_ohe.rename(columns={0:'Nomal Weight', 1:'Obese', 2:'Overweight'})

train_data['Gender']
gender_ohe = ohe.fit_transform(train_data[['Gender']]) 
gender_ohe.toarray()
gender_ohe = pd.DataFrame(gender_ohe.toarray().astype(int))
gender_ohe = gender_ohe.rename(columns={0:'Male', 1:'Female'})

# test_data
test_data['Weight_Status']
Ws_ohe = ohe.fit_transform(test_data[['Weight_Status']]) 
Ws_ohe.toarray()
Ws_ohe = pd.DataFrame(Ws_ohe.toarray().astype(int))
Ws_ohe = Ws_ohe.rename(columns={0:'Nomal Weight', 1:'Obese', 2:'Overweight'})

test_data['Gender']
gender_ohe = ohe.fit_transform(test_data[['Gender']]) 
gender_ohe.toarray()
gender_ohe = pd.DataFrame(gender_ohe.toarray().astype(int))
gender_ohe = gender_ohe.rename(columns={0:'Male', 1:'Female'})


# 3. train_data, test_data 합치기

# train_data
train_data = pd.concat([train_data, Ws_ohe], axis=1)
del train_data['Weight_Status']
train_data.info()

train_data = pd.concat([train_data, gender_ohe], axis=1)
del train_data['Gender']
train_data.info()

# test_data
test_data = pd.concat([test_data, Ws_ohe], axis=1)
del test_data['Weight_Status']
test_data.info()

test_data = pd.concat([test_data, gender_ohe], axis=1)
del test_data['Gender']
test_data.info()


## ID 데이터 삭제, 타겟데이터 분리
# train_data
del train_data['ID']

y = train_data['Calories_Burned']
del train_data['Calories_Burned']

train_data.info()

# test_data
del test_data['ID']

test_data.info()


## 운동강도
train_data['Exercise_Intensity'] = train_data['BPM'] / (220 - train_data['Age'])
test_data['Exercise_Intensity'] = test_data['BPM'] / (220 - test_data['Age'])

## 키 Inch로 변경
train_data['Height(inch)'] = train_data['Height(Feet)'] * 12 + train_data['Height(Remainder_Inches)']
test_data['Height(inch)'] = test_data['Height(Feet)'] * 12 + test_data['Height(Remainder_Inches)']

del train_data['Height(Feet)']
del train_data['Height(Remainder_Inches)']

del test_data['Height(Feet)']
del test_data['Height(Remainder_Inches)']
train_data['Gender']

## BMR
train_data['BMR'] = 10 * train_data['Weight(lb)'] * 0.453592 + 6.25 * train_data['Height(inch)'] * 2.54 - 5 * train_data['Age'] + train_data['Gender'].apply(lambda x: 5 if x == 1 else -161)
test_data['BMR'] = 10 * test_data['Weight(lb)'] * 0.453592 + 6.25 * test_data['Height(inch)'] * 2.54 - 5 * test_data['Age'] + test_data['Gender'].apply(lambda x: 5 if x == 1 else -161)

## 화씨 → 섭씨로 변경
C_Temperature = []
for i in train_data.iloc[:,2]:
    C_Temperature.append((i-32) *5/9)
train_data['Body_Temperature(F)'] = C_Temperature
train_data = train_data.rename(columns = {'Body_Temperature(F)' : 'Body_Temperature(C)'})
train_data.info()

C_Temperature = []
for i in test_data.iloc[:,2]:
    C_Temperature.append((i-32) *5/9)
test_data['Body_Temperature(F)'] = C_Temperature
test_data = test_data.rename(columns = {'Body_Temperature(F)' : 'Body_Temperature(C)'})
test_data.info()

## 온도 log 취해주기
train_data['Age'] = np.log(train_data['Age'])
test_data['Age'] = np.log(test_data['Age'])

## 파일 저장
train_data.to_csv('C:/Data/project/calories/train_le_data.csv', index=False)
test_data.to_csv('C:/Data/project/calories/test_le_data.csv', index=False)

train_data.to_csv('C:/Data/project/calories/train_BMR_data.csv', index=False)
test_data.to_csv('C:/Data/project/calories/test_BMR_data.csv', index=False)

train_data.to_csv('C:/Data/project/calories/train_ohe_0420.csv', index=False)
test_data.to_csv('C:/Data/project/calories/test_ohe_0420.csv', index=False)

y.to_csv('C:/Data/project/calories/label_0420.csv', index=False)
train_data['Calories_Burned'] = label_data

#%% EDA

## 히트맵
sns.heatmap(data=train_data.corr(), annot=True, cmap="Blues")

mask = np.zeros_like(train_data.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(data=train_data.corr(), annot=True, cmap="Blues", mask=mask)

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))

sns.histplot(train_data['Exercise_Duration'], kde=True, ax=ax[0][0], color='red')
sns.histplot(train_data['Body_Temperature(F)'], kde=True, ax=ax[0][1], color='orange')
sns.histplot(train_data['BPM'], kde=True, ax=ax[0][2], color='yellow')
sns.histplot(train_data['Weight(lb)'], kde=True, ax=ax[0][3], color='green')
sns.boxplot(data=train_data, x='Weight_Status', y='Calories_Burned', ax=ax[0][4])
sns.swarmplot(data=train_data, x='Weight_Status', y='Calories_Burned', alpha=0.15, ax=ax[0][4])

sns.boxplot(data=train_data, x='Gender', y='Calories_Burned', ax=ax[1][0])
sns.stripplot(data=train_data, x='Gender', y='Calories_Burned', alpha=0.15, ax=ax[1][0])

sns.histplot(train_data['Age'], kde=True, ax=ax[1][1], color='blue')
sns.histplot(train_data['Exercise_Intensity'], kde=True, ax=ax[1][2], color='purple')
sns.histplot(train_data['Height(Feet)'], kde=True, ax=ax[1][3], color='indigo')
sns.histplot(train_data['BMR'], kde=True, ax=ax[1][4], color='grey')

sns.histplot(train_data['Age'], kde=True, color='blue')
sns.histplot(np.log(train_data['Age']), kde=True, color='blue')

#%% 모델링

연습파일
train_data = pd.read_csv('C:/Data/project/calories/train_le_data.csv')
label_data = pd.read_csv('C:/Data/project/calories/label_0420.csv')
test_data = pd.read_csv('C:/Data/project/calories/test_le_data.csv')

train_data = pd.read_csv('C:/Data/project/calories/train_BMR_data.csv')
test_data = pd.read_csv('C:/Data/project/calories/test_BMR_data.csv')

train_data = pd.read_csv('C:/Data/project/calories/train_ohe_0420.csv')
test_data = pd.read_csv('C:/Data/project/calories/test_ohe_0420.csv')


train_data.info()
test_data.info()



## 스케일링
# minmax
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
train_mms_scaled = mms.fit_transform(train_data)

# standard
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
train_std_scaled = std.fit_transform(train_data)


# train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_mms_scaled, label_data, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(train_std_scaled, label_data, test_size=0.2, random_state=42)

### 1. xgboost
xgb = XGBRegressor(learning_rate=0.01, max_depth=4, min_child_weight=3, n_estimators=10000, n_jobs=4)

xgb.fit(X_train, y_train, eval_metric='rmse', verbose=True)

predict = xgb.predict(X_test)

print(xgb.score(X_train, y_train))
print(explained_variance_score(predict, y_test))

mse = mean_squared_error(y_test, predict)
np.sqrt(mse)

predict = xgb.predict(test_data)
predict = xgb.predict(mms.transform(test_data))
predict = xgb.predict(std.transform(test_data))

sub['Calories_Burned'] = predict
sub.to_csv('C:/Data/project/calories/sample_submission0422_3.csv', index=False)


### 2. DecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
dct = DecisionTreeRegressor(random_state=42)
dct.fit(X_train, y_train)

predict = dct.predict(X_test)

accuracy = accuracy_score(y_test, predict)
print('DecisionTree 예측 값:', accuracy)

mse = mean_squared_error(y_test, predict)
np.sqrt(mse)

predict = xgb.predict(std.transform(test_data))

sub['Calories_Burned'] = predict
sub.to_csv('C:/Data/project/calories/sample_submission0422_2.csv', index=False)




### 3. SVR
from sklearn.svm import SVR

## 최적의 파라미터 찾아보자
svr_rbf = SVR()
param_svr = {'kernel' : ['rbf'],
             'C' : [1.0, 3.0, 10.0, 100.0, 300.0, 1000.0],
             'gamma' : [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
             'epsilon' : [0.1, 0.5, 1.0, 1.5, 2.0]
             }
rcv_rfr = GridSearchCV(svr_rbf, param_grid = param_svr,
                       cv=5, refit=True, n_jobs=-1, verbose=2)

rcv_rfr.fit(X_train, y_train)

rcv_rfr.best_params_


# SVM 회귀 모델 생성 및 학습
svr_rbf = SVR()
 
svr = SVR(kernel='rbf', C=1000.0, gamma=0.03, epsilon=0.1)
svr.fit(X_train, y_train)

predict = svr.predict(X_test)

mse = mean_squared_error(y_test, predict)
np.sqrt(mse)

predict = svr.predict(std.transform(test_data))

sub['Calories_Burned'] = predict
sub.to_csv('C:/Data/project/calories/sample_submission0425_1.csv', index=False)
# C (1.0) : 오류를 얼마나 허용할 것인가, 클수록 하드마진, 작을수록 소프트마진
# kernel : rbf(가우시안), linear(선형데이터셋), poly(비선형데이터셋)
# degree : 다항식 커널은 차수를 지정해줘야 한다
# gamma : 클수록 과적합 가능성 높아짐


