import numpy as np

x1=np.array([range(100), range(311,411), range(100)])
x2=np.array([range(101,201), range(311,411), range(101,201)])
y=np.array([range(501,601)])
print(x1.shape) # (3,100)
print(x2.shape)
print(y.shape) # (1,100)

x1 = np.transpose(x1) # (100,3)
x2 = np.transpose(x2)
y = np.transpose(y) # (100,1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y_train, y_test = train_test_split(x1,y,random_state=66, test_size=0.4, shuffle=False)
x1_val, x1_test, y_val, y_test = train_test_split(x1_test,y_test,random_state=66, test_size=0.5, shuffle=False)
x2_train, x2_test = train_test_split(x2, random_state=66, test_size=0.4, shuffle=False)
x2_val, x2_test = train_test_split(x2_test, random_state=66, test_size=0.5, shuffle=False)

# 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
input1 = Input(shape=(3,))
dense1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

input2 = Input(shape=(3,))
dense2 = Dense(50, activation='relu')(input2)
dense2_2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])

model1 = Dense(10)(merge1)
model2 = Dense(5)(model1)
output = Dense(1)(model2)

model = Model(inputs= [input1, input2], outputs=output)
model.summary()


# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],y_train, epochs=100, batch_size=1, validation_data=([x1_val, x2_val],y_val))

# 평가예측
mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print("mse: ",mse)

y_predict = model.predict([x1_test,x2_test])
for i in range(len(y_predict)):
    print(y_test[i], y_predict[i])
    
'''
mse:  [94.44895935058594, 94.44895935058594]
[581] [587.04443]
[582] [588.399]
[583] [589.74445]
[584] [591.08984]
[585] [592.4422]
[586] [593.8118]
[587] [595.1814]
[588] [596.551]
[589] [597.9206]
[590] [599.2902]
[591] [600.6599]
[592] [602.0295]
[593] [603.39905]
[594] [604.7686]
[595] [606.1383]
[596] [607.5079]
[597] [608.8775]
[598] [610.247]
[599] [611.61664]
[600] [612.9863]

예측값과 약간의 오차가 있지만, 큰 문제는 아님 -> x1,x2 으로 전혀다른 501~600 을 예측하기 때문
'''
