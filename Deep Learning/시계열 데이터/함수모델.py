import numpy as np
x=np.array(range(1,101))
y=np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False)

from keras.models import Sequential, Model
from keras.layers import Dense, Input
input = Input(shape=(1,))
dense1= Dense(5, activation='relu')(input)
dense2= Dense(3)(dense1)
dense3= Dense(4)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs=input, outputs=output1)
model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val,y_val))

# 평가 예측
mse = model.evaluate(x_test, y_test, batch_size=3)
print("mse : ",mse)
y_predict= model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)