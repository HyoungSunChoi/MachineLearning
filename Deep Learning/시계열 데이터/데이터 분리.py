import numpy as np
x=np.array(range(1,101))
y=np.array(range(1,101))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=66, test_size=0.4, shuffle=False)
# test_size 를 40%, train_size는 60%
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, random_state=66, test_size=0.5)
# train, test, val -> 6: 2 : 2

x_predict=np.array(range(101,111))
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(5, input_shape=(1,), activation='relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val,y_val))

mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ",mse)

y_predict=model.predict(x_predict)
print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test,y_predict))

# R 2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ",r2_y_predict)

