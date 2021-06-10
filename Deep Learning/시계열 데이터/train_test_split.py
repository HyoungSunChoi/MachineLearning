from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state=66, test_size=0.4, shuffle=False)
# test_size 를 40%, shu