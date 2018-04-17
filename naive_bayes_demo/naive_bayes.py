from sklearn.naive_bayes import GaussianNB
import numpy as np

# assigning predictor and target variables
# score of TOFEL and GRE
x= np.array([[100,341], [80,300], [90,330], [104,289], [110,297], [72,120], [88,296], [101,309], [86,302], [97,305], [110,293], [85,295]])
# 0 reprents won't like, 1 represents will like
y = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0])

# Create a Gaussian Classifer
model = GaussianNB()

# Train the model
model.fit(x, y)

# Predict
# test cases:
# 1. TOFEL: 100 GRE: 320
# 2. TOFEL: 80 GRE: 308
predict = model.predict([[100, 320],[80, 308]])
print("Result: ", predict)


