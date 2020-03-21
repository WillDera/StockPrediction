# Building a model that predicts GOOGLE Stock Prices for a specific day
# Using SVR and Linear Regression

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('Dataset/Historical.csv')
df.head(7)

# Create the list / X and Y datasets

dates = []
prices = []

# Get the number of rows and columns in a dataset
df.shape

# Get the last row of data (this would be the data that we'd test our model on)
df.tail

# Get the data except for the last row
df = df.head(len(df)-1)

# New shape of the data
df.shape

# Getting all the rows from the date column
df_dates = df.loc[:, 'Date']

# Getting all the rows from the open column
df_open = df.loc[:, 'Close']

# Create the independent dataset X
for date in df_dates:
    dates.append([int(date)])

# Create the dependent dataset Y
for open_price in df_open:
    prices.append(float(open_price))

# See what days were recorded
# print(dates)


def predict_prices(dates, prices, x):
    # Create the 3 SVR models
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # Train the SVR models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    # Create the Linear regression model
    lin_reg = LinearRegression()

    # Train the LinearRegression model
    lin_reg.fit(dates, prices)

    # Plot the models on a graph to see which has the best fit
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='SVR_RBF')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='SVR_POLY')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='SVR_LIN')
    plt.plot(dates, lin_reg.predict(dates), color='orange', label='LIN_REG')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_poly.predict(x)[0], svr_lin.predict(x)[0], lin_reg.predict(x)[0]


# Predict the price of day 2017-01-17 = 830.00
predicted_price = predict_prices(dates, prices, [[176]])
print(predicted_price)
