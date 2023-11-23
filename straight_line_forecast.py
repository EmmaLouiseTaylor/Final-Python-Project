#imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#Fonction
def forecast_straight_line(X, y, future_periods):
    # Reshape X to a column vector
    X = X.values.reshape(-1, 1)
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    # Generate future X values for forecasting
    future_X = np.arange(X.max() + 1, X.max() + 1 + future_periods).reshape(-1, 1)
    # Predict future y values
    future_y = model.predict(future_X)
    return future_X, future_y
#Apple's data from Statista
data = pd.DataFrame({
'Time in years': [2018, 2019, 2020, 2021, 2022],
'Value in billions of dollars': [266, 260, 275, 366, 394]
})
# Number of periods to forecast
future_periods = 3
future_X, future_y = forecast_straight_line(data['Time in years'], data['Value in billions of dollars'], future_periods)
# Plot of the original data and the forecast
plt.scatter(data['Time in years'], data['Value in billions of dollars'], label='Original Data')
plt.plot(future_X, future_y, label='Forecast', color='red', linestyle='dashed')
plt.xlabel('Time in years')
plt.ylabel('Value in billions of dollars')
plt.legend()
plt.show()

