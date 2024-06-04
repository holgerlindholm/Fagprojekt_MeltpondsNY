import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1.1, 1.9, 10, 4.1, 6, 6.1, 7.0, 8.0, 9.1, 12])

# Fit a linear model (as an example of a smooth curve)
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
y_pred = model.predict(x.reshape(-1, 1))

# Calculate residuals
residuals = y - y_pred

# Method 1: Standard Deviation Method
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
outliers_std_method = np.abs(residuals - mean_residual) > 2 * std_residual

# Method 2: IQR Method
Q1 = np.percentile(residuals, 25)
Q3 = np.percentile(residuals, 75)
IQR = Q3 - Q1
outliers_iqr_method = (residuals < Q1 - 1.5 * IQR) | (residuals > Q3 + 1.5 * IQR)

# Plot data, fitted curve, and outliers
plt.scatter(x, y, label='Data points')
plt.plot(x, y_pred, color='red', label='Fitted curve')
plt.scatter(x[outliers_std_method], y[outliers_std_method], color='orange', label='Outliers (Std method)')
plt.scatter(x[outliers_iqr_method], y[outliers_iqr_method], color='green', label='Outliers (IQR method)')
plt.legend()
plt.show()
