#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Currency pair exchange rates for CAD/JPY
cad_jpy_df = pd.read_csv(
    Path("cad_jpy.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
cad_jpy_df.head()


# In[3]:


# Trim the dataset to begin on January 1st, 1990
cad_jpy_df = cad_jpy_df.loc["1990-01-01":, :]
cad_jpy_df.head()


# In[4]:


# Create a series using "Price" percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s

cad_jpy_df['Return'] = (cad_jpy_df[["Price"]].pct_change() * 100)
returns = cad_jpy_df.replace(-np.inf, np.nan).dropna()
returns.tail()


# In[5]:


# Create a lagged return using the shift function
cad_jpy_df['Lagged_Return'] = cad_jpy_df["Return"].shift()
cad_jpy_df = cad_jpy_df.dropna()
cad_jpy_df.tail()


# In[6]:


# Create a train/test split for the data using 2018-2019 for testing and the rest for training
train = cad_jpy_df[:'2018']
test = cad_jpy_df['2018':]


# In[7]:


# Create four dataframes:
# X_train (training set using just the independent variables), X_test (test set of of just the independent variables)
# Y_train (training set using just the "y" variable, i.e., "Futures Return"), Y_test (test set of just the "y" variable):

x_train = train["Lagged_Return"].to_frame()
y_train = train["Return"]
x_test = test["Lagged_Return"].to_frame()
y_test = test["Return"]


# In[8]:


# Create a Linear Regression model and fit it to the training data
from sklearn.linear_model import LinearRegression

# Fit a SKLearn linear regression using just the training set (X_train, Y_train):
model = LinearRegression()
model.fit(x_train, y_train)


# In[10]:


# Make a prediction of "y" values using just the test dataset
predictions = model.predict(x_test)


# In[11]:


# Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
results = y_test.to_frame()
results["Predicted Return"] = predictions
results.head(2)


# In[12]:


# Plot the first 20 predictions vs the true values
results[:20].plot(subplots=True, figsize = (10, 10))


# In[13]:


from sklearn.metrics import mean_squared_error
# Calculate the mean_squared_error (MSE) on actual versus predicted test "y" 
mse = mean_squared_error(
    results["Return"],
    results["Predicted Return"]
)

# Using that mean-squared-error, calculate the root-mean-squared error (RMSE):
rmse = np.sqrt(mse)
print(f"Out-of-Sample Root Mean Squared Error (RMSE): {rmse}")


# In[14]:


# Construct a dataframe using just the "y" training data:
in_sample_results = y_train.to_frame()

# Add a column of "in-sample" predictions to that dataframe:  
in_sample_results["In-sample Predictions"] = model.predict(x_train)

# Calculate in-sample mean_squared_error (for comparison to out-of-sample)
in_sample_mse = mean_squared_error(
    in_sample_results["Return"],
    in_sample_results["In-sample Predictions"]
)

# Calculate in-sample root mean_squared_error (for comparison to out-of-sample)
in_sample_rmse = np.sqrt(in_sample_mse)
print(f"In-sample Root Mean Squared Error (RMSE): {in_sample_rmse}")


# In[ ]:




