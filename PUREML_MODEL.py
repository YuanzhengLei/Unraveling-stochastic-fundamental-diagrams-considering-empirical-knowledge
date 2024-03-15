import matplotlib.pyplot as plt
import xlrd as xlrd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
x = []
y = []
x_small_sample = []
y_small_sample = []
#Case 1
data = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\364.xls')
table = data.sheet_by_index(0)

for i in range(1, table.nrows):
    x_small_sample.append(table.cell_value(i, 11) * 12/
                          (table.cell_value(i, 12) * 5))
    y_small_sample.append(table.cell_value(i, 12))
data1 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\365.xls')
table1 = data1.sheet_by_index(0)
for i in range(1, table1.nrows):
    x.append(table1.cell_value(i, 11) * 12 /
             (table1.cell_value(i, 12) * 5))
    y.append(table1.cell_value(i, 12))
data2 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\366.xls')
table2 = data2.sheet_by_index(0)
for i in range(1, table2.nrows):
    x.append(table2.cell_value(i, 11) * 12 /
                          (table2.cell_value(i, 12) * 5))
    y.append(table2.cell_value(i, 12))
data3 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\368.xls')
table3 = data3.sheet_by_index(0)
for i in range(1, table3.nrows):
    x.append(table3.cell_value(i, 9) * 12 /
             (table3.cell_value(i, 10) * 4))
    y.append(table3.cell_value(i, 10))
data4 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\374.xls')
table4 = data4.sheet_by_index(0)
for i in range(1, table4.nrows):
    x_small_sample.append(table4.cell_value(i, 9) * 12 /
                  (table4.cell_value(i, 10) * 4))
    y_small_sample.append(table4.cell_value(i, 10))
#Case 2

data5 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\375.xls')
table5 = data5.sheet_by_index(0)
for i in range(1, table5.nrows):
    x_small_sample.append(table5.cell_value(i, 11) * 12 /
             (table5.cell_value(i, 12) * 5))
    y_small_sample.append(table5.cell_value(i, 12))
data6 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\377.xls')
table6 = data6.sheet_by_index(0)
for i in range(1, table6.nrows):
    x.append(table6.cell_value(i, 9) * 12 /
                          (table6.cell_value(i, 10) * 5))
    y.append(table6.cell_value(i, 10))
data7 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\381.xls')
table7 = data7.sheet_by_index(0)
for i in range(1, table7.nrows):
    x.append( table7.cell_value(i, 9) * 12 /
              (table7.cell_value(i, 10) * 4))
    y.append(table7.cell_value(i, 10))
data8 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\384.xls')
table8 = data8.sheet_by_index(0)
for i in range(1, table8.nrows):
    x.append(table8.cell_value(i, 11) * 12 /
             (table8.cell_value(i, 12) * 5))
    y.append(table8.cell_value(i, 12))
data9 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\388.xls')
table9 = data9.sheet_by_index(0)
for i in range(1, table9.nrows):
    x.append(table9.cell_value(i, 9) * 12 /
             (table9.cell_value(i, 10) * 4))
    y.append(table9.cell_value(i, 10))
data10 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\389.xls')
table10 = data10.sheet_by_index(0)
for i in range(1, table10.nrows):
    x.append(table10.cell_value(i, 9) * 12 /
             (table10.cell_value(i, 10) * 4))
    y.append(table10.cell_value(i, 10))
data11 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\391.xls')
table11 = data11.sheet_by_index(0)
for i in range(1, table11.nrows):
    x.append(table11.cell_value(i, 11) * 12 /
             (table11.cell_value(i, 12) * 5))
    y.append(table11.cell_value(i, 12))

data12 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\393.xls')
table12 = data12.sheet_by_index(0)
for i in range(1, table12.nrows):
    x_small_sample.append(table12.cell_value(i, 11) * 12 /
             (table12.cell_value(i, 12) * 5))
    y_small_sample.append(table12.cell_value(i, 12))


X = np.array(x).reshape(-1, 1)
Y = np.array(y).reshape(-1, 1)
X_small_sample = np.array(x_small_sample).reshape(-1, 1)
Y_small_sample = np.array(y_small_sample).reshape(-1, 1)

sorted_indices = np.argsort(X, axis=0).flatten()
sorted_indices1 = np.argsort(X_small_sample, axis=0).flatten()
X_sorted = X[sorted_indices]
Y_sorted = Y[sorted_indices]
X_small_sample_sorted = X_small_sample[sorted_indices1]
Y_small_sample_sorted = Y_small_sample[sorted_indices1]
X_sorted = np.concatenate((X_sorted, X_small_sample_sorted), axis=0)
Y_sorted = np.concatenate((Y_sorted, Y_small_sample_sorted), axis=0)

r'''
interval = 0
for i in range(len(X_sorted) - 1):
    interval = interval + X_sorted[i + 1] - X_sorted[i]
interval = interval/(len(X_sorted) - 1)
#print(interval)
indices_to_remove = []

mark  = X_sorted[0]
for i in range(len(X_sorted) - 1):
    if X_sorted[i + 1] - X_sorted[i] >= interval:
        mark = X_sorted[i + 1]
    else:
        indices_to_remove.append(i + 1)

inducing_variable = np.delete(X_sorted, indices_to_remove)
inducing_variable = np.array(inducing_variable ).reshape(-1, 1)
#Y_sorted = np.delete(Y_sorted, indices_to_remove)
r'''


r'''
#k mean
n_inducing = math.floor(len(X_sorted) / 6)
inducing_variable, _ = kmeans(X_sorted, n_inducing)
inducing_variable_flattened = inducing_variable.ravel()
df0 = pd.DataFrame({'Index': inducing_variable_flattened})
df0.to_csv("Inducing_variable.csv", index=False)
r'''


df1 = pd.read_csv('Leve1_noise.csv')
X_noisy_1 = df1['Density (Veh/mile)'].values
Y_noisy_1 = df1['Speed (Mph)'].values
X_noisy_1 = X_noisy_1.reshape(-1, 1)
Y_noisy_1 = Y_noisy_1.reshape(-1, 1)

df2 = pd.read_csv('Leve2_noise.csv')
X_noisy_2 = df2['Density (Veh/mile)'].values
Y_noisy_2 = df2['Speed (Mph)'].values
X_noisy_2 = X_noisy_2.reshape(-1, 1)
Y_noisy_2 = Y_noisy_2.reshape(-1, 1)

df3 = pd.read_csv('Leve3_noise.csv')
X_noisy_3 = df3['Density (Veh/mile)'].values
Y_noisy_3 = df3['Speed (Mph)'].values
X_noisy_3 = X_noisy_3.reshape(-1, 1)
Y_noisy_3 = Y_noisy_3.reshape(-1, 1)


r'''
interval = 0
for i in range(len(X_small_sample_sorted) - 1):
    interval = interval + X_small_sample_sorted[i + 1] - X_small_sample_sorted[i]
interval = interval/(len(X_small_sample_sorted) - 1)
#print(interval)
indices_to_remove = []

mark  = X_small_sample_sorted[0]
for i in range(len(X_small_sample_sorted) - 1):
    if X_small_sample_sorted[i + 1] - X_small_sample_sorted[i] >= interval:
        mark = X_small_sample_sorted[i + 1]
    else:
        indices_to_remove.append(i + 1)

X_small_sample_sorted = np.delete(X_small_sample_sorted, indices_to_remove)
Y_small_sample_sorted = np.delete(Y_small_sample_sorted, indices_to_remove)

X_small_sample_sorted = np.array(X_small_sample_sorted).reshape(-1, 1)  # Reshape X to be a 2D array of shape [347, 1]
Y_small_sample_sorted = np.array(Y_small_sample_sorted).reshape(-1, 1)
r'''


x_test = []
y_test = []
#Case 1
data13 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\369.xls')
table13 = data13.sheet_by_index(0)
for i in range(1, table13.nrows):
    x_test.append(table13.cell_value(i, 9) * 12 /
                  (table13.cell_value(i, 10) * 4))
    y_test.append(table13.cell_value(i, 10))
data14 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\372.xls')
table14 = data14.sheet_by_index(0)
for i in range(1, table14.nrows):
    x_test.append( table14.cell_value(i, 11) * 12/
              (table14.cell_value(i, 12) * 5))
    y_test.append(table14.cell_value(i, 12))
#Case 2

data15 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\379.xls')
table15 = data15.sheet_by_index(0)
for i in range(1, table15.nrows):
    x_test.append(table15.cell_value(i, 11) * 12 /
                  (table15.cell_value(i, 12) * 5))
    y_test.append(table15.cell_value(i, 12))
data16 = xlrd.open_workbook(r'C:\Users\user\PycharmProjects\Data\386.xls')
table16 = data16.sheet_by_index(0)
for i in range(1, table16.nrows):
    x_test.append(table16.cell_value(i, 11) * 12 /
                  (table16.cell_value(i, 12) * 5))
    y_test.append(table16.cell_value(i, 12))

X_test = np.array(x_test).reshape(-1, 1)
Y_test = np.array(y_test).reshape(-1, 1)

sorted_indices2 = np.argsort(X_test, axis=0).flatten()
X_test_sorted = X_test[sorted_indices2]
Y_test_sorted = Y_test[sorted_indices2]


svr = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Fit the model on the training data
#svr.fit(X_small_sample_sorted, Y_small_sample_sorted)
svr.fit(X_sorted, Y_sorted)

# Predict the Y values for X_test_sorted
Y_pred_svr = svr.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_svr = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_svr))
mape_svr = mean_absolute_percentage_error(Y_test_sorted, Y_pred_svr)  * 100


print("Support Vector Machine")
print(f"RMSE: {rmse_svr:.4f}")
print(f"MAPE: {mape_svr:.4f}%")



svr_noisy_1 = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Fit the model on the training data
#svr.fit(X_small_sample_sorted, Y_small_sample_sorted)
svr_noisy_1.fit(X_noisy_1, Y_noisy_1)

# Predict the Y values for X_test_sorted
Y_pred_svr_noisy_1 = svr_noisy_1.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_svr_noisy_1 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_svr_noisy_1))
mape_svr_noisy_1 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_svr_noisy_1)  * 100


print("Support Vector Machine with 30% noise")
print(f"RMSE: {rmse_svr_noisy_1:.4f}")
print(f"MAPE: {mape_svr_noisy_1:.4f}%")

svr_noisy_3 = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Fit the model on the training data
#svr.fit(X_small_sample_sorted, Y_small_sample_sorted)
svr_noisy_3.fit(X_noisy_2, Y_noisy_2)

# Predict the Y values for X_test_sorted
Y_pred_svr_noisy_3 = svr_noisy_3.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_svr_noisy_3 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_svr_noisy_3))
mape_svr_noisy_3 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_svr_noisy_3)  * 100


print("Support Vector Machine with 50% noise")
print(f"RMSE: {rmse_svr_noisy_3:.4f}")
print(f"MAPE: {mape_svr_noisy_3:.4f}%")

svr_noisy_5 = SVR(kernel='rbf', C=1e3, gamma=0.1)

# Fit the model on the training data
#svr.fit(X_small_sample_sorted, Y_small_sample_sorted)
svr_noisy_5.fit(X_noisy_3, Y_noisy_3)

# Predict the Y values for X_test_sorted
Y_pred_svr_noisy_5 = svr_noisy_5.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_svr_noisy_5 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_svr_noisy_5))
mape_svr_noisy_5 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_svr_noisy_5)  * 100


print("Support Vector Machine with 80% noise")
print(f"RMSE: {rmse_svr_noisy_5:.4f}")
print(f"MAPE: {mape_svr_noisy_5:.4f}%")

mlp = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=1000)

# Fit the MLP model on the training data
#mlp.fit(X_small_sample_sorted, Y_small_sample_sorted)
mlp.fit(X_sorted, Y_sorted)

# Predict the Y values for X_test_sorted
Y_pred_mlp = mlp.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_mlp = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_mlp))
mape_mlp = mean_absolute_percentage_error(Y_test_sorted, Y_pred_mlp)  * 100

print("Multilayer Perceptron")
print(f"RMSE: {rmse_mlp:.4f}")
print(f"MAPE: {mape_mlp:.4f}%")


mlp_noisy_1 = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=1000)
# Fit the model on the training data
#mlp.fit(X_small_sample_sorted, Y_small_sample_sorted)
mlp_noisy_1.fit(X_noisy_1, Y_noisy_1)

# Predict the Y values for X_test_sorted
Y_pred_mlp_noisy_1 = mlp_noisy_1.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_mlp_noisy_1 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_mlp_noisy_1))
mape_mlp_noisy_1 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_mlp_noisy_1)  * 100


print("Multilayer Perceptron with 30% noise")
print(f"RMSE: {rmse_mlp_noisy_1:.4f}")
print(f"MAPE: {mape_mlp_noisy_1:.4f}%")

mlp_noisy_3 = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=1000)

# Fit the model on the training data
#mlp.fit(X_small_sample_sorted, Y_small_sample_sorted)
mlp_noisy_3.fit(X_noisy_2, Y_noisy_2)

# Predict the Y values for X_test_sorted
Y_pred_mlp_noisy_3 = mlp_noisy_3.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_mlp_noisy_3 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_mlp_noisy_3))
mape_mlp_noisy_3 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_mlp_noisy_3)  * 100


print("Multilayer Perceptron with 50% noise")
print(f"RMSE: {rmse_mlp_noisy_3:.4f}")
print(f"MAPE: {mape_mlp_noisy_3:.4f}%")

mlp_noisy_5 = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', max_iter=1000)

# Fit the model on the training data
#mlp.fit(X_small_sample_sorted, Y_small_sample_sorted)
mlp_noisy_5.fit(X_noisy_3, Y_noisy_3)

# Predict the Y values for X_test_sorted
Y_pred_mlp_noisy_5 = mlp_noisy_5.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_mlp_noisy_5 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_mlp_noisy_5))
mape_mlp_noisy_5 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_mlp_noisy_5)  * 100


print("Multilayer Perceptron with 80% noise")
print(f"RMSE: {rmse_mlp_noisy_5:.4f}")
print(f"MAPE: {mape_mlp_noisy_5:.4f}%")

rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the Random Forest model on the training data
#rf.fit(X_small_sample_sorted, Y_small_sample_sorted)
rf.fit(X_sorted, Y_sorted)

# Predict the Y values for X_test_sorted
Y_pred_rf = rf.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_rf = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_rf))
mape_rf = mean_absolute_percentage_error(Y_test_sorted, Y_pred_rf) * 100

print("Random Forest")
print(f'RMSE: {rmse_rf:.4f}')
print(f'MAPE: {mape_rf:.4f}%')

rf_noisy_1 = RandomForestRegressor(n_estimators=100, random_state=42)
# Fit the model on the training data
#rf.fit(X_small_sample_sorted, Y_small_sample_sorted)
rf_noisy_1.fit(X_noisy_1, Y_noisy_1)

# Predict the Y values for X_test_sorted
Y_pred_rf_noisy_1 = rf_noisy_1.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_rf_noisy_1 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_rf_noisy_1))
mape_rf_noisy_1 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_rf_noisy_1)  * 100


print("Random Forest with 30% noise")
print(f"RMSE: {rmse_rf_noisy_1:.4f}")
print(f"MAPE: {mape_rf_noisy_1:.4f}%")

rf_noisy_3 = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
#rf.fit(X_small_sample_sorted, Y_small_sample_sorted)
rf_noisy_3.fit(X_noisy_2, Y_noisy_2)

# Predict the Y values for X_test_sorted
Y_pred_rf_noisy_3 = rf_noisy_3.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_rf_noisy_3 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_rf_noisy_3))
mape_rf_noisy_3 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_rf_noisy_3)  * 100


print("Random Forest with 50% noise")
print(f"RMSE: {rmse_rf_noisy_3:.4f}")
print(f"MAPE: {mape_rf_noisy_3:.4f}%")

rf_noisy_5 = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
#rf.fit(X_small_sample_sorted, Y_small_sample_sorted)
rf_noisy_5.fit(X_noisy_3, Y_noisy_3)

# Predict the Y values for X_test_sorted
Y_pred_rf_noisy_5 = rf_noisy_5.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_rf_noisy_5 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_rf_noisy_5))
mape_rf_noisy_5 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_rf_noisy_5)  * 100


print("Random Forest with 80% noise")
print(f"RMSE: {rmse_rf_noisy_5:.4f}")
print(f"MAPE: {mape_rf_noisy_5:.4f}%")


# Initialize XGBRegressor
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror',
                             colsample_bytree = 0.3,
                             learning_rate = 0.1,
                             max_depth = 5,
                             alpha = 10,
                             n_estimators = 100)

# Fit the model
#xgb_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
xgb_model.fit(X_sorted, Y_sorted)
# Predict
Y_pred_xgb = xgb_model.predict(X_test_sorted)

# Calculate RMSE and MAPE
rmse_xgb = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_xgb))
mape_xgb = mean_absolute_percentage_error(Y_test_sorted, Y_pred_xgb) * 100

print("XGBoost")
print(f'RMSE: {rmse_xgb:.4f}')
print(f'MAPE: {mape_xgb:.4f}%')

xgb_model_noisy_1 = xgb.XGBRegressor(objective ='reg:squarederror',
                             colsample_bytree = 0.3,
                             learning_rate = 0.1,
                             max_depth = 5,
                             alpha = 10,
                             n_estimators = 100)
# Fit the model on the training data
#xgb_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
xgb_model_noisy_1.fit(X_noisy_1, Y_noisy_1)

# Predict the Y values for X_test_sorted
Y_pred_xgb_model_noisy_1 = xgb_model_noisy_1.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_xgb_model_noisy_1 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_xgb_model_noisy_1))
mape_xgb_model_noisy_1 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_xgb_model_noisy_1)  * 100


print("XGBoost with 30% noise")
print(f"RMSE: {rmse_xgb_model_noisy_1:.4f}")
print(f"MAPE: {mape_xgb_model_noisy_1:.4f}%")

xgb_model_noisy_3 = xgb.XGBRegressor(objective ='reg:squarederror',
                             colsample_bytree = 0.3,
                             learning_rate = 0.1,
                             max_depth = 5,
                             alpha = 10,
                             n_estimators = 100)

# Fit the model on the training data
#xgb_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
xgb_model_noisy_3.fit(X_noisy_2, Y_noisy_2)

# Predict the Y values for X_test_sorted
Y_pred_xgb_model_noisy_3 = xgb_model_noisy_3.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_xgb_model_noisy_3 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_xgb_model_noisy_3))
mape_xgb_model_noisy_3 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_xgb_model_noisy_3)  * 100


print("XGBoost with 50% noise")
print(f"RMSE: {rmse_xgb_model_noisy_3:.4f}")
print(f"MAPE: {mape_xgb_model_noisy_3:.4f}%")

xgb_model_noisy_5 = xgb.XGBRegressor(objective ='reg:squarederror',
                             colsample_bytree = 0.3,
                             learning_rate = 0.1,
                             max_depth = 5,
                             alpha = 10,
                             n_estimators = 100)

# Fit the model on the training data
#xgb_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
xgb_model_noisy_5.fit(X_noisy_3, Y_noisy_3)

# Predict the Y values for X_test_sorted
Y_pred_xgb_model_noisy_5 = xgb_model_noisy_5.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_xgb_model_noisy_5 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_xgb_model_noisy_5))
mape_xgb_model_noisy_5 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_xgb_model_noisy_5)  * 100


print("XGBoost with 80% noise")
print(f"RMSE: {rmse_xgb_model_noisy_5:.4f}")
print(f"MAPE: {mape_xgb_model_noisy_5:.4f}%")

# Initialize the GradientBoostingRegressor model
gbdt_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=0, loss='squared_error')

# Fit the model on the training data
#gbdt_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
gbdt_model.fit(X_sorted, Y_sorted)

# Predict on the test data
Y_pred_gbdt = gbdt_model.predict(X_test_sorted)

# Calculate the Root Mean Square Error (RMSE)
rmse_gbdt = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_gbdt))

# Calculate the Mean Absolute Percentage Error (MAPE)
mape_gbdt = mean_absolute_percentage_error(Y_test_sorted, Y_pred_gbdt) * 100

# Print the performance metrics
print("Gradient Boosting Decision Tree")
print(f'RMSE: {rmse_gbdt:.4f}')
print(f'MAPE: {mape_gbdt:.4f}%')

gbdt_model_noisy_1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=0, loss='squared_error')
# Fit the model on the training data
#gbdt_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
gbdt_model_noisy_1.fit(X_noisy_1, Y_noisy_1)

# Predict the Y values for X_test_sorted
Y_pred_gbdt_model_noisy_1 = gbdt_model_noisy_1.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_gbdt_model_noisy_1 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_gbdt_model_noisy_1))
mape_gbdt_model_noisy_1 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_gbdt_model_noisy_1)  * 100


print("Gradient Boosting Decision Tree with 30% noise")
print(f"RMSE: {rmse_gbdt_model_noisy_1:.4f}")
print(f"MAPE: {mape_gbdt_model_noisy_1:.4f}%")

gbdt_model_noisy_3 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=0, loss='squared_error')

# Fit the model on the training data
#gbdt_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
gbdt_model_noisy_3.fit(X_noisy_2, Y_noisy_2)

# Predict the Y values for X_test_sorted
Y_pred_gbdt_model_noisy_3 = gbdt_model_noisy_3.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_gbdt_model_noisy_3 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_gbdt_model_noisy_3))
mape_gbdt_model_noisy_3 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_gbdt_model_noisy_3)  * 100


print("Gradient Boosting Decision Tree with 50% noise")
print(f"RMSE: {rmse_gbdt_model_noisy_3:.4f}")
print(f"MAPE: {mape_gbdt_model_noisy_3:.4f}%")

gbdt_model_noisy_5 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=0, loss='squared_error')

# Fit the model on the training data
#gbdt_model.fit(X_small_sample_sorted, Y_small_sample_sorted)
gbdt_model_noisy_5.fit(X_noisy_3, Y_noisy_3)

# Predict the Y values for X_test_sorted
Y_pred_gbdt_model_noisy_5 = gbdt_model_noisy_5.predict(X_test_sorted)

# Calculate the RMSE and MAPE
rmse_gbdt_model_noisy_5 = np.sqrt(mean_squared_error(Y_test_sorted, Y_pred_gbdt_model_noisy_5))
mape_gbdt_model_noisy_5 = mean_absolute_percentage_error(Y_test_sorted, Y_pred_gbdt_model_noisy_5)  * 100


print("Gradient Boosting Decision Tree with 80% noise")
print(f"RMSE: {rmse_gbdt_model_noisy_5:.4f}")
print(f"MAPE: {mape_gbdt_model_noisy_5:.4f}%")

