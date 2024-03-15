import math
import random
from openpyxl import load_workbook
from check_shapes import inherit_check_shapes, check_shapes
from gpflow.kernels import Stationary, IsotropicStationary
from gpflow.kernels.base import ActiveDims,Kernel
from gpflow.utilities.ops import square_distance, difference_matrix
from scipy.cluster.vq import kmeans
import numpy as np
import gpflow
from gpflow.utilities import positive, print_summary
from gpflow.base import Parameter, TensorType
import xlrd as xlrd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from matplotlib import cm
import EMPIRICAL_MODEL
import os
from typing import Any, Optional


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15*1024)])
    except RuntimeError as e:
        print(e)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 16})
x = []
y = []

with open(r'C:\Users\user\PycharmProjects\Data\GA400.txt', 'r') as file:
    for line in file:
        # Split the line into parts
        parts = line.split()
        # Append data to respective lists
        if len(parts) >= 3:
            x.append(float(parts[1])/1.609344)
            y.append(float(parts[2])/1.609344)



X = np.array(x).reshape(-1, 1)
Y = np.array(y).reshape(-1, 1)

sorted_indices = np.argsort(X, axis=0).flatten()
X_sorted = X[sorted_indices]
Y_sorted = Y[sorted_indices]

X_test_sorted = X_sorted.copy()
Y_test_sorted = Y_sorted.copy()



#Cluster sampling
np.random.seed(1234)
n_inducing = 288
inducing_variable, _ = kmeans(X_sorted, n_inducing)
inducing_variable_flattened = inducing_variable.ravel()
df0 = pd.DataFrame({'Index': inducing_variable_flattened})
df0.to_csv("Inducing_variable.csv", index=False)


r'''
#simple random sampling
rng = np.random.default_rng(1234)
n_inducing = 288
inducing_variable = rng.choice(X_sorted, size=n_inducing, replace=False)
indices = np.array([np.where(np.all(X_sorted == iv, axis=1))[0][0] for iv in inducing_variable])
inducing_y = Y_sorted[indices]
inducing_variable_flattened = inducing_variable.ravel()
inducing_y_flattened = inducing_y.ravel()
df0 = pd.DataFrame({'Index': inducing_variable_flattened})
df0.to_csv("Inducing_variable.csv", index=False)
r'''

r'''
#systematic sampling
rng = np.random.default_rng(1234)
n_inducing = 288
K = math.floor(len(X_sorted)/n_inducing)
initial_pos = random.randint(1, K)
inducing_variable = X_sorted[initial_pos::K]
indices = np.array([np.where(np.all(X_sorted == iv, axis=1))[0][0] for iv in inducing_variable])
inducing_y = Y_sorted[indices]
inducing_variable_flattened = inducing_variable.ravel()
inducing_y_flattened = inducing_y.ravel()
df0 = pd.DataFrame({'Index': inducing_variable_flattened})
df0.to_csv("Inducing_variable.csv", index=False)
r'''

r'''
#weighted random sampling
# Assuming X_sorted is your dataset and probabilities is a list of selection probabilities for each point
df1 = pd.read_csv(r'C:\Users\user\PycharmProjects\WLSM\Weights.csv')
weights = df1['Weights'].values
total_weight = sum(weights)
probabilities = np.array([weights])  # Replace [...] with your probabilities
probabilities = probabilities.flatten()
probabilities = probabilities/total_weight
n = 288 # Number of samples you want to select
# Normalize probabilities if they don't sum to 1
probabilities /= probabilities.sum()
# Select n samples according to their probabilities
np.random.seed(1234)
selected_indices = np.random.choice(len(X_sorted), size=n, replace=False, p=probabilities)
inducing_variable = X_sorted[selected_indices]
indices = np.array([np.where(np.all(X_sorted == iv, axis=1))[0][0] for iv in inducing_variable])
inducing_y = Y_sorted[indices]
inducing_variable_flattened = inducing_variable.ravel()
inducing_y_flattened = inducing_y.ravel()
r'''

model_MacNicholas = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.MacNicholasMeanFunction(),
    inducing_variable = inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_MacNicholas.training_loss, model_MacNicholas.trainable_variables)

# Get the Estimations from the model
mean_MacNicholas, _ = model_MacNicholas.predict_y(X_test_sorted)

# Calculate the errors
errors_MacNicholas = mean_MacNicholas - Y_test_sorted

# Calculate RMSE
rmse_MacNicholas = tf.sqrt(tf.reduce_mean(tf.square(errors_MacNicholas)))

# Calculate MAPE
mape_MacNicholas = tf.reduce_mean(tf.abs(errors_MacNicholas / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_MacNicholas = rmse_MacNicholas.numpy()
mape_result_MacNicholas = mape_MacNicholas.numpy()

print("MacNicholas Model")
print(f"RMSE: {rmse_result_MacNicholas:.4f}")
print(f"MAPE: {mape_result_MacNicholas:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_MacNicholas.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")

model_PureGP = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_PureGP.training_loss, model_PureGP.trainable_variables)

# Get the Estimations from the model
mean_PureGP, _ = model_PureGP.predict_y(X_test_sorted)

# Calculate the errors
errors_PureGP = mean_PureGP - Y_test_sorted

# Calculate RMSE
rmse_PureGP = tf.sqrt(tf.reduce_mean(tf.square(errors_PureGP)))

# Calculate MAPE
mape_PureGP = tf.reduce_mean(tf.abs(errors_PureGP / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_PureGP = rmse_PureGP.numpy()
mape_result_PureGP = mape_PureGP.numpy()

print("PureGP Model")
print(f"RMSE: {rmse_result_PureGP:.4f}")
print(f"MAPE: {mape_result_PureGP:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_PureGP.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")

model_Cheng = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.ChengMeanFunction(),
    inducing_variable = inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Cheng.training_loss, model_Cheng.trainable_variables)

# Get the Estimations from the model
mean_Cheng, _ = model_Cheng.predict_y(X_test_sorted)

# Calculate the errors
errors_Cheng = mean_Cheng - Y_test_sorted

# Calculate RMSE
rmse_Cheng = tf.sqrt(tf.reduce_mean(tf.square(errors_Cheng)))

# Calculate MAPE
mape_Cheng = tf.reduce_mean(tf.abs(errors_Cheng / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Cheng = rmse_Cheng.numpy()
mape_result_Cheng = mape_Cheng.numpy()

print("Cheng Model")
print(f"RMSE: {rmse_result_Cheng:.4f}")
print(f"MAPE: {mape_result_Cheng:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Cheng.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")


model_Wang = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.WangMeanFunction(),
    inducing_variable = inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Wang.training_loss, model_Wang.trainable_variables)

# Get the Estimations from the model
mean_Wang, _ = model_Wang.predict_y(X_test_sorted)

# Calculate the errors
errors_Wang = mean_Wang - Y_test_sorted

# Calculate RMSE
rmse_Wang = tf.sqrt(tf.reduce_mean(tf.square(errors_Wang)))

# Calculate MAPE
mape_Wang = tf.reduce_mean(tf.abs(errors_Wang / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Wang = rmse_Wang.numpy()
mape_result_Wang = mape_Wang.numpy()

print("Wang Model")
print(f"RMSE: {rmse_result_Wang:.4f}")
print(f"MAPE: {mape_result_Wang:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Wang.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")



model_Greenshields = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.GreenshieldsMeanFunction(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Greenshields.training_loss, model_Greenshields.trainable_variables)

# Get the Estimations from the model
mean_Greenshields, _ = model_Greenshields.predict_y(X_test_sorted)

# Calculate the errors
errors_Greenshields = mean_Greenshields - Y_test_sorted

# Calculate RMSE
rmse_Greenshields = tf.sqrt(tf.reduce_mean(tf.square(errors_Greenshields)))

# Calculate MAPE
mape_Greenshields = tf.reduce_mean(tf.abs(errors_Greenshields / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Greenshields = rmse_Greenshields.numpy()
mape_result_Greenshields = mape_Greenshields.numpy()

print("Greenshields Model")
print(f"RMSE: {rmse_result_Greenshields:.4f}")
print(f"MAPE: {mape_result_Greenshields:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Greenshields.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")

model_Greenberg = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.GreenbergMeanFunction(),
    inducing_variable = inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Greenberg.training_loss, model_Greenberg.trainable_variables)

# Get the Estimations from the model
mean_Greenberg, _ = model_Greenberg.predict_y(X_test_sorted)

# Calculate the errors
errors_Greenberg = mean_Greenberg - Y_test_sorted

# Calculate RMSE
rmse_Greenberg = tf.sqrt(tf.reduce_mean(tf.square(errors_Greenberg)))

# Calculate MAPE
mape_Greenberg = tf.reduce_mean(tf.abs(errors_Greenberg / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Greenberg = rmse_Greenberg.numpy()
mape_result_Greenberg = mape_Greenberg.numpy()

print("Greenberg Model")
print(f"RMSE: {rmse_result_Greenberg:.4f}")
print(f"MAPE: {mape_result_Greenberg:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Greenberg.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")


model_Underwood = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.UnderwoodMeanFunction(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Underwood.training_loss, model_Underwood.trainable_variables)

# Get the Estimations from the model
mean_Underwood, _ = model_Underwood.predict_y(X_test_sorted)

# Calculate the errors
errors_Underwood = mean_Underwood - Y_test_sorted

# Calculate RMSE
rmse_Underwood = tf.sqrt(tf.reduce_mean(tf.square(errors_Underwood)))

# Calculate MAPE
mape_Underwood = tf.reduce_mean(tf.abs(errors_Underwood / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Underwood = rmse_Underwood.numpy()
mape_result_Underwood = mape_Underwood.numpy()

print("Underwood Model")
print(f"RMSE: {rmse_result_Underwood:.4f}")
print(f"MAPE: {mape_result_Underwood:.4f}%")


# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Underwood.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")

model_Newell = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.NewellMeanFunction(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Newell.training_loss, model_Newell.trainable_variables)

# Get the Estimations from the model
mean_Newell, _ = model_Newell.predict_y(X_test_sorted)

# Calculate the errors
errors_Newell = mean_Newell - Y_test_sorted

# Calculate RMSE
rmse_Newell = tf.sqrt(tf.reduce_mean(tf.square(errors_Newell)))

# Calculate MAPE
mape_Newell = tf.reduce_mean(tf.abs(errors_Newell / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Newell = rmse_Newell.numpy()
mape_result_Newell = mape_Newell.numpy()

print("Newell Model")
#print(f"RMSE: {rmse_result_Newell:.4f}")
#print(f"MAPE: {mape_result_Newell:.4f}%")
print(f"RMSE: {rmse_result_Newell:.4f}")
print(f"MAPE: {mape_result_Newell:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Newell.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")


model_Pipes = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.PipesMeanFunction(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Pipes.training_loss, model_Pipes.trainable_variables)

# Get the Estimations from the model
mean_Pipes, _ = model_Pipes.predict_y(X_test_sorted)

# Calculate the errors
errors_Pipes = mean_Pipes - Y_test_sorted

# Calculate RMSE
rmse_Pipes = tf.sqrt(tf.reduce_mean(tf.square(errors_Pipes)))

# Calculate MAPE
mape_Pipes = tf.reduce_mean(tf.abs(errors_Pipes / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Pipes = rmse_Pipes.numpy()
mape_result_Pipes = mape_Pipes.numpy()

print("Pipes Model")
print(f"RMSE: {rmse_result_Pipes:.4f}")
print(f"MAPE: {mape_result_Pipes:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Pipes.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")


model_Drew = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.DrewMeanFunction(),
    inducing_variable = inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Drew.training_loss, model_Drew.trainable_variables)

# Get the Estimations from the model
mean_Drew, _ = model_Drew.predict_y(X_test_sorted)

# Calculate the errors
errors_Drew = mean_Drew - Y_test_sorted

# Calculate RMSE
rmse_Drew = tf.sqrt(tf.reduce_mean(tf.square(errors_Drew)))

# Calculate MAPE
mape_Drew = tf.reduce_mean(tf.abs(errors_Drew / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Drew = rmse_Drew.numpy()
mape_result_Drew = mape_Drew.numpy()

print("Drew Model")
print(f"RMSE: {rmse_result_Drew:.4f}")
print(f"MAPE: {mape_result_Drew:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Drew.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")

model_Papageorgiou = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.PapageorgiouMeanFunction(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Papageorgiou.training_loss, model_Papageorgiou.trainable_variables)

# Get the Estimations from the model
mean_Papageorgiou, _ = model_Papageorgiou.predict_y(X_test_sorted)

# Calculate the errors
errors_Papageorgiou = mean_Papageorgiou - Y_test_sorted

# Calculate RMSE
rmse_Papageorgiou = tf.sqrt(tf.reduce_mean(tf.square(errors_Papageorgiou)))

# Calculate MAPE
mape_Papageorgiou = tf.reduce_mean(tf.abs(errors_Papageorgiou / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Papageorgiou = rmse_Papageorgiou.numpy()
mape_result_Papageorgiou = mape_Papageorgiou.numpy()


print("Papageorgiou Model")
print(f"RMSE: {rmse_result_Papageorgiou:.4f}")
print(f"MAPE: {mape_result_Papageorgiou:.4f}%")


# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Papageorgiou.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")



model_kerner_Konhauser = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.kerner_KonhauserMeanFunction(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_kerner_Konhauser.training_loss, model_kerner_Konhauser.trainable_variables)

# Get the Estimations from the model
mean_kerner_Konhauser, _ = model_kerner_Konhauser.predict_y(X_test_sorted)

# Calculate the errors
errors_kerner_Konhauser = mean_kerner_Konhauser - Y_test_sorted

# Calculate RMSE
rmse_kerner_Konhauser = tf.sqrt(tf.reduce_mean(tf.square(errors_kerner_Konhauser)))

# Calculate MAPE
mape_kerner_Konhauser = tf.reduce_mean(tf.abs(errors_kerner_Konhauser / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_kerner_Konhauser = rmse_kerner_Konhauser.numpy()
mape_result_kerner_Konhauser = mape_kerner_Konhauser.numpy()

print("kerner-Konhauser Model")
print(f"RMSE: {rmse_result_kerner_Konhauser:.4f}")
print(f"MAPE: {mape_result_kerner_Konhauser:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_kerner_Konhauser.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")



model_DelCastillo_Benitez = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.DelCastillo_BenitezMeanFunction(),
    inducing_variable=inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_DelCastillo_Benitez.training_loss, model_DelCastillo_Benitez.trainable_variables)

# Get the Estimations from the model
mean_DelCastillo_Benitez, _ = model_DelCastillo_Benitez.predict_y(X_test_sorted)

# Calculate the errors
errors_DelCastillo_Benitez = mean_DelCastillo_Benitez - Y_test_sorted

# Calculate RMSE
rmse_DelCastillo_Benitez = tf.sqrt(tf.reduce_mean(tf.square(errors_DelCastillo_Benitez)))

# Calculate MAPE
mape_DelCastillo_Benitez = tf.reduce_mean(tf.abs(errors_DelCastillo_Benitez / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_DelCastillo_Benitez = rmse_DelCastillo_Benitez.numpy()
mape_result_DelCastillo_Benitez = mape_DelCastillo_Benitez.numpy()

print("Del-Castillo-Benitez Model")
print(f"RMSE: {rmse_result_DelCastillo_Benitez:.4f}")
print(f"MAPE: {mape_result_DelCastillo_Benitez:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_DelCastillo_Benitez.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")



model_Jayakrishnan = gpflow.models.SGPR(
    (X_sorted, Y_sorted),
    kernel = gpflow.kernels.Exponential(),
    mean_function = EMPIRICAL_MODEL.JayakrishnanMeanFunction(),
    inducing_variable = inducing_variable,
)

opt = gpflow.optimizers.Scipy()
opt.minimize(model_Jayakrishnan.training_loss, model_Jayakrishnan.trainable_variables)

# Get the Estimations from the model
mean_Jayakrishnan, _ = model_Jayakrishnan.predict_y(X_test_sorted)

# Calculate the errors
errors_Jayakrishnan = mean_Jayakrishnan - Y_test_sorted

# Calculate RMSE
rmse_Jayakrishnan = tf.sqrt(tf.reduce_mean(tf.square(errors_Jayakrishnan)))

# Calculate MAPE
mape_Jayakrishnan = tf.reduce_mean(tf.abs(errors_Jayakrishnan / Y_test_sorted)) * 100

# To get the results as a numpy array, you can evaluate the tensors
rmse_result_Jayakrishnan = rmse_Jayakrishnan.numpy()
mape_result_Jayakrishnan = mape_Jayakrishnan.numpy()

print("Jayakrishnan Model")
print(f"RMSE: {rmse_result_Jayakrishnan:.4f}")
print(f"MAPE: {mape_result_Jayakrishnan:.4f}%")

# Assuming model is your trained SGPR model and X_test_sorted, Y_test_sorted are your test inputs and outputs
mean, var = model_Jayakrishnan.predict_y(X_test_sorted)
lower_bound = mean - 1.96 * np.sqrt(var)
upper_bound = mean + 1.96 * np.sqrt(var)

# Count how many actual values fall within the 95% CI
within_interval = np.sum((Y_test_sorted >= lower_bound) & (Y_test_sorted <= upper_bound))
percentage_within_interval = (within_interval / len(Y_test_sorted)) * 100
print(f"PWI: {percentage_within_interval:.4f}%")


Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Greenshields.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Greenshields.predict_y(Xplot)

f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)
#plt.plot(Xplot, f_lower, "--", color="navy", label="f 95% confidence", linewidth=0.5)
#plt.plot(Xplot, f_upper, "--", color="navy", linewidth=0.5)
#plt.fill_between(
#    Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="navy", alpha=0.3
#)
#plt.plot(Xplot, y_lower, ".", color="navy", label="Y 95% confidence", markersize=0.25)
##plt.plot(Xplot, y_upper, ".", color="navy", markersize=0.25)
plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Greenshields prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Greenshields prior.pdf')
plt.show()


Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Greenberg.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Greenberg.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)


plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Greenberg prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Greenberg prior.pdf')
plt.show()


Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Underwood.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Underwood.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Underwood prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Underwood prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Newell.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Newell.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Newell prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Newell prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Pipes.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Pipes.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Pipes prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Pipes prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Drew.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Drew.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)
plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Drew prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Drew prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Papageorgiou.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Papageorgiou.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)
plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Papageorgiou prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Papageorgiou prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_kerner_Konhauser.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_kerner_Konhauser.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)
plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on kerner-Konhauser prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on kerner-Konhauser prior.pdf')
plt.show()


Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_DelCastillo_Benitez.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_DelCastillo_Benitez.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)
plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on DelCastillo_Benitez prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on DelCastillo_Benitez prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Jayakrishnan.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Jayakrishnan.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)
#plt.plot(Xplot, f_lower, "--", color="navy", label="f 95% confidence", linewidth=0.5)
#plt.plot(Xplot, f_upper, "--", color="navy", linewidth=0.5)
plt.fill_between(
    Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="navy", alpha=0.1
)
#plt.plot(Xplot, y_lower, ".", color="navy", label="Y 95% confidence", markersize=0.25)
#plt.plot(Xplot, y_upper, ".", color="navy", markersize=0.25)
plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Jayakrishnan prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Jayakrishnan prior.pdf')
plt.show()





Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_PureGP.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_PureGP.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Pure Gaussian process regression model.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Pure Gaussian process regression model.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Cheng.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Cheng.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Cheng prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Cheng prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_Wang.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_Wang.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on Wang prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on Wang prior.pdf')
plt.show()

Xplot = np.linspace(0, 100, 1000)[:, None]

f_mean, f_var = model_MacNicholas.predict_f(Xplot, full_cov=False)
y_mean, y_var = model_MacNicholas.predict_y(Xplot)


f_lower = f_mean - 1.96 * np.sqrt(f_var)
f_upper = f_mean + 1.96 * np.sqrt(f_var)
y_lower = y_mean - 1.96 * np.sqrt(y_var)
y_upper = y_mean + 1.96 * np.sqrt(y_var)
y_lower_0 = y_mean - 1.645 * np.sqrt(y_var)
y_upper_0 = y_mean + 1.645 * np.sqrt(y_var)
y_lower_1 = y_mean - 2.576 * np.sqrt(y_var)
y_upper_1 = y_mean + 2.576 * np.sqrt(y_var)

plt.scatter(inducing_variable, inducing_y, label="Observations", color='red', s = 1)
plt.plot(Xplot, f_mean, "-", color="navy", label="Estimation", linewidth=1)

plt.fill_between(
    Xplot[:, 0], y_lower_0[:, 0], y_upper_0[:, 0], color="navy", alpha=0.35, label="90% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="olive", alpha=0.35, label="95% confidence interval"
)
plt.fill_between(
    Xplot[:, 0], y_lower_1[:, 0], y_upper_1[:, 0], color="darkred", alpha=0.35, label="99% confidence interval"
)
plt.ylabel('Traffic velocity (mile/h)')
plt.xlabel('Traffic density (Veh/mile)')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('2D Visualization of speed-density relationship based on MacNicholas prior.pdf')
plt.show()


# Create a new figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the perspective angle
ax.view_init(elev=30, azim=45)

# Plot the mean of the Gaussian process regression
Xplot_flat = Xplot.flatten()
f_mean_flat = f_mean.numpy().flatten()
f_var_flat = f_var.numpy().flatten()

# Plot GP Regression Mean line
ax.plot(Xplot_flat, f_mean_flat, np.zeros_like(Xplot_flat), label='GP Regression Mean', color='b')

# Calculate probability density
Y_dense = np.linspace(f_mean_flat.min(), f_mean_flat.max(), 100)
X_dense, Y_dense = np.meshgrid(Xplot_flat, Y_dense)
pdf = norm.pdf(Y_dense, loc=f_mean_flat, scale=np.sqrt(f_var_flat))

# Plot probability density surface
surf = ax.plot_surface(X_dense, Y_dense, pdf, cmap=cm.viridis, linewidth=0, antialiased=False, alpha=0.3)

# Customize the z axis
ax.set_zlim(-0.01, pdf.max())

# Add gridlines
ax.xaxis._axinfo["grid"]['linestyle'] = "--"
ax.yaxis._axinfo["grid"]['linestyle'] = "--"
ax.zaxis._axinfo["grid"]['linestyle'] = "--"

# Set labels and title
ax.set_xlabel('Traffic density (Veh/mile)')
ax.set_ylabel('Vehicle speed (mph)')
ax.set_zlabel('Probability Density')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.savefig('3D Visualization of speed-density relationship based on MacNicholas prior.pdf')
plt.show()

