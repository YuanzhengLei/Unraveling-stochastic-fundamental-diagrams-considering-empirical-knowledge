import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import xlrd as xlrd
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
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

def tf_safe_wrapper(func):
    def safe_func(X, *args, **kwargs):
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        return func(X, *args, **kwargs)
    return safe_func

@tf_safe_wrapper
def greenshields(X):
    return 52.1198 * (1 - (X / 76.6752))

Y_pred = greenshields(X_test_sorted)

# Calculate the RMSE
rmse_greenshields = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_greenshields = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Greenshields Model")
print(f"RMSE: {rmse_greenshields:.4f}")
print(f"MAPE: {mape_greenshields:.4f}%")


@tf_safe_wrapper
def Cheng(X):
    return 68.6980/(1 + (X/20.0215)** 2.2141)**(2/2.2141)

Y_pred = Cheng(X_test_sorted)

# Calculate the RMSE
rmse_Cheng = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_Cheng = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Cheng Model")
print(f"RMSE: {rmse_Cheng:.4f}")
print(f"MAPE: {mape_Cheng:.4f}%")

@tf_safe_wrapper
def Wang(X):
    return 6.0222 + ((65.2328 - 6.0222)/((1 + tf.math.exp((X - 9.7276)/1.5342)) ** 0.1033))

Y_pred = Wang(X_test_sorted)

# Calculate the RMSE
rmse_Wang = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_Wang = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Wang Model")
print(f"RMSE: {rmse_Wang:.4f}")
print(f"MAPE: {mape_Wang:.4f}%")

@tf_safe_wrapper
def MacNicholas(X):
    return 73.52686 * ((205.54465 ** 3.7189584 - X ** 3.7189584)/(205.54465 ** 3.7189584 + 501.67047 * X ** 3.7189584))

Y_pred = MacNicholas(X_test_sorted)

# Calculate the RMSE
rmse_MacNicholas = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_MacNicholas = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original MacNicholas Model")
print(f"RMSE: {rmse_MacNicholas:.4f}")
print(f"MAPE: {mape_MacNicholas:.4f}%")

@tf_safe_wrapper
def greenberg(X):
    return 52.1198 * tf.math.log(92.4854 / (X + 1e-8))

Y_pred = greenberg(X_test_sorted)

# Calculate the RMSE
rmse_greenberg = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_greenberg = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Greenberg Model")
print(f"RMSE: {rmse_greenberg:.4f}")
print(f"MAPE: {mape_greenberg:.4f}%")

@tf_safe_wrapper
def underwood(X):
    return 80.5064 * tf.math.exp(-X/92.4854)

Y_pred = underwood(X_test_sorted)

# Calculate the RMSE
rmse_underwood = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_underwood = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Underwood Model")
print(f"RMSE: {rmse_underwood:.4f}")
print(f"MAPE: {mape_underwood:.4f}%")

@tf_safe_wrapper
def newell(X):
    return 69.6888 * (1 - tf.math.exp(- (1209.0202/69.6888)*(1/X - 1/25.0057)))

Y_pred = newell(X_test_sorted)

# Calculate the RMSE
rmse_newell = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_newell = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Newell Model")
print(f"RMSE: {rmse_newell:.4f}")
print(f"MAPE: {mape_newell:.4f}%")

@tf_safe_wrapper
def pipes(X):
    safe_X = tf.minimum(X, 51.0060)  # Ensuring X does not go above 8.7896
    return 76.0455 * (1 - safe_X / 51.0060) ** 1.2237

Y_pred = pipes(X_test_sorted)

# Calculate the RMSE
rmse_pipes = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_pipes = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Pipes Model")
print(f"RMSE: {rmse_pipes:.4f}")
print(f"MAPE: {mape_pipes:.4f}%")

@tf_safe_wrapper
def drew(X):
    return 73.990906 * (1 - (X / 283.10898) ** 2.8559134) ** 194.68063

Y_pred = drew(X_test_sorted)

# Calculate the RMSE
rmse_drew = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_drew = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Drew Model")
print(f"RMSE: {rmse_drew:.4f}")
print(f"MAPE: {mape_drew:.4f}%")


@tf_safe_wrapper
def papageorgiou(X):
    return 79.4948 * tf.math.exp(-(1/1.0243) * (X/24.8334) ** 1.0243)

Y_pred = papageorgiou(X_test_sorted)

# Calculate the RMSE
rmse_papageorgiou = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_papageorgiou = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Papageorgiou Model")
print(f"RMSE: {rmse_papageorgiou:.4f}")
print(f"MAPE: {mape_papageorgiou:.4f}%")

@tf_safe_wrapper
def kerner_konhauser(X):
    return 60.1684 * (1/(1 + tf.math.exp(((X/106.2724)-0.25)/(0.06)) - 3.72 * 10 **(-6)))

Y_pred = kerner_konhauser(X_test_sorted)

# Calculate the RMSE
rmse_kerner_konhauser = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_kerner_konhauser = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Kerner-konhauser Model")
print(f"RMSE: {rmse_kerner_konhauser:.4f}")
print(f"MAPE: {mape_kerner_konhauser:.4f}%")

@tf_safe_wrapper
def delcastillo_benitez(X):
    return 69.6888 * (1 - tf.math.exp((11.1527/108.4056)* (1 - 108.4056/X)))

Y_pred = delcastillo_benitez(X_test_sorted)

# Calculate the RMSE
rmse_delcastillo_benitez = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_delcastillo_benitez = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original delcastillo-benitez Model")
print(f"RMSE: {rmse_delcastillo_benitez:.4f}")
print(f"MAPE: {mape_delcastillo_benitez:.4f}%")

@tf_safe_wrapper
def jayakrishnan(X):
    return 35.0052 + (52.1198 - 35.0052) * (1 - X/25.1779)

Y_pred = jayakrishnan(X_test_sorted)

# Calculate the RMSE
rmse_jayakrishnan = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_jayakrishnan = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Jayakrishnan Model")
print(f"RMSE: {rmse_jayakrishnan:.4f}")
print(f"MAPE: {mape_jayakrishnan:.4f}%")

@tf_safe_wrapper
def edie(X, breakpoint):
    condition = X <= breakpoint
    return tf.where(
        condition,
        86.19598 * tf.math.exp(-(X ) / 51.856693),
        18.114496 * tf.math.log(159.73201 / (X ))
    )

Y_pred = edie(X_test_sorted, 50)

# Calculate the RMSE
rmse_edie = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_edie = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original Edie Model")
print(f"RMSE: {rmse_edie:.4f}")
print(f"MAPE: {mape_edie:.4f}%")

@tf_safe_wrapper
def two_regime(X, breakpoint):
    condition = X <= breakpoint
    return tf.where(
        condition,
        82.95282 - 1.1018451 * (X),
        20.520544 - 0.12301712 * (X)
    )

Y_pred = two_regime(X_test_sorted, 65)

# Calculate the RMSE
rmse_two_regime = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_two_regime = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original two-regime Model")
print(f"RMSE: {rmse_two_regime:.4f}")
print(f"MAPE: {mape_two_regime :.4f}%")


def modified_greenberg(X, breakpoint):
    condition = X <= breakpoint
    return tf.where(
        condition,
        66.131805 + X * 0,
        22.09243 * X * tf.math.log(147.85953 / (X))
    )

Y_pred = modified_greenberg(X_test_sorted, 35)

# Calculate the RMSE
rmse_modified_greenberg = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_modified_greenberg = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original modified-greenberg Model")
print(f"RMSE: {rmse_modified_greenberg:.4f}")
print(f"MAPE: {mape_modified_greenberg:.4f}%")

@tf_safe_wrapper
def three_regime(X, breakpoint1, breakpoint2):
    condition1 = X <= breakpoint1
    condition2 = tf.logical_and(breakpoint1 < X, X <= breakpoint2)
    condition3 = X > breakpoint2

    result1 = 82.207214 - 1.0211389 * (X)
    result2 = 52.25136 - 0.5437932 * (X)
    result3 = 20.520544 - 0.12301712 * (X)

    return tf.where(condition1, result1, tf.where(condition2, result2, result3))

Y_pred = three_regime(X_test_sorted, 40, 65)

# Calculate the RMSE
rmse_three_regime = tf.sqrt(tf.reduce_mean(tf.square(Y_test_sorted - Y_pred)))

# Calculate the MAPE
mape_three_regime = tf.reduce_mean(tf.abs((Y_test_sorted - Y_pred) / Y_test_sorted)) * 100

print("Original three-regime Model")
print(f"RMSE: {rmse_three_regime:.4f}")
print(f"MAPE: {mape_three_regime:.4f}%")