import tensorflow as tf
from check_shapes import inherit_check_shapes
import gpflow
import numpy as np

class GreenshieldsMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(52.1198)
        self.p_max = gpflow.Parameter(76.6752)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.v_max * (1 - (X / self.p_max))

class DrakeMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(80.5064)
        self.p_critical = gpflow.Parameter(50.0114)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.v_max * tf.math.exp( -(X/self.p_critical) ** 2)

class VanAerdeMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(91.620346)
        self.c_1 = gpflow.Parameter(1)
        self.c_2 = gpflow.Parameter(1)
        self.c_3 = gpflow.Parameter(1)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return 1/(self.c_1 + self.c_2/(self.v_max - X) + self.c_3 * X)

class MacNicholasMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(66.68755)
        self.p_max = gpflow.Parameter(153.01634)
        self.n = gpflow.Parameter(2.7017875)
        self.m = gpflow.Parameter(183.81673)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.v_max * ((self.p_max ** self.n - X ** self.n)/(self.p_max ** self.n + self.m * X ** self.n))

class WangMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(65.2328)
        self.v_critical = gpflow.Parameter(6.0222)
        self.p_critical = gpflow.Parameter(9.7276)
        self.theta_1 = gpflow.Parameter(1.5342)
        self.theta_2 = gpflow.Parameter(0.1033)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.v_critical + ((self.v_max - self.v_critical)/((1 + tf.math.exp((X - self.p_critical)/self.theta_1))**self.theta_2))

class NiMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(91.620346)
        self.gamma = gpflow.Parameter(50)
        self.tau = gpflow.Parameter(1)
        self.l = gpflow.Parameter(1)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return 1/((self.gamma * X ** 2 + self.tau * X + self.l)*(1 - tf.math.log(1 - X/self.v_max)))

class ChengMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(68.6980)
        self.p_critical = gpflow.Parameter(20.0215)
        self.m = gpflow.Parameter(2.2141)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.v_max/(1 + (X/self.p_critical)** self.m)**(2/self.m)

class GreenbergMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_critical = gpflow.Parameter(52.1198)
        self.p_max = gpflow.Parameter(92.4854)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.v_critical  * tf.math.log(self.p_max / (X + 1e-8))

class UnderwoodMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(80.5064)
        self.p_critical = gpflow.Parameter(92.4854)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        return self.v_max * tf.math.exp(- X/self.p_critical)

class NewellMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(69.6888)
        self.p_max = gpflow.Parameter(25.0057)
        self.lambda_para = gpflow.Parameter(1209.0202)

    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        # Adding a small constant epsilon to avoid division by zero
        epsilon = 1e-6
        X_safe = tf.maximum(X, epsilon)  # Ensuring X is never zero
        return self.v_max * (1 - tf.math.exp(- (self.lambda_para/self.v_max)*(1/X_safe - 1/self.p_max)))


class PipesMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(76.0455)
        self.p_max = gpflow.Parameter(51.0060)
        self.n = gpflow.Parameter(1.2237)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        safe_X = tf.minimum(X, self.p_max)
        return self.v_max * (1 - safe_X / self.p_max) ** self.n

class DrewMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(74.77972)
        self.p_max = gpflow.Parameter(280.64917)
        self.m_1 = gpflow.Parameter(2.9984262)
        self.m_2 = gpflow.Parameter(300.37567)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X = np.asarray(X)
        ratio = X / self.p_max
        ratio = tf.maximum(ratio, 0)
        return self.v_max * (1 - ratio ** self.m_1) ** self.m_2

class PapageorgiouMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(79.4948)
        self.p_max = gpflow.Parameter(24.8334)
        self.alpha = gpflow.Parameter(1.0243)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X = np.asarray(X)
        return self.v_max * tf.math.exp(-(1/self.alpha) * (X/self.p_max) ** self.alpha)

class kerner_KonhauserMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(60.1684)
        self.p_critical = gpflow.Parameter(106.2724)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X = np.asarray(X)
        return self.v_max* (1/(1 + tf.math.exp(((X/self.p_critical)-0.25)/(0.06)) - 3.72 * 10 **(-6)))

class DelCastillo_BenitezMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(69.6888)
        self.p_max = gpflow.Parameter(108.4056)
        self.P_W = gpflow.Parameter(11.1527)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X = np.asarray(X)
        return self.v_max * (1 - tf.math.exp((self.P_W/self.v_max)* (1 - self.p_max/X)))

class JayakrishnanMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(52.1198)
        self.p_max = gpflow.Parameter(25.1779)
        self.v_min = gpflow.Parameter(35.0052)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X = np.asarray(X)
        return self.v_min + (self.v_max - self.v_min) * (1 - X/self.p_max)

class ArdekaniandGhandehari(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_critical = gpflow.Parameter(40.4081)
        self.p_max = gpflow.Parameter(56.8403)
        self.p_min = gpflow.Parameter(0.0135)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X = np.asarray(X)
        return self.v_critical * tf.math.log((self.p_max + self.p_min)/(X + self.p_min))

class LeeMeanFunction(gpflow.functions.MeanFunction):
    def __init__(self):
        super().__init__()
        self.v_max = gpflow.Parameter(86.1024)
        self.p_max = gpflow.Parameter(60.2292)
        self.E = gpflow.Parameter(5.3013)
        self.gamma = gpflow.Parameter(682.8654)
    def __call__(self, X: gpflow.base.TensorType) -> tf.Tensor:
        X = np.asarray(X)
        return self.v_max * (1 - X/self.p_max)*(1 + (self.E) * (X/self.p_max) ** self.gamma)**(-1)


class EdieMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, breakpoint):
        super().__init__()
        self.breakpoint = tf.constant(breakpoint, dtype=gpflow.default_float())


    def __call__(self, X):
        X = tf.reshape(X, (-1, 1))
        return tf.where(
            X <= self.breakpoint,
            82.63208 * tf.exp(-(X) / 23.580702),
            11.010691 * tf.math.log(133.80333 / (X))
        )

class Two_regimeMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, breakpoint):
        super().__init__()
        self.breakpoint = tf.constant(breakpoint, dtype=gpflow.default_float())

    def __call__(self, X):
        X = tf.reshape(X, (-1, 1))
        return tf.where(
            X <= self.breakpoint,
            60.55406 - 1.0032127 * (X),
            15.434173 - 0.12082339 * (X)
        )

class ModifiedGreenbergMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, breakpoint):
        super().__init__()
        self.breakpoint = tf.constant(breakpoint, dtype=gpflow.default_float())

    def __call__(self, X):
        X = tf.reshape(X, (-1, 1))
        return tf.where(
            X <= self.breakpoint,
            41.46335 + X * 0,
            12.3204775 * X * tf.math.log(124.85025 / (X))
        )


class Three_regimeMeanFunction(gpflow.mean_functions.MeanFunction):
    def __init__(self, breakpoint1, breakpoint2):
        super().__init__()
        self.breakpoint1 = tf.constant(breakpoint1, dtype=gpflow.default_float())
        self.breakpoint2 = tf.constant(breakpoint2, dtype=gpflow.default_float())

    def __call__(self, X):
        X = tf.reshape(X, (-1, 1))
        condition1 = X <= self.breakpoint1
        condition2 = tf.logical_and(self.breakpoint1 < X, X <= self.breakpoint2)
        condition3 = X > self.breakpoint2

        result1 = 72.01518 - 1.639043 * (X)
        result2 = 23.658642 - 0.24986239 * (X)
        result3 = 15.434173 - 0.12082339 * (X)

        result = tf.where(condition1, result1, tf.where(condition2, result2, result3))
        return result