import numpy as np
from matplotlib import pyplot as plt

from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession

from scalable_gps.gp import GaussianProcess

def test_func(x):
    return 0.5*np.sin(x)*np.exp(-0.3*np.abs(x))

x_plot = np.linspace(-10, 10, 100)
y_plot = test_func(x_plot)

x_points = (np.random.random((7, 1)) - 0.5) * 20
y_points = test_func(x_points.squeeze())

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

x_points_par = sc.parallelize(x_points).map(lambda x: Vectors.dense(x))
x_points_par.cache()
y_points_par = sc.parallelize(y_points)
y_points_par.cache()

gp = GaussianProcess(x=x_points_par, y=y_points_par, lengthscales=Vectors.dense(np.ones(1)), noise=0.2, signal_variance=1)
gp.optimize(
        lengthscale_constraints=(np.array(0.01), np.array(10)),
        noise_constraints=(0.0001, 0.01),
        signal_variance_constraints=(0.001, 1.0),
        n_steps = 25,
)

plot_par = sc.parallelize(x_plot).map(lambda x: Vectors.dense(x))
plot_par.cache()

mean1, var1 = gp.predict_mean_var(plot_par)

plt.plot(x_plot, y_plot, label="True Function")
plt.scatter(x_points, y_points, c="r",marker="x", label="Training points")
plt.plot(x_plot, mean1, c="g", label="GP")
plt.fill_between(x_plot, mean1-var1, mean1+var1, color="g", alpha=0.2)
plt.legend()
plt.show()