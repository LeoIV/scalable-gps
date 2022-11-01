from scalable_gps.objective import Ackley
from scalable_gps.turbo_state import TurboInstance
from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    turbos = sc.parallelize(
        [TurboInstance(batch_size=4, function=Ackley(5), identifier=f"TR-{i}", seed=i) for i in range(5)])
    turbos.foreach(lambda t: t.optimize())
