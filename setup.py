from setuptools import setup

setup(
    name="scalable_gps",
    version="0.0.2",
    packages=["scalable_gps"],
    install_requires=[
        "pyspark[sql,pandas_on_spark]",
        "scikit-learn",
        "botorch",
        "sparktorch"
    ],
    exclude_package_data={'': ["results/*"]}
)
