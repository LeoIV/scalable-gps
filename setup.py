from setuptools import setup

setup(
    name="scalable_gps",
    version="0.0.14",
    packages=["scalable_gps"],
    install_requires=[
        "pyspark[sql,pandas_on_spark]",
        "scikit-learn",
        "botorch",
        "sparktorch",
        "tqdm",
    ],
    exclude_package_data={'': ["results/*"]}
)
