from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "algebra>=1.0",
    "backends>=1.4.11",
    "backends-matrix>=1.2.1",
    "plum-dispatch>=1.0",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
