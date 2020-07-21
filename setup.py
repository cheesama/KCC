from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="KCC",
    version="0.1",
    description="Korean Commented bert based Classifier",
    author="Cheesama",
    install_requires=required,
    packages=find_packages(exclude=["docs", "tests", "tmp", "data"]),
    python_requires=">=3",
    zip_safe=False,
    include_package_data=True,
)
