from setuptools import find_packages, setup
import os

requirements = []

if os.path.exists("requirements.txt"):
    with open("requirements.txt") as f:
        content = f.readlines()
    requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='deepdive',
    version="0.0.1",
    description="DeepDive Model (api_pred)",
    license="MIT",
    author="Priyanka Gunnoo, Yann Labour",
    author_email="contact@lewagon.org",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False
)
