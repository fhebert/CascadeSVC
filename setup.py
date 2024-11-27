# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="cascadesvc",
    version="0.0.1",
    author="Florian Hébert",
    author_email="<florian.hebert@gmail.com>",
    description="Cascade SVC - Fit SVM classifiers on large samples",
    long_description=long_description,
    packages=find_packages(),
    package_dir={"cascadesvc": "cascadesvc"},
    install_requires=["scikit-learn>=1.2.2"],
    keywords=["SVM", "SVC", "Classification"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
    ],
)
