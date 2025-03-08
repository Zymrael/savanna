from setuptools import find_packages, setup

"""
Minimal setup to enable local installation.

E.g., 
```
pip install -e .
```
"""
setup(
    name="savanna",  
    version="0.1",
    packages=find_packages(), 
    install_requires=[
    ],
    entry_points={

    },
)