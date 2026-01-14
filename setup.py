"""
Setup script for ifc-hydro library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ifc-hydro",
    version="3.0.0",
    author="Andre Muller",
    description="Hydraulic system analysis for IFC models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrebmuller/ifc-hydro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "ifcopenshell>=0.7.0",
    ],
)
