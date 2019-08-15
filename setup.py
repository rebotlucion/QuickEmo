# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:30:56 2019

@author: Paco
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QuickEmo-pkg-rebotlucion",
    version="0.0.1",
    author="Francisco Portal López",
    author_email="rebotlucion@gmail.com",
    description="Reconocimiento rápido de emociones en el discurso",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/rebotlucion/QuickEmo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
)