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
    url="https://github.com/rebotlucion/QuickEmo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
)
"""
https://docs.hektorprofe.net/python/modulos-y-paquetes/paquetes/
Distribución
Para crear un paquete distribuible tenemos que crear un 
fichero setup.py fuera de la raíz, indicando una información básica, 
de la siguiente forma:


setup.py
paquete/
    __init__.py
    hola/
        __init__.py
        saludos.py
    adios/
        __init__.py
        despedidas.py
        
        
        setup.py

from setuptools import setup
setup(
    name="paquete",
    version="0.1",
    description="Este es un paquete de jemplo",
    author="Hector Costa",
    author_email="hola@hektorprofe.com",
    url="http://www.hektorprofe.net",
    packages=['paquete','paquete.hola','paquete.adios']
    scripts=[]
)
Una vez hemos definido el distribuible con su información básica, 
incluyendo los paquetes y subpaquetes que lo forman, así como los 
posibles scripts, debemos crearlo. Para hacerlo utilizaremos el siguiente
 comando allí donde tenemos el setup.py:


python setup.py sdist

Ahora, si todo ha funcionado correctamente, se habrá creado una nueva
 carpeta llamada dist, y en ella encontraremos un fichero zip en Windows 
 o tar.gz si estamos en Linux. Este fichero es nuestro distribuible y 
 ahora podríamos compartirlo con todo el mundo para que puedan instalar 
 nuestro paquete.


Dependencias
Ahora imaginaros que en vuestro paquete algún código utiliza funciones 
de un módulo externo o paquete que hay que instalar manualmente. Esto se conoce como dependencias del paquete, y por suerte podemos indicar a un parámetro que descargue todos los paquetes en la versión que nosotros indiquemos, se trata de install_requires.

Por ejemplo imaginad que dentro de nuestro paquete necesitamos utilizar 
el módulo Pillow para manejar imágenes. Por regla general podemos instalarlo desde la terminal con el comando:


pip install pillow
Pero si queremos que el paquete lo instale automáticamente sólo 
tenemos que indicarlo de esta forma:

setup(...,
      install_requires=["pillow"],
)
"""

"""
https://entrenamiento-python-basico.readthedocs.io/es/latest/leccion8/distribucion.html

8.3.3. Estructura de proyecto
Para poder empaquetar un proyecto necesita como mínimo la estructura de archivos siguiente:

DIRECTORIO-DEL-PROYECTO
├── LICENSE
├── MANIFEST.in
├── README.txt
├── setup.py
└── NOMBRE-DEL-PAQUETE
     ├── __init__.py
     ├── ARCHIVO1.py
     ├── ARCHIVO2.py
     └── MODULO (OPCIONAL)
           ├── __init__.py
           └── MAS_ARCHIVOS.py
A continuación se detallan el significado y uso de la estructura de directorio anterior:

DIRECTORIO-DEL-PROYECTO puede ser cualquiera, no afecta en absoluto, lo que cuenta es lo que hay dentro.

NOMBRE-DEL-PAQUETE tiene que ser el nombre del paquete, si el nombre es tostadas_pipo, este directorio tiene que llamarse también tostadas_pipo. Y esto es así. Dentro estarán todos los archivos que forman la librería.

LICENSE: es el archivo donde se define los términos de licencia usado en su proyecto. Es muy importate que cada paquete cargado a PyPI incluirle una copia de los términos de licencia. Esto le dice a los usuario quien instala el paquete los términos bajos los cuales pueden usarlo en su paquete. Para ayuda a seleccionar una licencia, consulte https://choosealicense.com/. Una vez tenga seleccionado una licencia abra el archivo LICENSE e ingrese el texto de la licencia. Por ejemplo, si usted elije la licencia GPL:
    
MANIFEST.in: es el archivo donde se define los criterios de inclusión y 
exclusión de archivos a su distribución de código fuente de su proyecto. 
Este archivo incluye la configuración del paquete como se indica a 
continuación:

include LICENSE
include *.txt *.in
include *.py
recursive-include tostadas_pipo *
global-exclude *.pyc *.pyo *~
prune build
prune dist
README.txt: es el archivo donde se define la documentación general del paquete, este archivo es importante debido a que no solo es usado localmente en un copia descargada, sino como información usada el en sitio de PyPI. Entonces abra el archivo README.txt e ingrese el siguiente contenido. Usted puede personalizarlo como quiera:

==================
NOMBRE-DEL-PAQUETE
==================

Este es un ejemplo simple de un paquete Python.

Usted puede usar para escribir este contenido la guía
`Restructured Text (reST) and Sphinx CheatSheet <http://openalea.gforge.inria.fr/doc/openalea/doc/_build/html/source/sphinx/rest_syntax.html>`_.

"""

