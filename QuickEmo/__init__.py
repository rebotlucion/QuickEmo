# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:34:29 2019

@author: Paco
"""

"""
Los paquetes de python son un espacio de nombres que contiene varios 
módulos o paquetes, a veces relacionados entre ellos aunque no tiene porqué.
Se crean en un directorio que debe incluir obligatoriamente un fichero 
especial llamado __init__.py que es el que indica que se trata de un paquete 
y luego puede haber otros módulos e incluso otros paquetes. 
"""


print(f'Invoking __init__.py for {__name__}')
A = ['quux', 'corge', 'grault']




# El fichero __init__.py puede y suele estar vacío, 
# aunque se puede usar para importar modulos comunes entre paquetes.
import pkg.mod1, pkg.mod2


"""

La siguiente es la estructura típica de un paquete:

mi_paquete/
    __init__.py
    modulo1.py
    modulo2.py
    utiles/
        __init__py
        utiles1.py
        config.py

"""
