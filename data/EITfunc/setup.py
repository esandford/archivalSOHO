# to run: python setup.py build_ext --inplace
#from setuptools import setup,Extension
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('EITfunc.EITlook',['EITfunc/EITlook.py'])]

setup(name='EITfunc',
      version='0.0',
      description='',
      author='Emily Sandford',
      author_email='es835@cam.ac.uk',
      url='',
      license='MIT',
      packages=['EITlook'],
      include_dirs=[np.get_include()])
      #install_requires=['numpy','matplotlib','warnings','scipy','copy','math','itertools','collections'],
      #ext_modules=cythonize(extensions))

