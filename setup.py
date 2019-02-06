import versioneer
from setuptools import setup, find_packages

setup(name='leabra-tf',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      license='Not open source',
      author='apra93',
      packages=find_packages(),
      description='A short description of the project.',
      )
