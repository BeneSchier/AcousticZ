from setuptools import setup, find_packages

setup(
    name='AcousticZ',
    version='0.0.1',
    author='Benedikt Schier',
    author_email='benedikt.schier@gmail.com',
    description='Room Impulse Response simulation using stochastic ray tracing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/BeneSchier/AcousticZ',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'trimesh',
        'scipy',
        'soundfile',
        'matplotlib',
        # List any additional dependencies your package requires
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.11',
    ],
)
