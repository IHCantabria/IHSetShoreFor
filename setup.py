from setuptools import setup, find_packages

setup(
    name='IHSetShoreFor',
    version='1.7.4',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'xarray',
        'scipy',
        'datetime',
        'spotpy',
        'pandas',
        'IHSetCalibration @ git+https://github.com/defreitasL/IHSetCalibration.git',
        'IHSetUtils @ git+https://github.com/IHCantabria/IHSetUtils.git',
        'fast_optimization @ git+https://github.com/defreitasL/fast_optimization.git'
    ],
    author='Lucas de Freitas Pereira',
    author_email='lucas.defreitas@unican.es',
    description='IH-SET Miller and Dean (2004)',
    url='https://github.com/defreitasL/IHSetShoreFor',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)