from setuptools import setup

setup(
    name='marlopt',
    version='0.0.1',
    author='Gabriel-AB',
    packages=['marlopt'],
    install_requires=[
        'optfuncs @ git+https://github.com/Gabriel-AB/optfuncs@packaging',
        'pettingzoo',
        'supersuit',
    ]
)