from setuptools import setup

setup(
    name='marlopt',
    version='1.0.0',
    author='Gabriel-AB',
    packages=['marlopt'],
    install_requires=[
        'id-mava[tf,reverb,launchpad] @ git+https://github.com/instadeepai/Mava',
        'optfuncs @ git+https://github.com/Gabriel-AB/optfuncs@packaging',
        'pettingzoo',
        'supersuit',
    ]
)