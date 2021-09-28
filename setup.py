from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

with open('requirements.txt') as f:
    all_reqs = f.read().split('\n')
install_requires = [x.strip() for x in all_reqs]

setup(
    name="trading-gym-next",
    version="0.0.3",
    packages=['trading_gym_next'],
    install_requires = install_requires,
    description='Trading Gym with Backtesting.py',
    long_description=readme,
    author='Rui Hirano',
    license='MIT',
)