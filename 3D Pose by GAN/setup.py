from setuptools import setup, find_packages

setup(
    name='GAN',
    version='1.0',
    packages=['GAN'],
    description='Generate 3D-pose from 2D-pose.',
    install_requires=requirements(),
    test_suite='tests',
)
