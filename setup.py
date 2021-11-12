"""
setup script for anaties package

Note I found the following site excellent, and used a lot of the ideas there:
https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/

https://github.com/EricThomson/anaties
"""

from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "readme.md").read_text()

setup(
    name='anaties',
    version='0.1.4',
    description="Common analysis utilities",
    author="Eric Thomson",
    author_email="thomson.eric@gmail.com",
    licence="MIT",
    url="https://github.com/EricThomson/anaties",
    packages=find_packages(include=['anaties', 'anaties.*']),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'],
    long_description=long_description,
    long_description_content_type="text/markdown"
)
