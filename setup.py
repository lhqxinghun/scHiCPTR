import sys

if sys.version_info < (3, 6):
    sys.exit('scHiCPTR requires Python >= 3.6')
from pathlib import Path
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scHiCPTR",
    version="0.1.2",
    author="",
    author_email="",
    description="An unsupervised pseudotime inference pipeline through dual graph refinement for single cell Hi-C data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lhqxinhun/scHiCPTR",
    project_urls={
        "Bug Tracker": "https://github.com/lhqxinhun/scHiCPTR",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
    ]
)
