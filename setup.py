import sys
from setuptools import setup, find_packages
import os
thelibFolder = os.getcwd()
requirementPath = thelibFolder + '/requirements.txt'

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python >=3.7 is required for fairlib.')

with open('README.md', encoding="utf8") as f:
    # strip the header and badges etc
    readme = f.read().split('--------------------')[-1]

try:
    with open(requirementPath) as f:
        reqs = list(f.read().splitlines())
except:
    reqs = [
        "tqdm",
        "numpy",
        "docopt",
        "pandas",
        "scikit-learn",
        "torch",
        "PyYAML",
        "seaborn",
        "matplotlib",
        "pickle5",
        "transformers",
        "sacremoses",
    ]




if __name__ == '__main__':
    setup(
        name='fairlib',
        version="0.0.9",
        author="Xudong Han",
        author_email="xudongh1@student.unimelb.edu.au",
        description='Unified framework for assessing and improving fairness.',
        long_description=readme,
        long_description_content_type='text/markdown',
        url='https://github.com/HanXudong/fairlib',
        python_requires='>=3.7',
        # package_dir={"": ""},
        packages=find_packages(),
        install_requires=reqs,
        include_package_data=True,
        package_data={'': ['*.txt', '*.md', '*.opt']},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )