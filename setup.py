import sys
from setuptools import setup, find_packages
import os
thelibFolder = os.getcwd()
requirementPath = thelibFolder + '/requirements.txt'

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python >=3.7 is required for FairCLS.')

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
    ]




if __name__ == '__main__':
    setup(
        name='faircls',
        version="0.0.1",
        author="Xudong Han",
        author_email="xudongh1@student.unimelb.edu.au",
        description='Unified framework for bias detection and mitigation in classification.',
        long_description=readme,
        long_description_content_type='text/markdown',
        url='https://github.com/HanXudong/Fair_NLP_Classification',
        python_requires='>=3.7',
        # package_dir={"": ""},
        packages=find_packages(),
        install_requires=reqs,
        include_package_data=True,
        package_data={'': ['*.txt', '*.md', '*.opt']},
        # entry_points={
        #     "console_scripts": ["faircls=faircls.command_line:main"],
        # },
        # scripts=["faircls-train"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )