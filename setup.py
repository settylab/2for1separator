from setuptools import setup

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent

setup(
    name='sep241',
    version='0.4.3',
    description='Deconvolve CUT&Tag 2for1 data.',
    url='https://github.com/settylab/2for1separator',
    author='Setty Lab',
    author_email='msetty@fredhutch.org',
    license='GNU General Public License v3.0',
    entry_points={
        'console_scripts': [
            'sep241prep = sep241.sep241prep:main',
            'sep241deconvolve = sep241.sep241deconvolve:main',
            'sep241peakcalling = sep241.sep241peakcalling:main',
            'sep241events = sep241.sep241events:main',
            'sep241mkbw = sep241.sep241mkbw:main',
            'sep241mkdt = sep241.sep241mkdt:main',
        ]
    },
    packages=['sep241'],
    install_requires=['numpy==1.21.5',
                      'pandas==1.2.5',
                      'scipy==1.6.3',
                      'pymc3==3.11.4',
                      'KDEpy==1.1.0',
                      'tqdm==4.61.1',
                      'pyBigWig==0.3.18',
                      'pytabix==0.1',
                      'threadpoolctl==2.2.0',
                      'plotnine==0.8.0',
                      'scikit-learn==0.24.2',
                      'gtfparse==1.2.1',
                      'scikit-sparse==0.4.6',
                      ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    long_description = (this_directory / "README.md").read_text(),
    long_description_content_type='text/markdown',
)
