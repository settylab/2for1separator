from setuptools import setup

setup(
    name='sep241',
    version='1.0.0',
    description='Deconvolve CUT&Tag 2for1 data.',
    url='https://github.com/settylab/2for1separator',
    author='Dominik Otto, Brennan Dury',
    author_email='dotto@fredhutch.org, brennandury@gmail.com',
    license='GNU General Public License v3.0',
    packages=['sep241'],
    install_requires=['numpy==1.21.5',
                      'pandas==1.2.5',
                      'scipy==1.6.3',
                      'pymc3==3.11.4',
                      'KDEpy==1.1.0',
                      'tqdm==4.61.1',
                      #'pyBigWig==0.3.18',
                      'pytabix==0.1',
                      'threadpoolctl==2.2.0',
                      'plotnine==0.8.0',
                      'scikit-learn==0.24.2',
                      'gtfparse==1.2.1',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
