from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyirmembraneanalyzer',
    version='1.0.0',
    py_modules=['ir_membrane_analyzer'],
    author='Takeshi Sato, Hiroko Tamagaki-Asahina',
    author_email='takeshi@mb.kyoto-phu.ac.jp',
    description='Automated analysis of polarized ATR-FTIR spectra for membrane protein orientation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/pyirmembraneanalyzer',
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.23.0',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='FTIR spectroscopy membrane-proteins dichroic-ratio',
)
