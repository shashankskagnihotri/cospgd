from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "cospgd/README.md").read_text()

setup(
    name='cospgd',
    version='0.1.3',    
    description='A tool for benchmarking adversarial robustness of pixel-wise prediction tasks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/shashankskagnihotri/cospgd',
    author='Shashank Agnihotri, Steffen Jung, Prof. Dr. Margret Keuper',
    author_email='shashanksagnihotri@gmail.com',
    license='MIT',
    packages=['cospgd'],
    install_requires=['torch>=1.7',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
