from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
 
# def get_requirements(file_path: str) -> List[str]:
#     '''
#     this function returns the list of requirements
#     '''

#     with open(file_path,'rb') as file:
#         requirements = file.readlines()
#         requirements = [req.replace('\n','') for req in requirements]
 
#     if HYPEN_E_DOT in requirements:
#         requirements.remove(HYPEN_E_DOT)

#     return requirements

setup(
    name = 'StudentPerformance',
    version = '0.0.0',
    author = 'sachin',
    author_email = 'bsachinmiryala@gmail.com',
    packages = find_packages()
    # install_requires = get_requirements('requirements.txt') 
)