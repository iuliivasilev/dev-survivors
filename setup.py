from setuptools import setup, find_packages
from os.path import join, dirname
from pkg_resources import parse_requirements as _parse_requirements

SELF_DATASETS = []  # ["datasets/data/ONK/*", "datasets/data/COVID/*"]

PACKAGE_DATA = {"survivors": ["datasets/data/*"] + SELF_DATASETS}


def parse_requirements(filename):
    with open(filename) as fin:
        parsed_requirements = _parse_requirements(fin)
        requirements = [str(ir) for ir in parsed_requirements]
    return requirements

# Installing
# python setup.py install

# Final library will be established by path:
# C:\ProgramData\Anaconda3\envs\survive\Lib\site-packages

# Uploading to PYPI
# Old method (https://github.com/shirakiya/pypirc3)
# 1. pip install pypirc3
# 2. pypirc3 create -u **** -p ****
# 3. python setup.py register sdist upload

# Twine method (https://github.com/pypa/twine)
# 1. pip install twine
# 2. python setup.py sdist (recently delete the dist folder)
# 3. (test) twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/* -u **** -p ****
# 4. (release) twine upload --skip-existing --repository-url https://upload.pypi.org/legacy/ dist/* -u **** -p ****

# New twine method
# 1. python setup.py sdist bdist_wheel
# 2. twine upload --repository survivors dist/* (see in .pypirc file)

install_requires = parse_requirements("requirements/requirements.txt")

setup(
    name='survivors',
    version='1.7.0',
    license='BSD 3-Clause License',
    author='Iulii Vasilev',
    author_email='iuliivasilev@gmail.com',
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.rst')).read(),
    include_package_data=False,
    package_data=PACKAGE_DATA,
    python_requires='>=3.10',
    install_requires=install_requires,  # requirements_list
    keywords=["survival analysis", "time-to-event", "event data", "machine learning"]
)
