# Lint as: python3
""" HuggingFace/jat is an open library the training of Jack of All Trades (JAT) agents.

Note:

    VERSION needs to be formatted following the MAJOR.MINOR.PATCH convention
    (we need to follow this convention to be able to retrieve versioned scripts)

Simple check list for release from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for pypi.

0. Prerequisites:
    - Dependencies:
      - twine: "pip install twine"
    - Create an account in (and join the 'simulate' project):
      - PyPI: https://pypi.org/
      - Test PyPI: https://test.pypi.org/

1. Change the version in:
    - __init__.py
    - setup.py

2. Commit these changes: "git commit -m 'Release: VERSION'"

3. Add a tag in git to mark the release: "git tag VERSION -m 'Add tag VERSION for pypi'"
    Push the tag to remote: git push --tags origin main

4. Build both the sources and the wheel. Do not change anything in setup.py between
    creating the wheel and the source distribution (obviously).

    First, delete any "build" directory that may exist from previous builds.

    For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
    (this will build a wheel for the python version you use to build it).

    For the sources, run: "python setup.py sdist"
    You should now have a /dist directory with both .whl and .tar.gz source versions.

5. Check that everything looks correct by uploading the package to the pypi test server:

    twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

    Check that you can install it in a virtualenv/notebook by running:
    pip install -i https://testpypi.python.org/pypi simulate

6. Upload the final version to actual pypi:
    twine upload dist/* -r pypi

7. Fill release notes in the tag in GitHub once everything is looking hunky-dory.

8. Change the version in __init__.py and setup.py to X.X.X+1.dev0 (e.g. VERSION=1.18.3 -> 1.18.4.dev0).
    Then push the change with a message 'set dev version'
"""
# from skbuild import setup
from distutils.core import setup

from setuptools import find_packages


__version__ = "0.0.1.dev0"  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)

REQUIRED_PKGS = [
    "accelerate==0.22.0",
    "datasets==2.14.4",
    "gymnasium==0.29.1",  # For RL action spaces and API
    "huggingface_hub>=0.10",  # For sharing objects, environments & trained RL policies
    "numpy",
    "opencv-python",
    "torch==2.0.1",
    "torchvision",
    "transformers==4.32.1",
    "wandb",
]


DEV_REQUIRE = []

TESTS_REQUIRE = [
    "free-mujoco-py",
    "gymnasium[accept-rom-license,atari,mujoco]",
    "metaworld @ git+https://github.com/qgallouedec/Metaworld@gym2.6_register",
    "minigrid",
    "pytest-xdist",
    "pytest",
]

QUALITY_REQUIRE = ["black[jupyter]~=22.0", "ruff", "pyyaml>=5.3.1"]

EXTRAS_REQUIRE = {
    "dev": DEV_REQUIRE + TESTS_REQUIRE + QUALITY_REQUIRE,
    "test": TESTS_REQUIRE,
}


setup(
    name="jat",
    version=__version__,
    description="is an open library for the training of Jack of All Trades (JAT) agents.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="HuggingFace Inc.",
    author_email="edward@huggingface.co",
    url="https://github.com/huggingface/jat",
    download_url="https://github.com/huggingface/jat/tags",
    license="Apache 2.0",
    package_dir={"": "./"},
    packages=find_packages(where="./", include="jat*"),
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="simulation environments machine learning reinforcement learning deep learning",
    zip_safe=False,  # Required for mypy to find the py.typed file
    python_requires=">=3.8",
)

# When building extension modules `cmake_install_dir` should always be set to the
# location of the package you are building extension modules for.
# Specifying the installation directory in the CMakeLists subtley breaks the relative
# paths in the helloTargets.cmake file to all of the library components.
