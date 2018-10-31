import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="baby_zero",
    version="0.0.1",
    author="Jonah Simpson",
    author_email="jonahs99@gmail.com",
    description="An implementation of AlphaZero algorithm with NEAT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonahs99/baby-zero",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)