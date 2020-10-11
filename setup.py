import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lazyq-ischubert",
    version="0.0.1",
    author="Ingmar Schubert",
    author_email="mail@ingmarschubert.com",
    description="Value-Based Reinforcement Learning with lazy action model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ischubert/lazyq",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=["numpy", "matplotlib", "tensorflow", "cvxopt", "tqdm"]
)
