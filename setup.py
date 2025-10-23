from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bealgo",
    version="0.1.0",
    author="Your Name",  # TODO: Update with actual authors
    author_email="your.email@example.com",  # TODO: Update with contact email
    description="Backward Elimination Algorithm for Joint PMF Computation in Network Propagation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BEALGO",  # TODO: Update with actual GitHub URL
    py_modules=["be_joint_pmf"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",  # TODO: Update if different license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    keywords="network propagation, probability, joint pmf, backward elimination, graph algorithms",
)
