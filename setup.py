from setuptools import setup, find_packages

setup(
    name="parametric-uhpc",       # this is the PyPI/distribution name
    version="0.1.0",
    packages=find_packages(),     # will now find ["parametric_uhpc"]
    install_requires=["numpy","matplotlib","pandas"],
    python_requires=">=3.8",
)
