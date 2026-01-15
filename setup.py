from setuptools import setup, find_packages

setup(
    name="uidai_hackathon_project",
    version="0.1.0",
    author="Kanishka Sharma",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "seaborn",
        "requests"
    ],
    python_requires=">=3.10",
)
