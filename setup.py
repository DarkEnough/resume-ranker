from setuptools import setup, find_packages

setup(
    name="resume-ranker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pypdf>=3.0.0",
        "python-docx>=0.8.11",
        "sentence-transformers>=2.2.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "streamlit>=1.28.0",
    ],
    python_requires=">=3.8",
) 