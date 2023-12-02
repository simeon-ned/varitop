import setuptools

setuptools.setup(
    name="varitop",
    version="0.0.1",
    description="The VarITOP offering a Python library for optimal motion planning through variational integration techniques.",
    long_description_content_type="text/markdown",
    url="https://github.com/simeon-ned/varitop",
    packages=["varitop"],
    install_requires=[
        "darli",
    ],
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pre-commit",
        ]
    },
)
