import setuptools

with open("README.md", "r") as rm:
    long_description = rm.read()

setuptools.setup(
    name="Sandbox",
    packages=['Sandbox'],
    version="0.0.1",
    author="Kevin Shen, Joseph Marsilla",
    author_email="kshen3778@gmail.com",
    description="Medical library for end-to-end radiotherapy machine learning workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
