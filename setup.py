import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='L-HAPI',
     version='0.1',
     scripts=['HAPILite.py'] ,
     author="Prajwal Niraula",
     author_email="prajwalniraula@gmail.com",

     description="reduced but faster HAPI version for generating atomic cross-sections",
     long_description=long_description,

     long_description_content_type="text/markdown",
     url="https://github.com/prajwal309/HAPILite.git",
     packages=setuptools.find_packages(),

     classifiers=[
         "Programming Language :: Python :: >3.5",
         "License :: MIT License",
         "Operating System :: LINUX",

     ],

 )
