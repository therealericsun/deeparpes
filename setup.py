from setuptools import setup

setup(
   name='deeparpes',
   version='1.0',
   description='A module to analyze ARPES imaging with deep autoencoders.',
   author='Eric Sun',
   author_email='therealericsun@gmail.com',
   packages=['deeparpes'],  #same as name
   install_requires=['scikit-learn', 'matplotlib', 'yellowbrick', 'numba', 'h5py', 'numpy', 'matplotlib', 'opencv-python', 'tensorflow', 'keras'], #external packages as dependencies
   scripts=[
            'scripts/cool',
            'scripts/skype',
           ]
)

