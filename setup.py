from setuptools import setup
  
setup(
    name='flipping_page',
    version='0.1',
    description='A package to predict whether a sequence of video has flipping actions or not',
    author='Zihui Ouyang',
    author_email='makingsurgeon@gmail.com',
    packages=['my_package'],
    install_requires=[
        'pytorch'
        'scikit-learn',
        'pandas',
        'natsort',
        'albumentations',
        'numpy',
        'cv2',
        'glob',
        'tqdm'
    ],
)