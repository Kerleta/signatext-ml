from setuptools import setup, find_packages

setup(
    name="yolov5",
    version="6.2.0",
    packages=find_packages(include=['yolov5', 'yolov5.*']),
    include_package_data=True,
    install_requires=[
        # Sesuaikan dengan requirements.txt
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'numpy>=1.22',
        'opencv-python>=4.6',
        'matplotlib>=3.3',
        'Pillow>=7.1',
        'PyYAML>=5.3.1',
        'tqdm>=4.64.0',
        'scipy>=1.4.1',
        'pandas>=1.1.4',
        'seaborn>=0.11.0'
    ],
)
