# Contenuto del file: setup.py

from setuptools import find_packages, setup

package_name = 'metadrive_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    # ==> AGGIUNGI QUESTA SEZIONE <==
    entry_points={
        'console_scripts': [
            'metadrive_controller = metadrive_controller.main:main',
        ],
    },
)
