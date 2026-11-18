from setuptools import setup

setup(
    name="griptile_viewer",
    version="0.1.0",
    packages=["griptile_viewer"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/griptile_viewer"]),
        ("share/griptile_viewer", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="GripTile maintainers",
    maintainer_email="yanglei18@users.noreply.github.com",
    description="Inspect one GripTile tactile array over ROS 2.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["viewer = griptile_viewer.app:main"]},
)
