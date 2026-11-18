from setuptools import setup

setup(
    name="griptile_core",
    version="0.1.0",
    description="GripTile row44 acquisition and rolling ADC grids",
    maintainer="GripTile maintainers",
    maintainer_email="yanglei18@users.noreply.github.com",
    license="Apache-2.0",
    packages=["griptile_core"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/griptile_core"]),
        ("share/griptile_core", ["package.xml", "LICENSE"]),
    ],
    install_requires=["setuptools", "numpy", "pyserial"],
    tests_require=["pytest"],
    zip_safe=False,
)
