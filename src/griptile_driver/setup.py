from glob import glob

from setuptools import setup


package = "griptile_driver"

setup(
    name=package,
    version="0.1.0",
    packages=[package],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package]),
        ("share/" + package, ["package.xml", "LICENSE"]),
        ("share/" + package + "/launch", glob("launch/*.launch.py")),
        ("share/" + package + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="GripTile maintainers",
    maintainer_email="yanglei18@users.noreply.github.com",
    description="GripTile loss-aware tactile array acquisition and replay.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "griptile_node = griptile_driver.node:main",
            "griptile_replay = griptile_driver.replay:main",
        ],
    },
)
