from setuptools import find_packages, setup

package_name = "dvlt_slam_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/replay.launch.py"]),
        ("share/" + package_name + "/config", ["config/dvlt_slam.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="henry",
    maintainer_email="hychan@umich.edu",
    description="ROS 2 wrapper for dvlt-slam",
    license="MIT",
    # `colcon` generates console-script launchers with the Python interpreter
    # that built the package (the ROS system interpreter here).  DVLT-SLAM's
    # runtime dependencies and the top-level `slam` package live in the
    # repository's .venv-ros environment, so install source-owned wrappers
    # which select that interpreter at runtime instead.
    scripts=[
        "scripts/slam_node",
        "scripts/image_folder_publisher",
    ],
)
