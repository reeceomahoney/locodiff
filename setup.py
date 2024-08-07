from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os


class CMakeBuild(build_ext):
    def run(self):
        source_dir = os.path.abspath("env")
        build_dir = os.path.abspath("env/build")

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        subprocess.check_call(["cmake", source_dir], cwd=build_dir)
        subprocess.check_call(["cmake", "--build", build_dir])

        super().run()


setup(
    name="locodiff",
    version="1.0",
    description="Quadruped locomotion using diffusion models",
    license="MIT",
    author="Reece O'Mahoney",
    packages=find_packages(),
    cmdclass={
        "build_ext": CMakeBuild,
    },
)
