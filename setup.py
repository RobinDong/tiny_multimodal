from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename, "r", encoding="utf-8") as fp:
        return [ln.strip() for ln in fp.read().split("\n")]


def fetch_readme(filename):
    with open(filename, "r", encoding="utf-8") as fp:
        return fp.read()


setup(
    name="tinymm",
    version="0.10",
    author="Robin Dong",
    description="A simple and 'tiny' implementation of many multimodal models",
    long_description=fetch_readme("README.md"),
    long_description_content_type="text/markdown",
    keywords="Vision-Language, Multimodal, Deep Learning, Library, PyTorch",
    license="MIT License",
    packages=find_namespace_packages(include="tinymm.*"),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.8.0",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)
