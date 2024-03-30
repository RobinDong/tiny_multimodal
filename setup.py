from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="tinymm",
    version="0.10",
    author="Robin Dong",
	description = "A simple and 'tiny' implementation of many multimodal models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
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
