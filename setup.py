from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="constrai",
    version="0.3.0",
    author="Ambar",
    description="Formal safety framework for AI agents with provable guarantees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ambar-13/ConstrAI",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],  # Zero dependencies for core
    extras_require={
        "anthropic": ["anthropic>=0.20.0"],
        "openai": ["openai>=1.0.0"],
        "dev": ["pytest>=7.0"],
    },
    keywords=[
        "formal-methods",
        "ai-safety",
        "autonomous-agents",
        "execution-semantics",
        "invariants",
        "verification",
        "agent-framework",
        "state-machines",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
)
