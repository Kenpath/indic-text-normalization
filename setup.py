#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Kenpath Technologies Pvt Ltd.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from setuptools import setup, find_packages

# Read README
long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="indic_text_normalization",
    version="0.1.0",
    description="Comprehensive text normalization for 19 Indian languages - extension of NVIDIA NeMo Text Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kenpath/indic-text-normalization",
    author="Kenpath Technologies Pvt Ltd",
    author_email="aditya0chhabra@gmail.com",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="indic hindi bengali tamil telugu tts asr text-processing text-normalization wfst pynini nemo",
    packages=find_packages(include=["indic_text_normalization", "indic_text_normalization.*"]),
    python_requires=">=3.9",
    install_requires=[
        "cdifflib",
        "editdistance",
        "inflect",
        "joblib",
        "pandas",
        "regex",
        "sacremoses>=0.0.43",
        "setuptools>=65.5.1",
        "tqdm>=4.41.0",
        "transformers",
        "wget",
        "wrapt",
        "pynini>=2.1.6.post1",
    ],
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Documentation": "https://github.com/kenpath/indic-text-normalization#readme",
        "Source": "https://github.com/kenpath/indic-text-normalization",
        "Bug Reports": "https://github.com/kenpath/indic-text-normalization/issues",
        "Original Work": "https://github.com/NVIDIA/NeMo-text-processing",
    },
)

