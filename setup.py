import ast
import os
from pathlib import Path
import re
from setuptools import setup, find_packages
import subprocess
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "sparse_attn"

NVCC_THREADS = os.getenv("NVCC_THREADS") or "4"

def get_package_version() -> str:
    with open(Path(this_dir) / "sparse_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)

def get_requirements(path: str = "requirements.txt") -> list[str]:
    with open(path) as f:
        requirements = f.read().strip().split("\n")
    resolved_requirements = []
    for line in requirements:
        if line.startswith("-r "):
            resolved_requirements += get_requirements(line.split()[1])
        elif (
            not line.startswith("--")
            and not line.startswith("#")
            and line.strip() != ""
        ):
            resolved_requirements.append(line)
    return resolved_requirements

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            nvcc_threads = max(1, int(NVCC_THREADS))

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            # Assume worst-case peak observed memory usage of ~2GB per NVCC thread.
            # Limit: peak_threads = max_jobs * nvcc_threads and peak_threads * 2GB <= free_memory.
            max_num_jobs_memory = max(1, int(free_memory_gb / (2 * nvcc_threads)))

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            print(
                f"Auto set MAX_JOBS to `{max_jobs}`, NVCC_THREADS to `{nvcc_threads}`. "
                "If you see memory pressure, please use a lower `MAX_JOBS=N` or `NVCC_THREADS=N` value."
            )
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

if os.path.isdir(".git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
else:
    assert (
        os.path.exists("csrc/cutlass/include/cutlass/cutlass.h")
    ), "csrc/cutlass is missing, please use source distribution or git clone"

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    author="Tangefly",
    author_email="22301165@bjtu.edu.cn",
    description="Sparse Attention: Fast and Memory-Efficient Exact Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tangefly/sparse-attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "sparse_attn.egg-info",
            "sparse_attn.cute",
            "sparse_attn.cute.*",
        )
    ),
    ext_modules=[
        CUDAExtension(
            name="sparse_attn_cuda",
            sources=[
                "csrc/sparse_attn/src/add_kernel.cu",
                "csrc/sparse_attn/sparse_api.cpp",
                "csrc/sparse_attn/src/flash_fwd_sparse_hdim128_bf16_causal_sm80.cu",
                "csrc/sparse_attn/src/flash_fwd_sparse_hdim128_bf16_sm80.cu",
                "csrc/sparse_attn/src/flash_fwd_sparse_hdim128_fp16_causal_sm80.cu",
                "csrc/sparse_attn/src/flash_fwd_sparse_hdim128_fp16_sm80.cu",
                "csrc/sparse_attn/src/vertical_slash_index.cu"
            ],
            include_dirs=[
                Path(this_dir) / "csrc" / "sparse_attn",
                Path(this_dir) / "csrc" / "sparse_attn" / "src",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    ],
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.9",
    install_requires=get_requirements()
)
