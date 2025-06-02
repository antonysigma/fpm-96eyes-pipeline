# 96-Eyes Ptychographic reconstruction algorithm and file decoder

This repository is one of the following components of the 96-Eyes instrument design project:

1. (this repository) 96-Eyes Ptychographic reconstruction algorithm and file
   decoder;

2. GPU-accelerated, Zernike-guided lens aberration recovery (FPM-EPRY) algorithm for
   brightfield-only FPM images, and graphical user interfaces;

3. 96-Eyes instrument control code for concurrent, asynchronous camera capture
   and disk I/O;

4. Electronic circuit design and circuit board drawing of the illumination,
   thermal, and incubation modules;

## Status

This repository is currently in *code freeze* status; new feature will not be
developed here. However, we do occasionally patch the code to track the latest
CPU/GPU hardware changes, especially those hosted on the build validation
server. We also accept pull requests and issue tickets.

## Contributing

We do accept pull requests (PR) for bug fixes (e.g. broken links, C/C++
compilation error). Please state your affiliations and the brief (~25 words)
summary of the changes in the PR.

## Quick start

1. (Optional) Download the 96-Eyes raw data;

2. Clone this repository to path `fpm-96eyes-reconstruction/`;

3. Assuming your computer has the [UV the python-pip
   accelerator](https://docs.astral.sh/uv/#installation) installed, run `cd
   fpm-96eyes-reconstruction/; uv venv --python=3.12; uv venv` to setup the
   Python virtual environment;

4. Activate the virtual environment by `source .venv/bin/activate`;

5. Install the build system by `uv pip install meson ninja`;

6. Download 3rd party C/C++ dependencies: `meson setup build/`;

7. If the `meson setup` command returns error, it means your computer is not
   configured to compile C/C++ projects. Stop here and file an issue on Github;

8. Compile everything with `ninja -C fpm-96eyes-reconstruction/build/ all`;

9. Test everything with `ninja -C fpm-96eyes-reconstruction/build/ test`.

10. (Optional) install the standalone executables by `meson -C build/ install
    --skip-subprojects`.

Expected outputs:

```shell
The Meson build system
Version: 1.2.3
Source dir: /home/antony/Projects/fpm-96eyes-pipeline
Build dir: /home/antony/Projects/fpm-96eyes-pipeline/build
Build type: native build
Project name: u96eyes
Project version: 0.1
C compiler for the host machine: ccache cc (gcc 9.4.0 "cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
C linker for the host machine: cc ld.bfd 2.34
C++ compiler for the host machine: ccache c++ (gcc 9.4.0 "c++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0")
C++ linker for the host machine: c++ ld.bfd 2.34
Host machine cpu family: x86_64
Host machine cpu: x86_64

...
Build targets in project: 21

u96eyes 0.1

  Subprojects
    armadillo-code     : YES
    catch2             : YES 2 warnings
    cpp-base64         : YES
    cxxopts            : YES
    halide-x86_64-linux: YES
    highfive           : YES
    pugixml            : YES
    taskflow           : YES

Found ninja-1.10.0 at /usr/bin/ninja
```

## Project directory structure

Each subfolder contains a standalone functionality of the reconstruction App. We
don't have a global `source/` folder; each sub-folder has their standalone
source folder `src/`, header files `inc/`, and test logic `tests/`.

- `algorithms/`: image processing pipelines written in [Halide language](https://doi.org/10.1145/3150211),
  potentially GPU-accelerated;

- `common/`: shared data structure, memory aligned for efficient data access;

- `wavevector_calibration/`: air-to-liquid meniscus compensating oblique illumination angle estimation;

- `metadata/`: Imaging environment parameters serialization in XML;

- `storage/`: 96-Eyes file format specifications, and encode/decode logic;

- `subprojects/`: 3rd party C/C++ libraries;

- `test-data/`: Mock data for unit testing;

- `apps/`: Standalone executables (EXE) integrating all the above functionalities.


## Software design philosophy

Unlike conventional C/C++ library projects, this project builds the command-line
interface (CLI) as a standalone executable (EXE) to be deployed on the 96-Eyes
instrument. We don't support Matlab/Go/Python binding yet, but pull requests are
welcome.

In the case of 96-Eyes project, the CLI is called by a separate job server
hosted in the instrument. The job server queues the reconstruction jobs and
pushes the jobs to the GPU sequentially by calling the CLI. The CLI is favored
over Matlab/Python/Go bindings and/or C++ RPC servers because of the simplicity
of the cancellation logic and the exception handling logic.

## Obtaining the raw data

The 96-Eyes instruction, by design, streams multi-modal cell culture images
reaching up to 20GB per time point. So, it is more scalable to share the data
via peer-to-peer, decentralized online storage.

To download the raw data, please click the following link to accept the terms,
and then download the raw data via Bittorrent client.

https://academictorrents.com/details/c95c06e98a74a580ccbcceafdc1188ea144021c8

To learn about the AcademicTorrent initiative, please read the white papers at

- Cohen, Joseph Paul, and Henry Z. Lo. “Academic Torrents: A
  Community-Maintained Distributed Repository.” Annual Conference of the Extreme
  Science and Engineering Discovery Environment, 2014,
  http://doi.org/10.1145/2616498.2616528.

- Henry Z. Lo. and Cohen, Joseph Paul “Academic Torrents: Scalable Data
  Distribution.” Neural Information Processing Systems Challenges in Machine
  Learning (CiML) Workshop, 2016, http://arxiv.org/abs/1603.04395.

- https://academictorrents.com/docs/about.html

## Citations

A.C.S. Chan, J Kim, A Pan, H Xu, D Nojima, C Hale, S Wang, C Yang, “Parallel
Fourier ptychographic microscopy for high-throughput screening with 96 cameras
(96 Eyes)” Scientific Reports 9, 11114 (2019). http://dx.doi.org/10.1038/s41598-019-47146-z