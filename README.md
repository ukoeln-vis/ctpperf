C++ Compile Time Polymorphism for Ray Tracing
=============================================

Source code repository accompanying the scientific paper

S. Zellmann, U. Lang (2017)

_"C++ Compile Time Polymorphism for Ray Tracing"._

Proceedings of the 22nd International Symposium on Vision, Modeling and Visualization (VMV 2017), Bonn, Germany, September 25, 2017 (accepted for publication).

Description
-----------

This repository includes the complete benchmark routine for the paper mentioned above. Note that the benchmark code is not intended to be "production ready" and was only tested in a very limited environment: Ubuntu 16.04, compiling with GCC 5.4.0 or CUDA 8.0 nvcc. Running the benchmarks requires switching git branches and editing some compile time defines in header files!

We created the benchmark suite by first forking [Visionaray][1] at commit [047e641ca5ee2cb35507923f109bcd9fe2a00b2e][2]. For simplicity, we implemented the benchmark itself as a _Visionaray example_ that can be found in

```src/examples/wavefront_pathtracer```

For the paper, we implemented the following benchmark tests using a wavefront pathtracing approach:
- Shading, compile time polymorphism to determine the correct shading routines (CPU & CUDA).
  Implemented on branch [shading_benchmarks][3].

- Shading, object oriented polymorphism to determine the correct shading routines (CPU only).
  Implemented on branch [shading_benchmark_oop][4].

- Shading, one kernel per material type, and sorting (CPU & CUDA).
  Implemented on branch [shading_benchmarks][3].

- Intersect, compile time polymorphism to support BVHs containing multiple primitive types (CPU & CUDA).
  Implemented on branch [intersect_benchmarks][5].

- Intersect, object oriented polymorphism to support BVHs containing multiple primitive types (CPU only).
  Implemented on branch [intersect_benchmark_oop][6].

- Intersect, with one BVH per primitive type (CPU & CUDA).
  Implemented on branch [intersect_benchmarks][5].
  
Tipps for building the benchmark suite
--------------------------------------

Follow the [general instructions to build Visionaray][7]. Make sure that you **build the examples** (cmake variable ```VSNRAY_ENABLE_EXAMPLES```! Use the cmake variable ```VSNRAY_ENABLE_CUDA``` to switch between CPU and GPU builds. Make sure to configure release builds: ```CMAKE_BUILD_TYPE=Release```.

The tests "CTP vs. sorting" and "CTP vs. kernel per primitive" can be found on branches [shading_benchmarks][3] and [intersect_benchmarks][5]. Activate the respective tests by manipulating the C preprocessor defines [SORT_BY_MATERIAL_TYPE][8] and [INTERSECT_PER_PRIMITIVE_TYPE][9] in ```pathtracer.h``` before compiling on the respective branch.

For the "kernel per primitive" tests, we add randomly distributed spheres to the scenes. Note that we hardcoded specific setups (e.g. sphere size) for the Conference Room and San Miguel test scenes. You may play around with the C preprocessor defines in [main.cpp][10] for the various configurations.

Tipps for running the benchmarks
--------------------------------

Invoke the ```wavefront_pathtracer``` application with a command line similar to the following one:

```
src/examples/wavefront_pathtracer/wavefront_pathtracer data/models/model-dir/model-name.obj -camera=camera-file.txt -bvh=split -width=2560 -height=1024
```

The models we used for our tests can be found under ```data/models```, camera files can be found under ```/data/cameras```.

License
-------

* Visionaray is licensed under the MIT License (MIT)
* The benchmark suite uses code from https://gist.github.com/jappa/62f30b6da5adea60bad3 to support C++14 ```integer_sequence``` with CUDA 8.0 (C++11 only).
* The 3D models were downloaded from Morgan McGuires Meshes repository: http://casual-effects.com/data/index.html
  - Conference Room is in the Public Domain.
  - San-Miguel and Crytek Sponza are licensed under the CC BY 3.0 (http://creativecommons.org/licenses/by/3.0/).
  - See the ```readme.txt``` files under ```data/models``` for modifications we made to the 3D models.

[1]:    https://github.com/szellmann/visionaray
[2]:    https://github.com/szellmann/visionaray/commit/047e641ca5ee2cb35507923f109bcd9fe2a00b2e
[3]:    https://github.com/ukoeln-vis/ctpperf/tree/shading_benchmarks
[4]:    https://github.com/ukoeln-vis/ctpperf/tree/shading_benchmark_oop
[5]:    https://github.com/ukoeln-vis/ctpperf/tree/intersect_benchmarks
[6]:    https://github.com/ukoeln-vis/ctpperf/tree/intersect_benchmark_oop
[7]:    https://github.com/szellmann/visionaray/wiki/Getting-started
[8]:    https://github.com/ukoeln-vis/ctpperf/blob/shading_benchmarks/src/examples/wavefront_pathtracer/pathtracer.h#L10
[9]:    https://github.com/ukoeln-vis/ctpperf/blob/intersect_benchmarks/src/examples/wavefront_pathtracer/pathtracer.h#L11
[10]:   https://github.com/ukoeln-vis/ctpperf/blob/intersect_benchmarks/src/examples/wavefront_pathtracer/main.cpp#L664
