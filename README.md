# Realtime Cubist Rendering

Realtime Cubist Rendering project with several folks from the Department of Visualization, Texas A&M University. This project is made on top of NVidia Optix SDK. The latest SDK version tested is `7.2.0 (October 2020) build`.

### Building the project: Linux
1. Install the latest `cuda` and `optix` from NVidia's website: https://developer.nvidia.com/optix
2. After the CMake configuration, (the step where you use `ccmake`), add this entire repository under the `SDK` directory. The project directory name **should be exactly `optixCubistRender`**: `NVDIA-Optix.../SDK/optixCubistRender`
3. Add `optixCubistRender` as a build target along with the other projects in the root CMakelists.txt: 
  ```
  add_subdirectory( optixCubistRender )
  ```
4. Run `make` to build the SDK, including this project.
5. You'll find the executable `optixCubistRender` inside the `bin`.

### Building the project: Windows
Sorry, I haven't tested in Windows...


