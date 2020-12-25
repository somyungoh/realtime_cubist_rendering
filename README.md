# Realtime Cubist Rendering

### Building the project (Linux)
1. Install the latest `cuda` and `optix` from NVidia's website: https://developer.nvidia.com/optix
2. After the configuration (using `ccmake`), Before running `make`, add this entire repository under the `SDK` directory. The project directory name should be `optixCubistRender`: `NVDIA-Optix.../SDK/optixCubistRender`
3. Add `optixCubistRender` as build target along with the other projects in the root CMakelists.txt: 
  ```
  add_subdirectory( optixCubistRender )
  ```
4. Run `make`.
