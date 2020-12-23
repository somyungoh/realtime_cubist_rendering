#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# cubistutil source codes
set(cubistutil_sources
  
  cubistutil/Aabb.h
  cubistutil/Camera.cpp
  cubistutil/Camera.h
  cubistutil/CUDAOutputBuffer.h
  cubistutil/Exception.h
  cubistutil/GLDisplay.cpp
  cubistutil/GLDisplay.h
  cubistutil/Matrix.h
  cubistutil/PPMLoader.cpp
  cubistutil/PPMLoader.h
  cubistutil/Preprocessor.h
  cubistutil/Quaternion.h
  cubistutil/Record.h
  cubistutil/Scene.cpp
  cubistutil/Scene.h
  cubistutil/cubistutilapi.h
  cubistutil/cubistutil.cpp
  cubistutil/cubistutil.h
  cubistutil/Trackball.cpp
  cubistutil/Trackball.h
  cubistutil/vec_math.h
  )

# set optixCubstRender as SDK's project
OPTIX_add_sample_executable( optixCubistRender target_name
  
  optixCubistRender.cpp
  ${cubistutil_sources}

  OPTIONS -rdc true
  )
  
# link external libraries
target_link_libraries( ${target_name}

  ${CUDA_LIBRARIES}

  # required by cubistutil 
  ${GLFW_LIB_NAME}
  glad
  imgui
)
  
if( WIN32 )
  target_compile_definitions( ${target_name} PUBLIC GLAD_GLAPI_EXPORT )
endif()

# Use gcc rather than g++ to link if we are linking statically against libgcc_s and libstdc++
if(USING_GNU_C OR USING_GNU_CXX)
  if(GCC_LIBSTDCPP_HACK)
    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE "C")
    target_link_libraries(${target_name} LINK_PRIVATE ${STATIC_LIBSTDCPP})
    endif()
endif()

# haven't actually tested CUDA_NVRTC_ENABLED disabled...
# (it's done during the ccmake command configuration)
if(NOT CUDA_NVRTC_ENABLED)
  CUDA_COMPILE_PTX(ptx_files

  ../sampleConfig.h
  ../cuda/sphere.cu
  ../cuda/whitted.cu
  ../cuda/BufferView.h
  ../cuda/GeometryData.h
  ../cuda/Light.h
  ../cuda/LocalGeometry.h
  ../cuda/MaterialData.h
  ../cuda/util.h
  ../cuda/helpers.h
  
  ${cubistutil_sources}
  )
endif()

if(CUDA_NVRTC_ENABLED)
  target_link_libraries(${target_name} LINK_PRIVATE ${CUDA_nvrtc_LIBRARY})
endif()
if(WIN32)
  target_link_libraries(${target_name} LINK_PRIVATE winmm.lib)
endif()


