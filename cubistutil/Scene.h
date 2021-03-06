//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "../cuda/BufferView.h"
#include "../cuda/MaterialData.h"
#include "Aabb.h"
#include "Camera.h"
#include "Matrix.h"
#include "Preprocessor.h"
#include "cubistutilapi.h"

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>


namespace cubist
{


class Scene
{
public:
    CUBISTAPI Scene();
    CUBISTAPI ~Scene();
    struct MeshGroup
    {
        std::string                       name;
        Matrix4x4                         transform;

        std::vector<GenericBufferView>    indices;
        std::vector<BufferView<float3> >  positions;
        std::vector<BufferView<float3> >  normals;
        std::vector<BufferView<float2> >  texcoords;

        std::vector<int32_t>              material_idx;

        OptixTraversableHandle            gas_handle = 0;
        CUdeviceptr                       d_gas_output = 0;

        Aabb                              object_aabb;
        Aabb                              world_aabb;
    };


    CUBISTAPI void addCamera  ( const Camera& camera            )    { m_cameras.push_back( camera );   }
    CUBISTAPI void addMesh    ( std::shared_ptr<MeshGroup> mesh )    { m_meshes.push_back( mesh );      }
    CUBISTAPI void addMaterial( const MaterialData::Pbr& mtl    )    { m_materials.push_back( mtl );    }
    CUBISTAPI void addBuffer  ( const uint64_t buf_size, const void* data );
    CUBISTAPI void addImage(
                const int32_t width,
                const int32_t height,
                const int32_t bits_per_component,
                const int32_t num_components,
                const void*   data
                );
    CUBISTAPI void addSampler(
                cudaTextureAddressMode address_s,
                cudaTextureAddressMode address_t,
                cudaTextureFilterMode  filter_mode,
                const int32_t          image_idx
                );

    CUBISTAPI CUdeviceptr                    getBuffer ( int32_t buffer_index  )const;
    CUBISTAPI cudaArray_t                    getImage  ( int32_t image_index   )const;
    CUBISTAPI cudaTextureObject_t            getSampler( int32_t sampler_index )const;

    CUBISTAPI void                           finalize();
    CUBISTAPI void                           cleanup();

    CUBISTAPI Camera                                    camera()const;
    CUBISTAPI OptixPipeline                             pipeline()const              { return m_pipeline;   }
    CUBISTAPI const OptixShaderBindingTable*            sbt()const                   { return &m_sbt;       }
    CUBISTAPI OptixTraversableHandle                    traversableHandle() const    { return m_ias_handle; }
    CUBISTAPI cubist::Aabb                               aabb() const                 { return m_scene_aabb; }
    CUBISTAPI OptixDeviceContext                        context() const              { return m_context;    }
    CUBISTAPI const std::vector<MaterialData::Pbr>&     materials() const            { return m_materials;  }
    CUBISTAPI const std::vector<std::shared_ptr<MeshGroup>>& meshes() const          { return m_meshes;     }

    CUBISTAPI void createContext();
    CUBISTAPI void buildMeshAccels( uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT );
    CUBISTAPI void buildInstanceAccel( int rayTypeCount = cubist::RAY_TYPE_COUNT );

private:
    void createPTXModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    // TODO: custom geometry support

    std::vector<Camera>                  m_cameras;
    std::vector<std::shared_ptr<MeshGroup> >  m_meshes;
    std::vector<MaterialData::Pbr>       m_materials;
    std::vector<CUdeviceptr>             m_buffers;
    std::vector<cudaTextureObject_t>     m_samplers;
    std::vector<cudaArray_t>             m_images;
    cubist::Aabb                          m_scene_aabb;

    OptixDeviceContext                   m_context                  = 0;
    OptixShaderBindingTable              m_sbt                      = {};
    OptixPipelineCompileOptions          m_pipeline_compile_options = {};
    OptixPipeline                        m_pipeline                 = 0;
    OptixModule                          m_ptx_module               = 0;

    OptixProgramGroup                    m_raygen_prog_group        = 0;
    OptixProgramGroup                    m_radiance_miss_group      = 0;
    OptixProgramGroup                    m_occlusion_miss_group     = 0;
    OptixProgramGroup                    m_radiance_hit_group       = 0;
    OptixProgramGroup                    m_occlusion_hit_group      = 0;
    OptixTraversableHandle               m_ias_handle               = 0;
    CUdeviceptr                          m_d_ias_output_buffer      = 0;
};


CUBISTAPI void loadScene( const std::string& filename, Scene& scene );

} // end namespace cubist

