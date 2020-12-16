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
#include <optix.h>

#include "LocalGeometry.h"
#include "random.h"
#include "whitted.h"
#include "../util/vec_math.h"

#include "../util/vec_math.h"

// #include "whitted_cuda.h"

#define OPTIX_CUBIST_ENABLED

extern "C"
{
__constant__ whitted::LaunchParams params;
}


//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
// TODO: move into header so can be shared by path tracer and bespoke renderers
//
//------------------------------------------------------------------------------

__device__ float3 schlick( const float3 spec_color, const float V_dot_H )
{
    return spec_color + ( make_float3( 1.0f ) - spec_color ) * powf( 1.0f - V_dot_H, 5.0f );
}


__device__ float vis( const float N_dot_L, const float N_dot_V, const float alpha )
{
    const float alpha_sq = alpha*alpha;

    const float ggx0 = N_dot_L * sqrtf( N_dot_V*N_dot_V * ( 1.0f - alpha_sq ) + alpha_sq );
    const float ggx1 = N_dot_V * sqrtf( N_dot_L*N_dot_L * ( 1.0f - alpha_sq ) + alpha_sq );

    return 2.0f * N_dot_L * N_dot_V / (ggx0+ggx1);
}


__device__ float ggxNormal( const float N_dot_H, const float alpha )
{
    const float alpha_sq   = alpha*alpha;
    const float N_dot_H_sq = N_dot_H*N_dot_H;
    const float x          = N_dot_H_sq*( alpha_sq - 1.0f ) + 1.0f;
    return alpha_sq/( M_PIf*x*x );
}


__device__ float3 linearize( float3 c )
{
    return make_float3(
            powf( c.x, 2.2f ),
            powf( c.y, 2.2f ),
            powf( c.z, 2.2f )
            );
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        whitted::PayloadRadiance*   payload
        )
{
    unsigned int u0=0, u1=0, u2=0, u3=0;
    optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            whitted::RAY_TYPE_RADIANCE,        // SBT offset
            whitted::RAY_TYPE_COUNT,           // SBT stride
            whitted::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1, u2, u3 );

     payload->result.x = __int_as_float( u0 );
     payload->result.y = __int_as_float( u1 );
     payload->result.z = __int_as_float( u2 );
     payload->depth    = u3;
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    unsigned int occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            whitted::RAY_TYPE_OCCLUSION,      // SBT offset
            whitted::RAY_TYPE_COUNT,          // SBT stride
            whitted::RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded );
    return occluded;
}


__forceinline__ __device__ void setPayloadResult( float3 p )
{
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
}


__forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<int>( occluded ) );
}


__forceinline__ __device__ uchar4 make_color( const float3&  c )
{
    const float gamma = 2.2f;
    return make_uchar4(
            static_cast<int>( powf( clamp( c.x, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<int>( powf( clamp( c.y, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<int>( powf( clamp( c.z, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            255u
            );
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx      = optixGetLaunchIndex();
    const uint3  launch_dims     = optixGetLaunchDimensions();
    const float3 eye             = params.eye;
    const float3 U               = params.U;
    const float3 V               = params.V;
    const float3 W               = params.W;
    const int    subframe_index  = params.subframe_index;

    //
    // Generate camera ray
    //
    unsigned int seed = tea<4>( launch_idx.y*launch_dims.x + launch_idx.x, subframe_index );

    const float2 subpixel_jitter = subframe_index == 0 ?
        make_float2( 0.0f, 0.0f ) :
        make_float2( rnd( seed )-0.5f, rnd( seed )-0.5f );

    const float2 d = 2.0f * make_float2(
            ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
            ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y )
            ) - 1.0f;
    const float3 ray_direction = normalize(d.x*U + d.y*V + W);
    const float3 ray_origin    = eye;

    //
    // Trace camera ray
    //
    whitted::PayloadRadiance payload;
    payload.result = make_float3( 0.0f );
    payload.importance = 1.0f;
    payload.depth = 0.0f;

    traceRadiance(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload );

    //
    // Update results
    // TODO: timview mode
    //
    const int image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
    float3         accum_color  = payload.result;


    // :::::::::::::::::::::::::::::::  Multi-Camera ::::::::::::::::::::::::::::::::: //
    
#ifdef OPTIX_CUBIST_ENABLED

    float3   new_raydir = normalize (ray_direction + accum_color * 0.2);
    
    traceRadiance (
        params.handle,
        ray_origin,
        new_raydir,
        0.01f,  // tmin       // TODO: smarter offset
        1e16f,  // tmax
        &payload );
    
    accum_color = payload.result;

#endif

    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //


    if( subframe_index > 0 )
    {
        const float                 a = 1.0f / static_cast<float>( subframe_index+1 );
        const float3 accum_color_prev = make_float3( params.accum_buffer[ image_index ]);
        accum_color = lerp( accum_color_prev, accum_color, a );
    }
    params.accum_buffer[ image_index ] = make_float4( accum_color, 1.0f);
    params.frame_buffer[ image_index ] = make_color ( accum_color );
}

// TODO (David): add a function that returns an environment map
//               instead of a constant color.

// :::::::::::::::::::::::::::::::  Multi-Camera ::::::::::::::::::::::::::::::::: //

extern "C" __global__ void __miss__env_radiance()
{    
    const float3 ray_dir = optixGetWorldRayDirection();
    //const float3 ray_org = optixGetWorldRayOrigin();

    float M_PI = 3.14159265358979323846;
    float u = 0.5f + atan2(ray_dir.x, ray_dir.z) / (2 * M_PI);
    float v = 0.5f - asin(ray_dir.y) / M_PI;

    float4 env_hit_color = tex2D<float4>(params.env_texture, u, v);
    float3 out_color;
    out_color.x = env_hit_color.x;
    out_color.y = env_hit_color.y;
    out_color.z = env_hit_color.z;
        
    setPayloadResult( out_color );
}
// :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //


extern "C" __global__ void __miss__constant_radiance()
{    
    setPayloadResult( params.miss_color );
}

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    const whitted::HitGroupData* hit_group_data = reinterpret_cast<whitted::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );

    //
    // Retrieve material data
    //
    float3 base_color = make_float3( hit_group_data->material_data.pbr.base_color );
    if( hit_group_data->material_data.pbr.base_color_tex )
        base_color *= linearize( make_float3(
                    tex2D<float4>( hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y )
                    ) );

    float metallic  = hit_group_data->material_data.pbr.metallic;
    float roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex = make_float4( 1.0f );
    if( hit_group_data->material_data.pbr.metallic_roughness_tex )
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = tex2D<float4>( hit_group_data->material_data.pbr.metallic_roughness_tex, geom.UV.x, geom.UV.y );
    roughness *= mr_tex.y;
    metallic  *= mr_tex.z;


    //
    // Convert to material params
    //
    const float  F0         = 0.04f;
    const float3 diff_color = base_color*( 1.0f - F0 )*( 1.0f - metallic );
    const float3 spec_color = lerp( make_float3( F0 ), base_color, metallic );
    const float  alpha      = roughness*roughness;

    //
    // compute direct lighting
    //

    float3 N = geom.N;
    if( hit_group_data->material_data.pbr.normal_tex )
    {
        const float4 NN = 2.0f*tex2D<float4>( hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y ) - make_float4(1.0f);
        N = normalize( NN.x*normalize( geom.dpdu ) + NN.y*normalize( geom.dpdv ) + NN.z*geom.N );
    }

    float3 result = make_float3( 0.0f );

    for( int i = 0; i < params.lights.count; ++i )
    {
        Light light = params.lights[i];

        // TODO: optimize
        const float  L_dist  = length( light.point.position - geom.P );
        const float3 L       = ( light.point.position - geom.P ) / L_dist;
        const float3 V       = -normalize( optixGetWorldRayDirection() );
        const float3 H       = normalize( L + V );
        const float  N_dot_L = dot( N, L );
        const float  N_dot_V = dot( N, V );
        const float  N_dot_H = dot( N, H );
        const float  V_dot_H = dot( V, H );

        if( N_dot_L > 0.0f && N_dot_V > 0.0f )
        {
            const float tmin     = 0.001f;          // TODO
            const float tmax     = L_dist - 0.001f; // TODO
            const bool  occluded = traceOcclusion( params.handle, geom.P, L, tmin, tmax );
            if( !occluded )
            {
                const float3 F     = schlick( spec_color, V_dot_H );
                const float  G_vis = vis( N_dot_L, N_dot_V, alpha );
                const float  D     = ggxNormal( N_dot_H, alpha );

                const float3 diff = ( 1.0f - F )*diff_color / M_PIf;
                const float3 spec = F*G_vis*D;

                result += light.point.color*light.point.intensity*N_dot_L*( diff + spec );
            }
        }
    }
    // TODO: add debug viewing mode that allows runtime switchable views of shading params, normals, etc
    //result = make_float3( roughness );
    //result = N*0.5f + make_float3( 0.5f );
    //result = geom.N*0.5f + make_float3( 0.5f );
    setPayloadResult( result );
}
