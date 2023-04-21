﻿//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#ifndef CUPROG_H
#define CUPROG_H

#include <sutil/vec_math.h>
#include <cmath>

#include "whitted.h"
#include"optixPathTracer.h"
#include"BDPTVertex.h" 
#include"decisionTree/classTree_device.h"
#define SCENE_EPSILON 1e-3f


#define ISINVALIDVALUE(ans) (ans.x>100000.0f|| isnan(ans.x)||ans.y>100000.0f|| isnan(ans.y)||ans.z>100000.0f|| isnan(ans.z))
#define VERTEX_MAT(v) (v.getMat(Tracer::params.materials))
struct labelUnit
{
    float3 position;
    float3 normal;
    float3 dir;

    float2 uv;
    float2 tri_uv;
    int objectId;
    int tri_id;
    int type;
    bool light_side;
    RT_FUNCTION labelUnit(float3 position, float3 normal, float3 dir, float2 uv, float2 tri_uv, int objectId, int tri_id, bool light_side, int type = 0) :
        position(position), normal(normal), dir(dir), uv(uv), tri_uv(tri_uv), objectId(objectId), tri_id(tri_id), light_side(light_side), type(type) {}
    RT_FUNCTION labelUnit(float3 position, float3 normal, float3 dir, bool light_side) : position(position), normal(normal), dir(dir), light_side(light_side) {}

    RT_FUNCTION int getLabel();

};

 
RT_FUNCTION bool refract(float3& r, float3 i, float3 n, float ior)
{ 
    float eta = ior;


    float3 nn = n;
    float3 ii = -i;
    float negNdotV = dot(ii, nn);

    if (negNdotV > 0.0f)
    {
        eta = ior;
        nn = -n;
        negNdotV = -negNdotV;
    }
    else
    {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

    if (k < 0.0f) {
        // Initialize this value, so that r always leaves this function initialized.
        r = make_float3(0.f);
        return false;
    }
    else {
        r = normalize(eta * ii - (eta * negNdotV + sqrtf(k)) * nn);
        return true;
    }


    //if (dot(n, i) > 0) eta = 1 / ior;
    //float cosThetaI = dot(n, i);
    //float sin2ThetaI = 1 - cosThetaI * cosThetaI;
    //float sin2ThetaT = eta * eta * sin2ThetaI;
    //if (sin2ThetaT >= 1)
    //{
    //    return false;
    //}

    //float cosThetaT = sqrt(1 - sin2ThetaT);
    //
    //r = eta * -i + (eta * cosThetaI - cosThetaT) * n;

    ////printf("length test %f\n", length(r));
    //return true;
}
RT_FUNCTION bool isRefract(float3 normal, float3 in_dir, float3 out_dir)
{
    return dot(normal, in_dir) * dot(normal, out_dir) < 0;
}
namespace Tracer {
    using namespace whitted;

    extern "C" {
        __constant__ MyParams params;
    }
}

RT_FUNCTION float connectRate_SOL(int eye_label, int light_label, float lum_sum)
{
    return Tracer::params.subspace_info.gamma_ss(eye_label, light_label) * lum_sum * CONNECTION_N;
}

RT_FUNCTION float3 connectRate_SOL(int eye_label, int light_label, float3 lum_sum)
{
    return Tracer::params.subspace_info.gamma_ss(eye_label, light_label) * lum_sum * CONNECTION_N;
}


struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    __forceinline__ __device__ void transform(float3& p) const
    {
        p = make_float3(dot(p, m_tangent), dot(p, m_binormal), dot(p, m_normal));
    }
    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
    // Uniformly sample disk.
    const float r = sqrtf(u1);
    const float phi = 2.0f * M_PIf * u2;
    p.x = r * cosf(phi);
    p.y = r * sinf(phi);

    // Project up to hemisphere.
    p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}
RT_FUNCTION float2 sample_reverse_cosine(float3 N, float3 dir)
{
    Onb onb(N);
    onb.transform(dir);

    float r1 = dir.x * dir.x + dir.y * dir.y;
    float r = sqrtf(r1);
    float cosPhi = dir.x / r;
    float sinPhi = dir.y / r;
    float phiA = acos(cosPhi);
    float phiB = asin(sinPhi);
    float phi = phiB > 0 ? phiA : 2 * M_PI - phiA;

    float r2 = phi / 2.0f / M_PIf; 
    return make_float2(r1, r2);
}
struct envInfo_device :envInfo
{

    RT_FUNCTION int coord2index(int2 coord)const
    {
        return coord.x + coord.y * width;
    }
    RT_FUNCTION int2 index2coord(int index)const
    {
        int w = index % width;
        int h = index / width;
        return make_int2(w, h);
    }
    RT_FUNCTION float2 coord2uv(int2 coord)const
    {
        float u, v;
        u = coord.x / float(width);
        v = coord.y / float(height);
        return make_float2(u, v);
    }
    RT_FUNCTION int2 uv2coord(float2 uv)const
    {
        int x = uv.x * width;
        int y = uv.y * height;
        x = min(x, width - 1);
        y = min(y, height - 1);
        return make_int2(x, y);
    }
    RT_FUNCTION float2 coord2uv(int2 coord, unsigned int& seed)const
    {
        float r1 = rnd(seed), r2 = rnd(seed);
        float u, v;
        u = float(coord.x + r1) / float(width);
        v = float(coord.y + r2) / float(height);
        return make_float2(u, v);
    }
    RT_FUNCTION float3 reverseSample(float2 uv)const
    {
        return uv2dir(uv);
    }
    RT_FUNCTION float3 sample(unsigned int& seed)const
    {
        float index = rnd(seed);
        int mid = size / 2 - 1, l = 0, r = size;
        while (r - l > 1)
        {
            if (index < cmf[mid])
            {
                r = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + r) / 2 - 1;
        }
        int2 coord = index2coord(l);        
        float2 uv = coord2uv(coord, seed);
        uv.y = 1-uv.y;
        return uv2dir(uv);
    }
    RT_FUNCTION float3 sample_projectPos(float3 dir, unsigned int& seed)const
    {
        const float r1 = rnd(seed);
        const float r2 = rnd(seed);
        float3 pos;
        Onb onb(dir);
        cosine_sample_hemisphere(r1, r2, pos);

        return 10 * r * (dir) + pos.x * r * onb.m_tangent + pos.y * r * onb.m_binormal + center;
    }
    RT_FUNCTION float2 trace_reverse_uv(float3 position, float3 dir)const
    {
        float3 vec2center = center - position;
        float3 cosThetaProject = dot(vec2center, dir) * dir;
        float3 projectPos_scale = (position + cosThetaProject  - center) / r;

        return sample_reverse_cosine(dir, projectPos_scale);
//        return make_float2(0.5, 0.5);
    }
    RT_FUNCTION float projectPdf()const
    {
        return 1 / (M_PI * r * r);
    }

    RT_FUNCTION int getLabel(float3 dir)const
    {
        float2 uv = dir2uv(dir);
        int2 coord = uv2coord(uv);
        int index = coord2index(coord);


        int2 uv_div = make_int2(
            clamp(static_cast<int>(floorf(uv.x * divLevel)),int(0), int(divLevel - 1)),
            clamp(static_cast<int>(floorf(uv.y * divLevel)),int(0), int(divLevel - 1))           
            );

        int res_id = uv_div.x * divLevel + uv_div.y; 

        return NUM_SUBSPACE - 1 - res_id;
    }
    RT_FUNCTION float3 getColor(float3 dir)const
    {
        float2 uv = dir2uv(dir); 
       // return make_float3(pdf(dir)) * 5;
        return make_float3(tex2D<float4>(tex, uv.x, uv.y)); 
    }

    RT_FUNCTION float3 color(float3 dir)const
    {
//        printf("dir sample %f %f %f, %f %f %f\n", dir.x, dir.y, dir.z, getColor(dir).x, getColor(dir).y, getColor(dir).z);
        return getColor(dir);
    } 
    RT_FUNCTION float pdf(float3 dir)const
    {
        float2 uv = dir2uv(dir);
        uv.y = 1-uv.y;
        int2 coord = uv2coord(uv);
        int index = coord2index(coord);

        float pdf1 = index == 0 ? cmf[index] : cmf[index] - cmf[index - 1];

        //if (luminance(color(dir)) / (pdf1 * size / (4 * M_PI)) > 1000)
        //{
            //rtPrintf("%d %d\n", coord.x,coord.y); 
            //return 1000;
        //}
        return pdf1 * size / (4 * M_PI);
    } 
};

#define SKY (*(reinterpret_cast<const envInfo_device*>(&Tracer::params.sky)))


namespace Tracer {
    RT_FUNCTION int binary_sample(float* cmf, int size, unsigned int& seed, float& pmf, float cmf_range = 1.0)
    {

        float index = rnd(seed) * cmf_range;
        int mid = size / 2 - 1, l = 0, r = size;
        while (r - l > 1)
        {
            if (index < cmf[mid])
            {
                r = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + r) / 2 - 1;
        }
        pmf = l == 0 ? cmf[l] : cmf[l] - cmf[l - 1];
        return l;
    }

    struct SubspaceSampler_device :public SubspaceSampler
    {
        RT_FUNCTION const BDPTVertex& sampleSecondStage(int subspaceId, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = subspace[subspaceId].jump_bias;
            int end_index = begin_index + subspace[subspaceId].size;
            //sample_pmf = 1.0 / subspace[subspaceId].size;
            //int index = rnd(seed) * subspace[subspaceId].size + begin_index;
            //printf("error %f %f %f\n", *(cmfs + begin_index), 1.0 / subspace[subspaceId].size, *(cmfs + end_index - 1));
            int index = binary_sample(cmfs + begin_index, subspace[subspaceId].size, seed, sample_pmf) + begin_index;
            //        if (params.lt.validState[index] == false)
            //            printf("error found");
                    //printf("index get %d %d\n",index, jump_buffer[index]);
            return LVC[jump_buffer[index]];
        }


        RT_FUNCTION const BDPTVertex& uniformSampleGlossy(unsigned int& seed, float& sample_pmf)
        {
            sample_pmf = 1.0 / glossy_count;
            int index = rnd(seed) * glossy_count;

            return LVC[glossy_index[index]];
        }

        RT_FUNCTION const BDPTVertex& SampleGlossySecondStage(int subspaceId, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = glossy_subspace_bias[subspaceId];
            int end_index = begin_index + glossy_subspace_num[subspaceId];

            sample_pmf = 1.0 / (end_index - begin_index);
            int index = rnd(seed) * (end_index - begin_index) + begin_index;

            return LVC[glossy_index[index]];
        }
        RT_FUNCTION const BDPTVertex& uniformSample(unsigned int& seed, float& sample_pmf)
        {
            sample_pmf = 1.0 / vertex_count;
            int index = rnd(seed) * vertex_count;

            return LVC[jump_buffer[index]];
        }
        RT_FUNCTION int sampleFirstStage(int eye_subsapce, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = eye_subsapce * NUM_SUBSPACE;
            int end_index = begin_index + NUM_SUBSPACE;
            int index = binary_sample(Tracer::params.subspace_info.CMFGamma + begin_index, NUM_SUBSPACE, seed, sample_pmf);
            //        if (params.lt.validState[index] == false)
            //            printf("error found");
                    //printf("index get %d %d\n",index, jump_buffer[index]); 
            //int index = int(rnd(seed) * NUM_SUBSPACE);
            //sample_pmf = 1.0 / NUM_SUBSPACE;
            return index;
        }

        RT_FUNCTION int SampleGlossyFirstStage(int eye_subsapce, unsigned int& seed, float& sample_pmf)
        {
            int begin_index = eye_subsapce * NUM_SUBSPACE;
            int end_index = begin_index + NUM_SUBSPACE;
            int index = binary_sample(Tracer::params.subspace_info.CMFCausticGamma + begin_index, NUM_SUBSPACE, seed, sample_pmf);
            return index;
        }
    };
    struct PayloadBDPTVertex
    {//记得初始化
        BDPTPath path;
        float3 origin;
        float3 ray_direction;
        float3 throughput;
        float3 result;
        float pdf;
        unsigned int seed;
        int depth;
        bool done;
        long long path_record;
        RT_FUNCTION void clear()
        {
            path.clear();
            depth = 0;
            done = false;
            throughput = make_float3(1);
            result = make_float3(0.0);
            path_record = 0;
        }
    };

    //------------------------------------------------------------------------------
    //
    // GGX/smith shading helpers
    // TODO: move into header so can be shared by path tracer and bespoke renderers
    //
    //------------------------------------------------------------------------------

    __device__ __forceinline__ float3 schlick(const float3 spec_color, const float V_dot_H)
    {
        return spec_color + (make_float3(1.0f) - spec_color) * powf(1.0f - V_dot_H, 5.0f);
    }

    __device__ __forceinline__ float vis(const float N_dot_L, const float N_dot_V, const float alpha)
    {
        const float alpha_sq = alpha * alpha;

        const float ggx0 = N_dot_L * sqrtf(N_dot_V * N_dot_V * (1.0f - alpha_sq) + alpha_sq);
        const float ggx1 = N_dot_V * sqrtf(N_dot_L * N_dot_L * (1.0f - alpha_sq) + alpha_sq);

        return 2.0f * N_dot_L * N_dot_V / (ggx0 + ggx1);
    }


    __device__ __forceinline__ float ggxNormal(const float N_dot_H, const float alpha)
    {
        const float alpha_sq = alpha * alpha;
        const float N_dot_H_sq = N_dot_H * N_dot_H;
        const float x = N_dot_H_sq * (alpha_sq - 1.0f) + 1.0f;
        return alpha_sq / (M_PIf * x * x);
    }

    __device__ __forceinline__ float float3sum(float3 c)
    {
        return c.x + c.y + c.z;
    }

    __device__ __forceinline__ float3 linearize(float3 c)
    {
        return make_float3(
            powf(c.x, 2.2f),
            powf(c.y, 2.2f),
            powf(c.z, 2.2f)
        );
    }


    //------------------------------------------------------------------------------
    //
    //
    //
    //------------------------------------------------------------------------------


    static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
    {
        const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }
    static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        whitted::PayloadRadiance* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_RADIANCE,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1);

    }
    static __forceinline__ __device__ void traceLightSubPath(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        PayloadBDPTVertex* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_LIGHTSUBPATH,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_LIGHTSUBPATH,        // missSBTIndex
            u0, u1);

    }
    static __forceinline__ __device__ void traceEyeSubPath(
        OptixTraversableHandle      handle, 
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        PayloadBDPTVertex* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_EYESUBPATH,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            //RayType::RAY_TYPE_EYESUBPATH,        // SBT offset
            //RayType::RAY_TYPE_COUNT,           // SBT stride
            //RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            u0, u1
        );

    }

    static __forceinline__ __device__ void traceEyeSubPath_simple(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        PayloadBDPTVertex* payload
    )
    {
        unsigned int u0, u1;
        packPointer(payload, u0, u1);
        optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
            RayType::RAY_TYPE_EYESUBPATH_SIMPLE,        // SBT offset
            RayType::RAY_TYPE_COUNT,           // SBT stride
            RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            //RayType::RAY_TYPE_EYESUBPATH,        // SBT offset
            //RayType::RAY_TYPE_COUNT,           // SBT stride
            //RayType::RAY_TYPE_EYESUBPATH,        // missSBTIndex
            u0, u1);

    }

    RT_FUNCTION bool visibilityTest(
        OptixTraversableHandle handle, float3 pos_A, float3 pos_B)
    {
        float3 bias_pos = pos_B - pos_A;
        float len = length(bias_pos);
        float3 dir = bias_pos / len;
        unsigned int u0 = __float_as_uint(1.f);
        optixTrace(
            handle,
            pos_A,
            dir,
            SCENE_EPSILON,
            len - SCENE_EPSILON,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RayType::RAY_TYPE_OCCLUSION,      // SBT offset
            RayType::RAY_TYPE_COUNT,          // SBT stride
            RayType::RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0);
        float res = __uint_as_float(u0);
        if (res > 0.5)
            return true;
        return false;
    }

    RT_FUNCTION bool visibilityTest(
        OptixTraversableHandle handle, const BDPTVertex& eyeVertex, const BDPTVertex& lightVertex)
    {
        if (lightVertex.is_DIRECTION())
        {
            float3 n_pos = -10 * SKY.r * lightVertex.normal + eyeVertex.position;
            return visibilityTest(handle, eyeVertex.position, n_pos);
        }
        else
        {
            return visibilityTest(handle, eyeVertex.position, lightVertex.position);
        }

    }

    __forceinline__ __device__ unsigned int getPayloadDepth()
    {
        return optixGetPayload_3();
    }

    static __forceinline__ __device__ float traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
    )
    {
        unsigned int u0 = __float_as_uint(1.f);
        optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            RayType::RAY_TYPE_OCCLUSION,      // SBT offset
            RayType::RAY_TYPE_COUNT,          // SBT stride
            RayType::RAY_TYPE_OCCLUSION,      // missSBTIndex
            u0);
        return __uint_as_float(u0);
    }


    __forceinline__ __device__ void setPayloadResult(float3 p)
    {
        optixSetPayload_0(__float_as_uint(p.x));
        optixSetPayload_1(__float_as_uint(p.y));
        optixSetPayload_2(__float_as_uint(p.z));
    }

    __forceinline__ __device__ float getPayloadOcclusion()
    {
        return __uint_as_float(optixGetPayload_0());
    }

    __forceinline__ __device__ void setPayloadOcclusion(float attenuation)
    {
        optixSetPayload_0(__float_as_uint(attenuation));
    }


#define RT_FUNCTION __device__ __forceinline__  
    struct lightSample
    {
        float3 position;
        float3 emission;
        float3 direction;
        float2 uv;
        const Light* bindLight;
        float pdf;
        union
        {
            float dir_pdf;
            float dir_pos_pdf;
        };
        int subspaceId;


        Light::Type type;
        RT_FUNCTION void ReverseSample(const Light& light, float2 uv_)
        {
            bindLight = &light;
            if (light.type == Light::Type::QUAD)
            {
                float r1 = uv_.x;
                float r2 = uv_.y;
                float r3 = 1 - r1 - r2;
                //printf("random %f %f\n", r1, r2);
                // 对四边形进行采样，已经经过验证
                position = light.quad.u * r1 + light.quad.v * r2 + light.quad.corner * r3;
                emission = light.quad.emission;
                pdf = 1.0 / light.quad.area;
                pdf /= params.lights.count;
                uv = make_float2(r1, r2);
                {
                    int x_block = clamp(static_cast<int>(floorf(uv.x * light.divLevel)), int(0), int(light.divLevel - 1));
                    int y_block = clamp(static_cast<int>(floorf(uv.y * light.divLevel)), int(0), int(light.divLevel - 1));
                    int lightSpaceId = light.ssBase + x_block * light.divLevel + y_block;
                    subspaceId = NUM_SUBSPACE - lightSpaceId - 1;
                }
            }
            else if (light.type == Light::Type::ENV)
            {
                direction = SKY.reverseSample(uv_);
                emission = SKY.color(direction);
                subspaceId = SKY.getLabel(direction);
                uv = uv;
                pdf = SKY.pdf(direction);
                pdf /= params.lights.count;
            }
        }
        RT_FUNCTION void operator()(const Light& light, unsigned int& seed)
        {
            if (light.type == Light::Type::QUAD)
            {
                float r1 = rnd(seed);
                float r2 = rnd(seed);
                float r3 = 1 - r1 - r2;
                ReverseSample(light, make_float2(r1, r2));
            }
            else if (light.type == Light::Type::ENV)
            {
                direction = SKY.sample(seed);
                emission = SKY.color(direction);
                subspaceId = SKY.getLabel(direction);
                uv = dir2uv(direction);
                pdf = SKY.pdf(direction);
                pdf /= params.lights.count;
            }
            bindLight = &light;
        }
        RT_FUNCTION void operator()(unsigned int& seed)
        {
            int light_id = clamp(static_cast<int>(floorf(rnd(seed) * params.lights.count)), int(0), int(Tracer::params.lights.count - 1));
            this->operator()(params.lights[light_id], seed);

        }
        RT_FUNCTION float3 normal()
        {
            if (bindLight)
            {
                if (bindLight->type == Light::Type::QUAD)
                {
                    return bindLight->quad.normal;
                }
                else
                {
                    return -direction;
                }
            }

            return make_float3(0);
        }
        RT_FUNCTION float3 trace_direction()const
        {
            return (bindLight->type == Light::Type::ENV) ? -direction : direction;
        }
        RT_FUNCTION void traceMode(unsigned int& seed)
        {
            if (bindLight->type == Light::Type::QUAD)
            {
                float r1 = rnd(seed);
                float r2 = rnd(seed);

                Onb onb(bindLight->quad.normal);
                cosine_sample_hemisphere(r1, r2, direction);
                onb.inverse_transform(direction);
                dir_pdf = abs(dot(direction, bindLight->quad.normal)) / M_PIf;
            }
            else if (bindLight->type == Light::Type::ENV)
            {
                position = SKY.sample_projectPos(direction, seed);
                dir_pos_pdf = SKY.projectPdf();
            }
        }
    };

    static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
    {
        const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    template<typename T = PayloadRadiance>
    static __forceinline__ __device__ T* getPRD()
    {
        const unsigned int u0 = optixGetPayload_0();
        const unsigned int u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

#include<cuda.h>
    RT_FUNCTION float sqr(float x) { return x * x; }

    RT_FUNCTION float SchlickFresnel(float u)
    {
        float m = clamp(1.0f - u, 0.0f, 1.0f);
        float m2 = m * m;
        return m2 * m2 * m; // pow(m,5)
    }

    RT_FUNCTION float GTR1(float NDotH, float a)
    {
        if (a >= 1.0f) return (1.0f / M_PIf);
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
        return (a2 - 1.0f) / (M_PIf * logf(a2) * t);
    }

    RT_FUNCTION float GTR2(float NDotH, float a)
    {
        float a2 = a * a;
        float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
        return a2 / (M_PIf * t * t);
    }

    RT_FUNCTION float smithG_GGX(float NDotv, float alphaG)
    {
        float a = alphaG * alphaG;
        float b = NDotv * NDotv;
        return 1.0f / (NDotv + sqrtf(a + b - a * b));
    }

    RT_FUNCTION float D(const float3& wh, const float3& n)
    {
        //BeckmannDistribution
        const float alphax = 0.5f;
        const float alphay = 0.5f;// what does this mean?
        float wDotn = dot(wh, n);
        float TanTheta = length(cross(wh, n)) / wDotn;
        float tan2Theta = TanTheta * TanTheta;
        if (isinf(tan2Theta)) return 0.;
        float cos4Theta = wDotn * wDotn * wDotn * wDotn;
        return exp(-tan2Theta * (1.0f / (alphax * alphax))) /
            (M_PIf * alphax * alphay * cos4Theta);
    }

    RT_FUNCTION float fresnel(float cos_theta_i, float cos_theta_t, float eta)
    {
        const float rs = (cos_theta_i - cos_theta_t * eta) /
            (cos_theta_i + eta * cos_theta_t);
        const float rp = (cos_theta_i * eta - cos_theta_t) /
            (cos_theta_i * eta + cos_theta_t);

        return 0.5f * (rs * rs + rp * rp);
    }

    RT_FUNCTION float3 logf(float3 v)
    {
        return make_float3(log(v.x), log(v.y), log(v.z));
    }
    RT_FUNCTION float lerp(const float& a, const float& b, const float t)
    {
        return a + t * (b - a);
    }

    RT_FUNCTION float Lambda(const float3& w, const float3& n)
    {
        //BeckmannDistribution
        const float alphax = 0.5f;
        const float alphay = 0.5f;// what does this mean?
        float wDotn = dot(w, n);
        float absTanTheta = abs(length(cross(w, n)) / wDotn);
        if (isinf(absTanTheta)) return 0.;
        // Compute _alpha_ for direction _w_
        float alpha = alphax;
        //std::sqrt(wDotn * wDotn * alphax * alphax + (1 - wDotn * wDotn) * alphay * alphay);
        float a = 1 / (alpha * absTanTheta);
        if (a >= 1.6f) return 0;
        return (1 - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
    }

}
namespace Shift
{

#define ROUGHNESS_A_LIMIT 0.001f
    RT_FUNCTION float2 sample_reverse_half(float a, float3 half)
    {
        a = max(ROUGHNESS_A_LIMIT, a);
        float cosTheta = half.z;
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = half.y / sinTheta;
        float cosPhi = half.x / sinTheta;
        float phiA = acos(cosPhi);
        float phiB = asin(sinPhi);
        float phi = phiB > 0 ? phiA : 2 * M_PI - phiA;

        float r1 = phi / 2.0f / M_PIf;
        float A = cosTheta * cosTheta;

        float r2 = (1 - A) / (A * a * a - A + 1);
        return make_float2(r1, r2);
    }

    RT_FUNCTION float2 sample_reverse_half(float a, const float3& N, float3 half)
    {
        Onb onb(N);
        onb.transform(half);
        return sample_reverse_half(a, half);
    }
    RT_FUNCTION float3 sample_half(float a, const float2 uv)
    {
        a = max(ROUGHNESS_A_LIMIT, a);

        float r1 = uv.x;
        float r2 = uv.y;
        float phi = r1 * 2.0 * M_PIf;
        float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
        float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
        float sinPhi = sinf(phi);
        float cosPhi = cosf(phi);

        float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
        return half;
    }

    RT_FUNCTION float3 sample_half(float a, const float3& N, const float2 uv)
    {
        Onb onb(N);
        float3 half = sample_half(a, uv);
        onb.inverse_transform(half);
        return half;
    }

    //eta 无论是否取倒数，结果都一样
    //wh 无论是朝着V还是L的方向，结果也都一样
    RT_FUNCTION float dwh_dwi_refract(float3 wh, float3 V, float3 L, float eta)
    {
        float sqrtDenom = dot(L, wh) + eta * dot(V, wh);
        float dwh_dwi =
            std::abs((eta * eta * dot(V, wh)) / (sqrtDenom * sqrtDenom));
        return dwh_dwi;
    }

    RT_FUNCTION float3 refract_half_fine(float3 in_dir, float3 out_dir, float3 normal, float eta)
    {
        if (dot(normal, in_dir) < 0) eta = 1 / eta;
        float3 ans;
        ans = eta * out_dir + in_dir;
        float3 half = normalize(ans);
        return dot(half, normal) > 0 ? half : -half;
    }
    RT_FUNCTION float3 refract_half(float3 in_dir, float3 out_dir, float3 normal, float eta)
    {
        eta = max(eta, 1 / eta);
        float3 ans;
        if (abs(dot(in_dir, normal)) > abs(dot(out_dir, normal)))
        {
            ans = eta * in_dir + out_dir;
        }
        else
        {
            ans = in_dir + eta * out_dir;
        }
        ans = normalize(ans);
        return dot(ans, normal) > 0 ? ans : -ans;
    }

}
namespace Tracer
{
    RT_FUNCTION float3 Eval_Transmit(const MaterialData::Pbr& mat, const float3& normal, const float3& V_vec, const float3& L_vec)
    {
        float3 N = normal;
        float3 V = V_vec;
        float3 L = L_vec;
        float NDotL = dot(N, L);
        float NDotV = dot(N, V);


        float mateta = mat.eta;
        float eta = mateta;
        if (NDotL > 0)
        {
            eta = 1 / eta;
            N = -N;
        }

        if (NDotL == 0 || NDotV == 0) return make_float3(0);
        float refract;
        if ((1 - NDotV * NDotV) / (eta * eta) >= 1)// ȫ����
            refract = 1;
        else
            refract = 0;
        float3 Cdlin = make_float3(mat.base_color);
        float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.
        float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
        float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);

        // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
        float3 wh = normalize(L * eta + V);
        if (dot(wh, N) < 0) wh = -wh;

        // Same side?
        if (dot(L, wh) * dot(V, wh) > 0) return make_float3(0);

        float sqrtDenom = eta * dot(L, wh) + dot(V, wh);
        float factor = 1;// modify
        float3 T = mat.trans * make_float3(sqrt(mat.base_color.x), sqrt(mat.base_color.y), sqrt(mat.base_color.z));
        //printf("%f\n", mat.trans);
        float roughg = sqr(mat.roughness * 0.5f + 0.5f);
        float Gs = 1 / (1 + Lambda(V, N) + Lambda(L, N));
        //float Gs = smithG_GGX(abs(NDotL), roughg) * smithG_GGX(abs(NDotV), roughg);
        float a = max(0.001f, mat.roughness);
        float Ds = GTR2(dot(N, wh), a);//D(wh, N);
        //GTR2(dot(wh,N), a);
        float FH = SchlickFresnel(dot(V, wh));
        float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
        //float F = fresnel(abs(dot(V, wh)), abs(dot(L, wh)), eta);
        //printf("Fresnel: %f\n", F);
        float F = 0;
        float cosI = abs(NDotV);
        float sin2T = (1 - NDotV * NDotV) / (eta * eta);

        if (sin2T <= 1)
        {
            float cosT = sqrt(1 - sin2T);
            F = fresnel(cosI, cosT, eta);
        }
        //if(NDotL<-0.8)
            //printf("NDotV:%f, NDotL:%f, NDotWh:%f, Ds:%f\n", NDotV, NDotL, dot(wh, N), Ds);
        float3 out = (1 - refract) * (1.f - F) * T *
            std::abs(Ds * Gs * eta * eta *
                abs(dot(L, wh)) * abs(dot(V, wh)) * factor * factor /
                (sqrtDenom * sqrtDenom));
        /*
        if (isRefract(N, V, L))
        {
            out *= eta * eta;
        }
        */
        //if(out.x!=0)
        //    printf("trans: %f,%f,%f\n", out.x, out.y, out.z); 
        return out;

    }
    RT_FUNCTION float3 Eval(const MaterialData::Pbr& mat, const float3& normal, const float3& V, const float3& L)
    {

        float3 N = normal;

        float mateta = mat.eta;
        float NDotL = dot(N, L);
        float NDotV = dot(N, V);
        float eta = 1 / mateta;
        if (NDotL * NDotV <= 0.0f)
            return Eval_Transmit(mat, normal, V, L);

        //return make_float3(0);

        if (NDotL < 0.0f && NDotV < 0.0f)
        {
            N = -normal;
            eta = 1 / eta;
            NDotL *= -1;
            NDotV *= -1;
        }
        float3 H = normalize(L + V);
        float NDotH = dot(N, H);
        float LDotH = dot(L, H);
        float VDotH = dot(V, H);

        float refract;
        if ((1 - NDotV * NDotV) * eta * eta >= 1)// ȫ����
            refract = 1;
        else
            refract = 0;

        float3 Cdlin = make_float3(mat.base_color);
        float Cdlum = 0.3f * Cdlin.x + 0.6f * Cdlin.y + 0.1f * Cdlin.z; // luminance approx.

        float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
        float3 Cspec0 = lerp(mat.specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
        float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
        // and mix in diffuse retro-reflection based on roughness
        float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
        float Fd90 = 0.5f + 2.0f * LDotH * LDotH * mat.roughness;
        float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

        // Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
        // 1.25 scale is used to (roughly) preserve albedo
        // Fss90 used to "flatten" retroreflection based on roughness
        float Fss90 = LDotH * LDotH * mat.roughness;
        float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
        float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

        // specular
        //float aspect = sqrt(1-mat.anisotrokPic*.9);
        //float ax = Max(.001f, sqr(mat.roughness)/aspect);
        //float ay = Max(.001f, sqr(mat.roughness)*aspect);
        //float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);

        float a = max(0.001f, mat.roughness);
        float Ds = GTR2(NDotH, a);
        float FH = SchlickFresnel(LDotH);
        float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
        float roughg = sqr(mat.roughness * 0.5f + 0.5f);
        float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

        // sheen
        float3 Fsheen = FH * mat.sheen * Csheen;

        // clearcoat (ior = 1.5 -> F0 = 0.04)
        float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
        float Fr = lerp(0.04f, 1.0f, FH);
        float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

        float trans = mat.trans;

        float cosThetaI = abs(dot(N, V));
        float sin2ThetaI = 1 - cosThetaI * cosThetaI;
        float sin2ThetaT = eta * eta * sin2ThetaI;
        float cosThetaT = 1;
        if (sin2ThetaT <= 1)
        {
            cosThetaT = sqrt(1 - sin2ThetaT);
        }
        float F = fresnel(cosThetaI, cosThetaT, eta);

        float3 out = (((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface) * Cdlin + Fsheen)
            * (1.0f - mat.metallic))
            * (1 - trans * (1 - F) * (1 - refract))
            ;
        if (trans > 0)
            out = out + Gs * Ds * (1 - trans * (1 - refract) * (1 - F));
        else
            out = out + Gs * Ds * Fs;
        //printf("%f %f\n", cosThetaI,F);
            //+ Gs * Ds * (1 - trans * F);// (1 - (1 - F));// *(1 - refract));
            //+ 0.25f * mat.clearcoat * Gr * Fr * Dr;
        //printf("eval: %f,%f,%f\n", out.x, out.y, out.z);
        return out;
    }
    RT_FUNCTION float3 Sample(const MaterialData::Pbr& mat, const float3& N, const float3& V, unsigned int& seed, float3 position = make_float3(0.0), bool use_pg = false)
    {
        if (use_pg && Tracer::params.pg_params.pg_enable)
        {
            //printf("A %f\n", Tracer::params.pg_params.guide_ratio);
            if(rnd(seed) < Tracer::params.pg_params.guide_ratio)
                return Tracer::params.pg_params.sample(seed, position);
        }

        //float3 N = normal;
        //float3 V = in_dir;
        //prd.origin = state.fhp;
        float r1 = rnd(seed);
        float r2 = rnd(seed);
        float r3 = rnd(seed);
        float r4 = rnd(seed);
        float3 dir;
        float3 normal = N;
        {
            Onb onb(normal);
            cosine_sample_hemisphere(r1, r2, dir);
            float r3 = rnd(seed);
            if (r3 < 0.5f)
                dir = -dir;
            onb.inverse_transform(dir);
            //return dir;
        }


        float mateta = mat.eta;
        float eta = 1 / mateta;
        if (dot(normal, V) < 0)
        {
            eta = 1 / eta;
            normal = -N;
        }

        if (false && mat.trans > .9)
        {
            float3 half = Shift::sample_half(mat.roughness * 3, normal, make_float2(r1, r2));
            if (dot(V, half) < 0)
            {
                half = -half;
            }
            float reflect_rate = .5;

            float cos_i = abs(dot(half, V));
            float sin_i = sqrt(1 - cos_i * cos_i);
            float sin_t2 = sin_i * eta * eta;
            if (sin_t2 > 1) reflect_rate = 1;
            if (rnd(seed) < reflect_rate)
            {
                return reflect(-V, half);
            }
            else
            {
                float3 out_dir;
                bool refract_good = refract(out_dir, V, half, 1 / eta);
                if (refract_good == false)
                {
                    printf("error refract in Sample\n");
                }
                return out_dir;
            }
        }

        float NdotV = abs(dot(normal, V));
        float transRatio = mat.trans;
        float transprob = rnd(seed);
        float refractRatio;
        float refractprob = rnd(seed);
        float probability = rnd(seed);
        float diffuseRatio = 0.5f * (1.0f - mat.metallic);// *(1 - transRatio);
        if (transprob < transRatio) // sample transmit
        {
            //refractRatio = 1 - fresnel(NdotV, sqrt(1 - sin2ThetaT), eta);
            refractRatio = 0.5;
            if (refractprob < refractRatio)
            {
                Onb onb(normal); // basis
                float a = mat.roughness;

                float phi = r3 * 2.0f * M_PIf;

                float cosTheta = sqrtf((1.0f - r4) / (1.0f + (a * a - 1.0f) * r4));
                float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
                float sinPhi = sinf(phi);
                float cosPhi = cosf(phi);
                float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
                onb.inverse_transform(half);

                if (dot(V, normal) == 0) return -V;

                float cosThetaI = abs(dot(half, V));
                float sin2ThetaI = 1 - cosThetaI * cosThetaI;
                float sin2ThetaT = eta * eta * sin2ThetaI;

                if (sin2ThetaT <= 1)
                {
                    float cosThetaT = sqrt(1 - sin2ThetaT);
                    //float y = -sqrt(1 - cosThetaI * cosThetaI * eta * eta)/sqrt(sin2ThetaI);
                    //float x = -(y + eta) * cosThetaI;
                    float3 L = normalize(eta * -V + (eta * cosThetaI - cosThetaT) * half);

                    //float3 L = x * half + y * V;
                    float HdotV = dot(half, V);
                    float HdotL = dot(half, L);
                    //if(eta * cosThetaI - cosThetaT > 0)
                    //	L = normalize(eta * -V - (eta * cosThetaI - cosThetaT) * half);
                    // printf("direct: %f, eta: %f, eval: %f\n", dot(V,N), eta, sqrt((1 - HdotL * HdotL)/(1 - HdotV * HdotV)));
                    return L;
                }
                else
                {
                    return half * 2 * dot(half, V) - V; // ȫ����
                }
            }
        }
        Onb onb(normal); // basis

        if (probability < diffuseRatio) // sample diffuse
        {
            cosine_sample_hemisphere(r1, r2, dir);
            onb.inverse_transform(dir);
        }
        else
        {
            float a = mat.roughness;

            float phi = r1 * 2.0f * M_PIf;

            float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
            float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            float3 half = make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
            onb.inverse_transform(half);

            dir = 2.0f * dot(V, half) * half - V; //reflection vector
        }
        return dir;
    }

    RT_FUNCTION float Pdf(MaterialData::Pbr& mat, float3 normal, float3 V, float3 L, float3 position = make_float3(0.0), bool use_pg = false)
    {

        //return abs(dot(L, normal)) * (.5f / M_PIf);

         
#ifdef BRDF
        if (mat.brdf)
            return 1.0f;// return abs(dot(L, normal));
#endif

        float transRatio = mat.trans;
        float3 n = normal;
        float mateta = mat.eta;
        //        float eta = dot(L, n) > 0 ? (mateta) : (1/mateta);     
        float eta = mateta;
        if (dot(n, V) < 0)
        {
            eta = 1 / eta;
            n = -normal;
        }
        float pdf = 0;
        float NdotV = abs(dot(V, n));
        float NdotL = abs(dot(L, n));
        float3 wh = -normalize(V + L * eta);

        float HdotV = abs(dot(V, wh));
        float HdotL = abs(dot(L, wh));

        float specularAlpha = mat.roughness;
        float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

        float diffuseRatio = 0.5f * (1.f - mat.metallic);// *(1 - transRatio);
        float specularRatio = 1.f - diffuseRatio;

        float3 half;
        half = normalize(L + V);

        float cosTheta = abs(dot(half, n));
        float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
        float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

        // calculate diffuse and specular pdfs and mix ratio
        float ratio = 1.0f / (1.0f + mat.clearcoat);
        float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
        float pdfDiff = abs(dot(L, n)) * (1.0f / M_PIf);

        //float refractRatio = 1 - fresnel(NdotV, sqrt(1 - sin2ThetaT), eta);
        float refractRatio = 0.5;
        pdf = (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * (1 - transRatio * refractRatio);// normal reflect

        float cosThetaI = abs(dot(wh, V));
        float sin2ThetaI = 1 - cosThetaI * cosThetaI;
        float sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

        if (sin2ThetaT <= 1)
        {
            // Compute change of variables _dwh\_dwi_ for microfacet transmission
            float sqrtDenom = eta * dot(L, wh) + dot(V, wh);
            float dwh_dwi =
                std::abs((eta * eta * dot(L, wh)) / (sqrtDenom * sqrtDenom));
            float a = max(0.001f, mat.roughness);
            float Ds = GTR2(abs(dot(wh, n)), a);
            float pdfTrans = Ds * abs(dot(n, wh)) * dwh_dwi;
            //printf("r pdf: %f\n", transRatio * pdfTrans * refractRatio);
            pdf += transRatio * pdfTrans * refractRatio;// refract
        }

        cosThetaI = abs(dot(half, V));
        sin2ThetaI = 1 - cosThetaI * cosThetaI;
        sin2ThetaT = 1 / (eta * eta) * sin2ThetaI;

        if (sin2ThetaT > 1)
        {
            //printf("l pdf: %f\n", (diffuseRatio * pdfDiff + specularRatio * pdfSpec));
            pdf += (diffuseRatio * pdfDiff + specularRatio * pdfSpec) * transRatio * refractRatio;// full reflect
        }

        /*
        if (mat.trans > 0 && isRefract(normal, V, L))
        {
            if (dot(normal, V) > 0)
            {
                pdf *= (mat.eta * mat.eta);
            }
            else
            {
                pdf /= (mat.eta * mat.eta);
            }
        }*/
         

        if (use_pg && Tracer::params.pg_params.pg_enable)
        { 
            pdf *= 1 - Tracer::params.pg_params.guide_ratio;
            pdf += Tracer::params.pg_params.guide_ratio * Tracer::params.pg_params.pdf(position, L);
        }
        return pdf;
    }

RT_FUNCTION float3 contriCompute(const BDPTVertex* path, int path_size)
{
    //要求：第0个顶点为eye，第size-1个顶点为light
    float3 throughput = make_float3(1);
    const BDPTVertex& light = path[path_size - 1];
    const BDPTVertex& lastMidPoint = path[path_size - 2];
    float3 lightLine = lastMidPoint.position - light.position;
    float3 lightDirection = normalize(lightLine);
    float lAng = dot(light.normal, lightDirection);
    if (lAng < 0.0f)
    {
        return make_float3(0.0f);
    }
    float3 Le = light.flux * lAng;
    throughput *= Le;
    for (int i = 1; i < path_size; i++)
    {
        const BDPTVertex& midPoint = path[i];
        const BDPTVertex& lastPoint = path[i - 1];
        float3 line = midPoint.position - lastPoint.position;
        throughput /= dot(line, line);
    }
    for (int i = 1; i < path_size - 1; i++)
    {
        const BDPTVertex& midPoint = path[i];
        const BDPTVertex& lastPoint = path[i - 1];
        const BDPTVertex& nextPoint = path[i + 1];
        float3 lastDirection = normalize(lastPoint.position - midPoint.position);
        float3 nextDirection = normalize(nextPoint.position - midPoint.position);

        MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
        mat.base_color = make_float4(midPoint.color, 1.0);
        throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection))
            * Eval(mat, midPoint.normal, lastDirection,nextDirection);
    }
    return throughput;
}
RT_FUNCTION float pdfCompute(const BDPTVertex* path, int path_size, int strategy_id)
{

    int eyePathLength = strategy_id;
    int lightPathLength = path_size - eyePathLength; 
    /*光源默认为面光源上一点，因此可以用cos来近似模拟其光照效果，如果是点光源需要修改以下代码*/

    float pdf = 1.0;
    if (lightPathLength > 0)
    {
        const BDPTVertex& light = path[path_size - 1];
        pdf *= light.pdf;
    }
    if (lightPathLength > 1)
    {
        const BDPTVertex& light = path[path_size - 1];
        const BDPTVertex& lastMidPoint = path[path_size - 2];
        float3 lightLine = lastMidPoint.position - light.position;
        float3 lightDirection = normalize(lightLine);
        pdf *= abs(dot(lightDirection, light.normal)) / M_PI;

        /*因距离和倾角导致的pdf*/
        for (int i = 1; i < lightPathLength; i++)
        {
            const BDPTVertex& midPoint = path[path_size - i - 1];
            const BDPTVertex& lastPoint = path[path_size - i];
            float3 line = midPoint.position - lastPoint.position;
            float3 lineDirection = normalize(line);
            pdf *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }

        for (int i = 1; i < lightPathLength - 1; i++)
        {
            const BDPTVertex& midPoint = path[path_size - i - 1];
            const BDPTVertex& lastPoint = path[path_size - i];
            const BDPTVertex& nextPoint = path[path_size - i - 2];
            float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            float3 nextDirection = normalize(nextPoint.position - midPoint.position);

            MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
            mat.base_color = make_float4(midPoint.color, 1.0);
            float rr_rate = fmaxf(midPoint.color);
            pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
        }

    }
    /*由于投影角导致的pdf变化*/
    for (int i = 1; i < eyePathLength; i++)
    {
        const BDPTVertex& midPoint = path[i];
        const BDPTVertex& lastPoint = path[i - 1];
        float3 line = midPoint.position - lastPoint.position;
        float3 lineDirection = normalize(line);
        pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
    }
    /*采样方向的概率*/
    for (int i = 1; i < eyePathLength - 1; i++)
    {
        const BDPTVertex& midPoint = path[i];
        const BDPTVertex& lastPoint = path[i - 1];
        const BDPTVertex& nextPoint = path[i + 1];
        float3 lastDirection = normalize(lastPoint.position - midPoint.position);
        float3 nextDirection = normalize(nextPoint.position - midPoint.position);

        MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
        mat.base_color = make_float4(midPoint.color, 1.0);
        float rr_rate = fmaxf(midPoint.color);
        pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
    }
    return pdf;
}

    RT_FUNCTION float MISWeight_SPCBPT(const BDPTVertex* path, int path_size, int strategy_id)
    {
        if (strategy_id <= 1 || strategy_id == path_size)
        {
            return pdfCompute(path, path_size, strategy_id);
        }
        int eyePathLength = strategy_id;
        int lightPathLength = path_size - eyePathLength;
        float pdf = 1.0;

        for (int i = 1; i < eyePathLength; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            float3 line = midPoint.position - lastPoint.position;
            float3 lineDirection = normalize(line);
            pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        /*采样方向的概率*/
        for (int i = 1; i < eyePathLength - 1; i++)
        {
            const BDPTVertex& midPoint = path[i];
            const BDPTVertex& lastPoint = path[i - 1];
            const BDPTVertex& nextPoint = path[i + 1];
            float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            float3 nextDirection = normalize(nextPoint.position - midPoint.position);

            MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
            mat.base_color = make_float4(midPoint.color, 1.0);
            float rr_rate = fmaxf(midPoint.color);
            pdf *= Tracer::Pdf(mat, midPoint.normal, lastDirection, nextDirection, midPoint.position) * rr_rate;
        }

        float3 light_contri = make_float3(1.0);

        if (lightPathLength > 0)
        {
            const BDPTVertex& light = path[path_size - 1];
            light_contri *= light.flux;
        }

        if (lightPathLength > 1)
        {
            const BDPTVertex& light = path[path_size - 1];
            const BDPTVertex& lastMidPoint = path[path_size - 2];

            /*因距离和倾角导致的pdf*/
            for (int i = 1; i < lightPathLength; i++)
            {
                const BDPTVertex& midPoint = path[path_size - i - 1];
                const BDPTVertex& lastPoint = path[path_size - i];
                float3 line = midPoint.position - lastPoint.position;
                float3 lineDirection = normalize(line);
                light_contri *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection)) * abs(dot(lastMidPoint.normal, lineDirection));
            }

            for (int i = 1; i < lightPathLength - 1; i++)
            {
                const BDPTVertex& midPoint = path[path_size - i - 1];
                const BDPTVertex& lastPoint = path[path_size - i];
                const BDPTVertex& nextPoint = path[path_size - i - 2];
                float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                float3 nextDirection = normalize(nextPoint.position - midPoint.position);

                MaterialData::Pbr mat = Tracer::params.materials[midPoint.materialId];
                mat.base_color = make_float4(midPoint.color, 1.0);
                light_contri *= Tracer::Eval(mat, midPoint.normal, lastDirection, nextDirection);
            }

        }

        float3 position;
        float3 dir;
        float3 normal;
        int eye_subspace_id = 0;
        int light_subspace_id = 0;
        position = path[strategy_id - 1].position;
        normal = path[strategy_id - 1].normal;
        dir = normalize(path[strategy_id - 2].position - path[strategy_id - 1].position);
        labelUnit lu(position, normal, dir, false);
        eye_subspace_id = lu.getLabel();

        if (strategy_id == path_size - 1)
        {
            light_subspace_id = path[strategy_id].subspaceId;
        }
        else
        {
            position = path[strategy_id].position;
            normal = path[strategy_id].normal;
            dir = normalize(path[strategy_id + 1].position - path[strategy_id].position);
            labelUnit lu(position, normal, dir, true);
            light_subspace_id = lu.getLabel();
        }
        return pdf * float3weight(connectRate_SOL(eye_subspace_id, light_subspace_id, light_contri));
    }
    RT_FUNCTION float rrRate(float3 color)
    {
        float rr_rate = fmaxf(color);

#ifdef RR_MIN_LIMIT
        rr_rate = rr_rate < MIN_RR_RATE ? MIN_RR_RATE : rr_rate;
#endif
       // if (rr_rate > 0.5)return 0.5;
        return rr_rate;
    }
    RT_FUNCTION float rrRate(const MaterialData::Pbr& pbr)
    {
        return rrRate(make_float3(pbr.base_color));
    }
} // namespace Tracer
RT_FUNCTION int labelUnit::getLabel()
{
    if (light_side)
    {
        if (Tracer::params.subspace_info.light_tree)
            return classTree::getLabel(Tracer::params.subspace_info.light_tree, position, normal, dir);
    }
    else
    {
        if (Tracer::params.subspace_info.eye_tree)
            return classTree::getLabel(Tracer::params.subspace_info.eye_tree, position, normal, dir);
    }
    return 0;

}

namespace TrainData
{

    struct nVertex_device :nVertex
    {
        RT_FUNCTION nVertex_device(const nVertex& a, const nVertex_device& b, bool eye_side)
        {
            position = a.position;
            dir = normalize(b.position - a.position);
            normal = a.normal;
            weight = eye_side ? b.forward_eye(a) : b.forward_light(a);
            pdf = eye_side ? weight.x : b.forward_light_pdf(a);
            color = a.color;
            materialId = a.materialId;
            valid = true;
            label_id = a.label_id;
            isBrdf = a.isBrdf;

            depth = b.depth + 1;
            //save_t = 0;

            //brdf_tracing_branch(a, b, eye_side);

        }
        RT_FUNCTION void brdf_tracing_branch(const nVertex& a, const nVertex_device& b, bool eye_side)
        {
            if (b.isBrdf == false && a.isBrdf == true)
            {
                save_t = length(a.position - b.position);
            }
            else if (b.isBrdf == true)
            {
                save_t = b.save_t + length(a.position - b.position);
            }
            if (a.isBrdf == true && b.isBrdf == false)
            {
                if (eye_side == false)
                {
                    weight = b.weight * abs(dot(dir, b.normal)) * (b.isLightSource() ? make_float3(1.0) : b.color);
                }
                else
                {
                    weight = b.weight * b.color;
                }
            }
            else if (a.isBrdf == true && b.isBrdf == true)
            {
                weight = b.weight * b.color;
            }
            else if (a.isBrdf == false && b.isBrdf == true)
            {
                weight = b.weight * abs(dot(a.normal, dir)) * b.color;
                weight /= save_t * save_t;
                save_t = 0;
            }

        }
        RT_FUNCTION int get_label()
        {
            if (isLightSource())
            {
                return label_id;
            }
            return 0;//to be rewrote
        }
        RT_FUNCTION nVertex_device(const BDPTVertex& a, bool eye_side) :nVertex(a, eye_side) {}
        RT_FUNCTION nVertex_device() {}

        RT_FUNCTION float forward_light_pdf(const nVertex& b)const
        {
            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);

            if (isLightSource())
            {
                if (isAreaLight())
                {
                    g *= abs(dot(normal, c_dir));
                    return pdf * g * 1.0 / M_PI;
                }
                if (isDirLight())
                {
                    g = abs(dot(c_dir, b.normal));
                    return pdf * g * Tracer::params.sky.projectPdf();
                }
            }
            
            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color,1.0);
            float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, false);
            float RR_rate = Tracer::rrRate(mat);
            return pdf * d_pdf * RR_rate * g;
        }

        RT_FUNCTION float3 forward_eye(const nVertex& b)const
        {
            if (b.isDirLight())
            {
                float3 c_dir = -b.normal;

                MaterialData::Pbr mat = Tracer::params.materials[materialId];
                mat.base_color = make_float4(color, 1.0);
                float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, true);
                float RR_rate = Tracer::rrRate(color);
                return weight * d_pdf * RR_rate;
            }

            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);
             
            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color, 1.0);
            float d_pdf = Tracer::Pdf(mat, normal, dir, c_dir, position, true);
            float RR_rate = Tracer::rrRate(color);
            //d_pdf /= isBrdf ? abs(dot(normal, c_dir)) : 1;
            return weight * d_pdf * RR_rate * g;
        }

        RT_FUNCTION float3 forward_areaLight(const nVertex& b)const
        {
            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);
            float g = abs(dot(c_dir, b.normal)) * abs(dot(c_dir, normal)) / dot(vec, vec);

            return weight * g;
        }

        RT_FUNCTION float3 forward_dirLight(const nVertex& b)const
        {
            return weight * abs(dot(normal, b.normal));
        }

        RT_FUNCTION float3 forward_light_general(const nVertex& b)const
        {
            float3 vec = b.position - position;
            float3 c_dir = normalize(vec);

            float g = isBrdf ?
                abs(dot(c_dir, b.normal)) / dot(vec, vec) :
                abs(dot(c_dir, b.normal)) * abs(dot(c_dir, normal)) / dot(vec, vec);


            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color, 1.0);
            float3 d_contri = Tracer::Eval(mat, normal, dir, c_dir);
            return weight * g * d_contri;

        }
        RT_FUNCTION float3 forward_light(const nVertex& b)const
        { 
            if (isAreaLight())
                return forward_areaLight(b);
            if (isDirLight())
                return forward_dirLight(b);
            return forward_light_general(b);
        }
        RT_FUNCTION float3 local_contri(const nVertex_device& b) const
        {
            float3 vec = b.position - position;
            float3 c_dir = b.isDirLight() ? -b.normal : normalize(vec);
            MaterialData::Pbr mat = Tracer::params.materials[materialId];
            mat.base_color = make_float4(color, 1.0);
            return Tracer::Eval(mat, normal,c_dir, dir);
        }

    };

}

RT_FUNCTION void init_EyeSubpath(BDPTPath& p, float3 origin, float3 direction)
{
    p.push();
    p.currentVertex().position = origin;
    p.currentVertex().flux = make_float3(1.0);
    p.currentVertex().pdf = 1.0f;
    p.currentVertex().RMIS_pointer = 0;
    p.currentVertex().normal = direction;
    p.currentVertex().isOrigin = true;
    p.currentVertex().depth = 0;
    p.currentVertex().singlePdf = 1.0;

    p.nextVertex().singlePdf = 1.0f;
}

/* 把光采样信息装到一个bdpt顶点中 */
RT_FUNCTION void init_vertex_from_lightSample(Tracer::lightSample& light_sample, BDPTVertex& v)
{
    v.position = light_sample.position;
    v.normal = light_sample.normal();
    v.flux = light_sample.emission;
    v.pdf = light_sample.pdf;
    v.singlePdf = v.pdf;
    v.isOrigin = true;
    v.isBrdf = false;
    v.subspaceId = light_sample.subspaceId;
    v.depth = 0;
    v.materialId = light_sample.bindLight->id;
    v.RMIS_pointer = 1;
    v.uv = light_sample.uv;
    if (light_sample.bindLight->type == Light::Type::QUAD)
    {
        v.type = BDPTVertex::Type::QUAD;
    }
    else if (light_sample.bindLight->type == Light::Type::ENV)
    {
        v.type = BDPTVertex::Type::ENV;
    }
    //其他光源的状况待补充
}
namespace Tracer
{
    /* Latest Update! */
    RT_FUNCTION BDPTVertex FastTrace(BDPTVertex& a, float3 direction, bool& success)
    {
        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = direction;
        payload.origin = a.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);
        
        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath_simple(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            success = 0;
            return BDPTVertex();
        }
        success = 1;
        return payload.path.currentVertex();
    }
}
namespace Shift
{
    RT_FUNCTION float dh_dwi_estimate(float a, float3 normal, float3 half, float3 v)
    {
        float3 wo = 2.0f * dot(v, half) * half - v;
        float2 origin_uv = sample_reverse_half(a,  half);
        float scale = .0001;
        float2 uv1 = origin_uv + make_float2(scale, 0);
        float2 uv2 = origin_uv + make_float2(0, scale);
        float3 half2 = sample_half(a,  uv2);
        float3 half1 = sample_half(a,  uv1);
        float3 v1 = half1;//2.0 * dot(wo, half1) * half1 - wo;
        float3 v2 = half2;//2.0 * dot(wo, half2) * half2 - wo;
        float3 vv = half;//2.0 * dot(wo, half) * half - wo;
        float area = length(cross(v1 - vv, v2 - vv));//dh duv
        area *= (4 * dot(v, half));
        //printf("dir compare %f %f\n", float3weight(v),float3weight(vv));

        v1 = 2.0 * dot(wo, half1) * half1 - wo;
        v2 = 2.0 * dot(wo, half2) * half2 - wo;
        vv = 2.0 * dot(wo, half) * half - wo;
        float area2 = length(cross(v1 - vv, v2 - vv));//dw duv
        if (origin_uv.y < .7)
        {
            //printf("dh dwi compare %f %f %f\n", area / scale / scale, area2 / scale / scale, area / area2);
        }
        return area2 / scale / scale;
        //float3 reflect = 2.0f * dot(in_dir, normal) * normal - in_dir;

    }
    RT_FUNCTION float robePdf(float a, float3 normal, float3 half, float3 V)
    {
        //return dh_dwi_estimate(a, normal, half, V);
        float cosTheta = abs(dot(half, normal));
        float pdfGTR2 = Tracer::GTR2(cosTheta, a) * cosTheta;
        return pdfGTR2 / (4.0 * abs(dot(half, V)));
    }
    RT_FUNCTION void pdf_valid_compare(float a, float b, float3 normal, float3 halfa, float3 halfb, float3 v1 ,float3 v2)
    {
        float cosThetaa = abs(dot(halfa, normal));
        float cosThetab = abs(dot(halfb, normal));
        float pdfGTR2a = Tracer::GTR2(cosThetaa, a) ;
        float pdfGTR2b = Tracer::GTR2(cosThetab, b) ;
        float o1 = robePdf(a, normal, halfa, v1);// *(4.0 * abs(dot(halfa, v1)));
        float o2 = robePdf(b, normal, halfb, v2);// *(4.0 * abs(dot(halfb, v2)));
        float t1 = dh_dwi_estimate(a, normal, halfa, v1);
        float t2 = dh_dwi_estimate(b, normal, halfb, v2);
        float2 uv1 = sample_reverse_half(a,  halfa);
        float2 uv2 = sample_reverse_half(b,  halfb);
        if(uv2.y<.7 &&uv1.y<.7)
            printf("pdf compare %f %f %f %f %f %f uv info %f %f %f %f\n", o1, o2, t1, t2, o1 / o2, t1 / t2, uv1.x, uv2.y, uv2.x, uv2.y );

    }

    RT_FUNCTION bool glossy(const MaterialData::Pbr& mat)
    {
        return mat.roughness < 0.4 && max(mat.metallic, mat.trans) >= 0.99;
    }
    RT_FUNCTION bool glossy(const BDPTVertex& v)
    {
        return v.type == BDPTVertex::Type::NORMALHIT && glossy(Tracer::params.materials[v.materialId]);
    }
    RT_FUNCTION bool RefractionCase(const MaterialData::Pbr& mat)
    {
        return mat.trans > 0.9;
    }

    RT_FUNCTION bool IsCausticPath(const BDPTVertex* path, int path_size)
    {
        for (int i = 1; i < path_size - 1; i++)
        {
            if (Shift::glossy(path[i]) == false && Shift::glossy(path[i + 1]) == false)
                break;

            if (Shift::glossy(path[i]) == false && Shift::glossy(path[i + 1]) == true)
                return true;
        }
        return false;
    }
    RT_FUNCTION bool RefractionCase(const BDPTVertex& v)
    {
        return v.type == BDPTVertex::Type::NORMALHIT && RefractionCase(Tracer::params.materials[v.materialId]);
    }
     
    RT_FUNCTION float3 SampleControlled(float& duv_dwi, const MaterialData::Pbr& mat, const float3& N, const float3& V, float2 uv, bool is_refract, bool& sample_good)
    {
        sample_good = true;
        float a = max(mat.roughness, ROUGHNESS_A_LIMIT);
        float3 half = sample_half(a, N, uv);
        float cosTheta = abs(dot(half, N));
        float duv_dwh = Tracer::GTR2(cosTheta, a) * cosTheta;
        float3 out_dir;
        if (is_refract)
        {
            sample_good = refract(out_dir, V, half, dot(V, N) > 0 ? mat.eta : 1 / mat.eta);
        }
        else
        {
            out_dir = reflect(-V, half);
        }
        float dwi_dwh = is_refract ? 1 / dwh_dwi_refract(half, V, out_dir, mat.eta) : 4 * abs(dot(half, V));
        duv_dwi = duv_dwh / dwi_dwh;
        return out_dir;
    }
    RT_FUNCTION bool back_trace(const BDPTVertex& midVertex, float2 uv, float3 anchor, BDPTVertex& new_vertex, float& pdf, bool is_refract = false)
    {
        if (is_refract == true && RefractionCase(midVertex))
        {
         //   printf("I haven't implemented the refraction case of the uv remapping. So you call the uv remapping in undesigned cases\n");
        }
        
        float3 in_dir = normalize(anchor - midVertex.position);
        float3 normal = midVertex.normal;
        MaterialData::Pbr mat = Tracer::params.materials[midVertex.materialId];
        mat.base_color = make_float4(midVertex.color, 1.0);
         

        bool refract_good;
        float3 new_dir = SampleControlled(pdf, mat, normal, in_dir, uv, is_refract, refract_good);
        if (refract_good == false)return false;
  
        //float cos_A = dot(normal, in_dir);
        //float cos_B = dot(normal, new_dir);
        //float sin_A = sqrt(1 - cos_A * cos_A);
        //float sin_B = sqrt(1 - cos_B * cos_B);
        //if (is_refract == true)
        //    printf("check info %f %f %f %d %f %f\n %f %f %f-%f %f %f\n\n", sin_A, sin_B, sin_A / sin_B, dot(normal, in_dir) > 0, pdf, cos_A, 
        //        normal.x, normal.y, normal.z, in_dir.x, in_dir.y, in_dir.z);



        Tracer::PayloadBDPTVertex payload;
        payload.clear();
        payload.seed = 0;
        payload.ray_direction = new_dir;
        payload.origin = midVertex.position;
        init_EyeSubpath(payload.path, payload.origin, payload.ray_direction);


        float3 ray_direction = payload.ray_direction;
        float3 ray_origin = payload.origin;
        int begin_depth = payload.path.size;
        Tracer::traceEyeSubPath_simple(Tracer::params.handle, ray_origin, ray_direction,
            SCENE_EPSILON,  // tmin
            1e16f,  // tmax
            &payload);
        if (payload.path.size == begin_depth)
        {
            return false;
        }
        new_vertex = payload.path.currentVertex();
        float3 dirVec = new_vertex.position - midVertex.position;
        pdf *= 1/ dot(dirVec, dirVec) * abs(dot(new_dir, new_vertex.normal));
        return true;
    }

    RT_FUNCTION bool shift_type_compare(const BDPTVertex& a, const BDPTVertex& b)
    {
        if(a.type == BDPTVertex::Type::NORMALHIT && b.type == BDPTVertex::Type::NORMALHIT)
            return glossy(a) == glossy(b);
        else if (a.type != BDPTVertex::Type::NORMALHIT && b.type != BDPTVertex::Type::NORMALHIT)
        {
            return a.materialId == b.materialId;
        }
        return false;
        if (a.depth == 0)
        {
            return b.hit_lightSource() && a.materialId == b.materialId;
            //            if(a.type == BDPTVertex::Type::ENV && b.type == BDPTVertex::Type::ENV_MISS)return true;
            //            if(a.type == BDPTVertex::Type::QUAD && b.type == BDPTVertex::Type::HIT_LIGHTSOURCE) return true;
        }
        else if (b.hit_lightSource())
        {
            return false;
        }

        return glossy(a) == glossy(b);
    }

    //简单的路径偏移实现
    //首先统计需要偏移的顶点数目，这个数目等同于从末端往前数的光滑顶点数目
    //计算原先的路径pdf
    //根据偏移顶点数目从末端开始重新追踪
    //重映射时要在最后注意可见性
    struct PathContainer
    {
        BDPTVertex* v;
        int it_step;// +1 or -1
        int m_size;
        RT_FUNCTION PathContainer(BDPTVertex* _v, int _step)
        {
            v = _v;
            it_step = _step;
        }
        RT_FUNCTION PathContainer(BDPTVertex* _v, int _step, int size)
        {
            v = _v;
            it_step = _step;
            m_size = size;
        }
        RT_FUNCTION int size() { return m_size; }
        RT_FUNCTION int setSize(int i) { m_size = i; }
        RT_FUNCTION BDPTVertex& get(int i)
        {
#define SHIFT_VALID_SIZE 6
            if (i < 0)
            {
                i = m_size + i;
            }

            if (i >= m_size || i < 0)
            {
                printf("wrong Path Container index\n");
                return *v;
            }
            if (i >= SHIFT_VALID_SIZE)
            {
                printf("wrong path container index - invalid address\n");
            }
            return *(v + i * it_step);
        }
    };

    RT_FUNCTION float3 etaCheck(float3 in_dir, float3 ref_dir, float3 normal)
    {
        float cosA = abs(dot(ref_dir, normal));
        float cosB = abs(dot(in_dir, normal));
        float sin_A = sqrt(1 - cosA * cosA);
        float sin_B = sqrt(1 - cosB * cosB);
        return make_float3(sin_A, sin_B, sin_A / sin_B);
        //printf("refract test %f %f %f\n", sin_A, sin_B, sin_A / sin_B);
    }
    RT_FUNCTION float3 etaCheck(PathContainer& a, float3 anchor)
    {
        float3 out_dir = normalize(a.get(1).position - a.get(0).position);
        float3 in_dir = normalize(anchor - a.get(0).position);
        
        return etaCheck(in_dir, out_dir, a.get(0).normal);
    }

    RT_FUNCTION float map_function(float x, float ratio, float& pdf)
    {
        pdf = 1 / ratio / ratio;
        return x * ratio;
    }
    RT_FUNCTION void refract_state_fill(bool* refract_state, PathContainer& path,float3 anchor, int g)
    { 
        for (int i = 0; i < g; i++)
        {
            float3 in_dir = (i == 0 ? anchor : path.get(i - 1).position) - path.get(i).position;
            float3 out_dir = path.get(i + 1).position - path.get(i).position;
            refract_state[i] = isRefract(path.get(i).normal, in_dir, out_dir);
        }
    }



    RT_FUNCTION float2 scale_map(float2 center, float2 origin, float& jacob)
    {
        float r = .1f;
        jacob = r * r * 4;
        if (center.x < r)center.x = r;
        if (center.y < r)center.y = r;
        if (center.x > 1 - r)center.x = 1 - r;
        if (center.y > 1 - r)center.y = 1 - r;
        if (abs(origin.x - center.x) > r || abs(origin.y - center.y) > r)
        {
            jacob = -1;
            return origin;
        }
        float2 anchor = make_float2(center.x - r, center.y - r);
        return (origin - anchor) / (2 * r);
    }


    struct fractChannelsFlag
    {
        bool xyz[3];
        RT_FUNCTION fractChannelsFlag()
        {
            xyz[0] = false;
            xyz[1] = false;
            xyz[2] = false;
        }
        RT_FUNCTION bool needCheck(float3 f, float rr)
        {
            if (xyz[0] == false && rr < f.x) return true;
            if (xyz[1] == false && rr < f.y) return true;
            if (xyz[2] == false && rr < f.z) return true;
            return false;
        }
        RT_FUNCTION void update(float3 f, float rr)
        {
            if (rr < f.x) xyz[0] = true;
            if (rr < f.y) xyz[1] = true;
            if (rr < f.z) xyz[2] = true;
        }

        RT_FUNCTION bool finish()
        {
            if (xyz[0] == false) return false;
            if (xyz[1] == false) return false;
            if (xyz[2] == false) return false;
            return false;
        }

        RT_FUNCTION float3 valid_channel()
        {
            float3 ans = make_float3(0.0);
            if (xyz[0] == false) ans.x = 1;
            if (xyz[1] == false) ans.y = 1;
            if (xyz[2] == false) ans.z = 1;
            return ans;
        }
    };
    RT_FUNCTION void info_print(PathContainer& path, float3 anchor, float3 dir)
    {  
        return;
        if (path.size() == 3)
        {
            printf("\n\nbegin \n%f %f %f to\n%f %f %f to\n%f %f %f to\n%f %f %f\n\n%f %f %f normal\n\n",
                anchor.x, -anchor.z, anchor.y,
                path.get(0).position.x, -path.get(0).position.z, path.get(0).position.y,
                path.get(1).position.x, -path.get(1).position.z, path.get(1).position.y,
                dir.x, -dir.z, dir.y,
                path.get(0).normal.x, -path.get(0).normal.z, path.get(0).normal.y
                );
        }
    }

    RT_FUNCTION float3 BSDF(const BDPTVertex& LastVertex, const BDPTVertex& MidVertex, const BDPTVertex& NextVertex)
    {
        MaterialData::Pbr mat = MidVertex.getMat(Tracer::params.materials);
        float3 dir_A = LastVertex.position - MidVertex.position;
        float3 dir_B = NextVertex.position - MidVertex.position;
        float3 normal = MidVertex.normal;
        return Tracer::Eval(mat, normal, dir_A, dir_B);
    }
    RT_FUNCTION float GeometryTerm(const BDPTVertex& a, const BDPTVertex& b)
    {
        if (a.type == BDPTVertex::Type::ENV || b.type == BDPTVertex::Type::ENV || a.type == BDPTVertex::Type::ENV_MISS || b.type == BDPTVertex::Type::ENV_MISS)
        {
            printf("Geometry Term call in Env light but we haven't implement it");
        }
        float3 diff = a.position - b.position;
        float3 dir = normalize(diff);
        return abs(dot(dir, a.normal) * dot(dir, b.normal)) / dot(diff, diff);
    }
    RT_FUNCTION float tracingPdf(const BDPTVertex& a, const BDPTVertex& b)
    {
        if (a.depth == 0&& a.type == BDPTVertex::Type::QUAD)// a为面光源，半球面采样
        {
            return GeometryTerm(a, b) * 1 / M_PI * Tracer::visibilityTest(Tracer::params.handle, a, b);
        }
        else
        {
            //TBD 
            return GeometryTerm(a, b) * 1 / M_PI * Tracer::visibilityTest(Tracer::params.handle, a, b);
        }
    }
    struct getClosestPointFunction
    {
        float3 a;
        float3 b;
        float3 c; 
        float3 p;
        RT_FUNCTION getClosestPointFunction(float3 a, float3 b, float3 c, float3 p) :a(a), b(b), c(c), p(p) {}
        RT_FUNCTION float3 getSegDis(float3 aa, float3 bb, float3 pp, float3 normal_plane)
        {
            float3 ap = aa - pp;
            float3 normal = normalize(cross(normal_plane, aa - bb));
            float dis = dot(ap, normal);
            float3 projectPoint = pp + dis * normal;

            float3 aapp = projectPoint - aa;
            float3 bbpp = projectPoint - bb;
            if (dot(aapp, bbpp) < 0)return projectPoint;
            if (dot(aapp, aapp) < dot(bbpp, bbpp))return aa;
            return bb;
        }
        RT_FUNCTION float3 operator()()
        {
            float3 normal = normalize(cross(b - a, c - a));
            float3 ap = a - p;
            
            float dis = dot(ap, normal);
            float3 projectPoint = p + dis * normal;

            float3 aP = a - projectPoint;
            float3 bP = b - projectPoint;
            float3 cP = c - projectPoint;

            float3 ab = a - b;
            float3 bc = b - c;
            float3 ca = c - a;
            if (dot(cross(aP, -ab), cross(bP, -bc)) > 0 && dot(cross(bP, -bc), cross(cP, -ca)) > 0) return projectPoint;

            float3 Ca = getSegDis(a, b, projectPoint, normal);
            float3 Cb = getSegDis(b, c, projectPoint, normal);
            float3 Cc = getSegDis(c, a, projectPoint, normal);

            float3 ans = Ca;
            if (length(Ca - p) > length(Cb - p))ans = Cb;
            if (length(Cb - p) > length(Cc - p))ans = Cc;
            return ans;
        } 

    };

    RT_FUNCTION float getClosestGeometry_upperBound(Light &light, float3 position,float3 normal,float3& sP)
    {
        Tracer::lightSample light_sample;
        float3 ca = getClosestPointFunction(light.quad.corner, light.quad.u, light.quad.v,position)();
        float3 cb = getClosestPointFunction(-light.quad.corner + light.quad.u + light.quad.v, light.quad.u, light.quad.v, position)();
        float3 c = length(ca - position) < length(cb - position) ? ca: cb;
        float3 diff = c - position;
        float3 dir = normalize(diff);
        sP = c;

        float cos_bound = abs(dot(normal,dir));
        if (abs(dot(normalize(light.quad.corner - position), normal)) > cos_bound)cos_bound = abs(dot(normalize(light.quad.corner - position), normal));
        if (abs(dot(normalize(light.quad.u - position), normal)) > cos_bound)cos_bound = abs(dot(normalize(light.quad.u - position), normal));
        if (abs(dot(normalize(light.quad.v - position), normal)) > cos_bound)cos_bound = abs(dot(normalize(light.quad.v - position), normal));
        if (abs(dot(normalize(-light.quad.corner + light.quad.u + light.quad.v - position), normal)) > cos_bound)
            cos_bound = abs(dot(normalize(-light.quad.corner + light.quad.u + light.quad.v - position), normal));


        return abs(dot(dir, light.quad.normal) ) / dot(diff, diff) * 1 / M_PI * cos_bound;

    }

    /*use to genertate low_difference seq*/
    RT_FUNCTION double halton(int index, int base) {
        double result = 0;
        double f = 1.0 / base;
        int i = index;

        while (i > 0) {
            result += f * (i % base);
            i /= base;
            f /= base;
        }

        return result;
    }

    /* calculate the absent path pdf */
    RT_FUNCTION float another_inverPdfEstimate(PathContainer& path, unsigned& seed)
    {
        /* now we only work with path as light-glossy */
        if (path.size() != 2 || glossy(path.get(0)) == false)
        {
            return 0;
        }

        Light light = Tracer::params.lights[path.get(1).materialId];
        //        light_sample.ReverseSample(light, path.get(1).uv);
        //        float pdf_ref = light_sample.pdf * tracingPdf(path.get(1),path.get(0));
        float3 sP;
        float ans = 0;

        float average_accumulate = 0;
        int suc_int = 0;
        float ratio = 1.0;
        int count = 50;
        int base1 = 2;
        int base2 = 3;
        int bias = rnd(seed) * 1000;//randomly,the num don't change the method
        BDPTVertex np;
        BDPTVertex& v = path.get(0);
        /* 从光源采样 均匀采样*/
        for (int i = 0; i < count; i++)
        {
            suc_int++;
            float x_cord = halton(i + bias, base1);
            float y_cord = halton(i + bias, base2);
            float2 uv = make_float2(x_cord, y_cord);
            if (rnd(seed) > ratio)
            {
                /* 从glossy顶点采样 */
                /* 建立局部坐标系，onb代表orthonormal basis*/
                Onb onb(dot(v.normal, path.get(1).position - v.position) > 0 ? v.normal : -v.normal);
                float3 dir;
                /* 半球空间采样 */
                cosine_sample_hemisphere(uv.x, uv.y, dir);
                onb.inverse_transform(dir);
                /* 从glossy顶点出发进行追踪 */
                bool success_hit;
                np = Tracer::FastTrace(v, dir, success_hit);
                /* 这里直接continue是正确的 */
                if (success_hit == false || np.type != BDPTVertex::Type::HIT_LIGHT_SOURCE)
                    continue;
                Light light = Tracer::params.lights[np.materialId];
                Tracer::lightSample light_sample;
                light_sample.ReverseSample(light, np.uv);
                /* 把信息装到np中 */
                init_vertex_from_lightSample(light_sample, np);
            }
            else
            {
                /* 从光源采样 */
                Tracer::lightSample light_sample;
                light_sample.ReverseSample(light, uv);
                /* 把光源采样的信息装到np中 */
                init_vertex_from_lightSample(light_sample, np);
            }
            /* 计算f(x)/p(x) */
            float pdf = (np.pdf * tracingPdf(np, path.get(0))) /
                (np.pdf * ratio + tracingPdf(np, path.get(0)) * (1 - ratio));
            average_accumulate += pdf;
        }
        ans = suc_int / average_accumulate;
        
        return ans;
    }

    RT_FUNCTION float TEST_1_inverPdfEstimate_LS(PathContainer& path, unsigned& seed)
    {
        /* path 0是glossy 1是光 */
        Light light = Tracer::params.lights[path.get(1).materialId];
        float3 sP;
        /* 估计pdf上界 */
        float upperbound = getClosestGeometry_upperBound(
            light,
            path.get(0).position,
            path.get(0).normal,
            sP
        );

        float bound = upperbound / 4;
        float ans = 0;
        float average_accumulate = 0;
        float sample_ratio = 0.8;
        float rrs_ratio;

        int suc_int = 0;
        int bias = rnd(seed) * 1000;//randomly,the num don't change the method
        BDPTVertex& v = path.get(0);
        BDPTVertex np;
        /* pdf估计的核心流程 */
        for (int i = 0; i < 1; i++)
        {
            ans = 0;
            suc_int++;
            float factor = 1;
            int loop_cnt = 0;
            // 
            while (true)
            {
                loop_cnt += 1;
                if (loop_cnt > 1000) {
                    //printf("Break due to loop_cnt > 1000 \n");
                    break;
                }

                ans += factor / bound;

                float x_cord = halton(i + bias, 2);
                float y_cord = halton(i + bias, 3);
                /* 使用哪种方法来采样残缺顶点？*/
                if (rnd(seed) > sample_ratio)
                {
                    /* 从glossy顶点采样 */
                    /* 建立局部坐标系，onb代表orthonormal basis*/
                    Onb onb(dot(v.normal, path.get(1).position - v.position) > 0 ? v.normal : -v.normal);
                    float3 dir;
                    /* 半球空间采样 */
                    cosine_sample_hemisphere(x_cord, y_cord, dir);
                    onb.inverse_transform(dir);
                    /* 从glossy顶点出发进行追踪 */
                    bool success_hit;
                    np = Tracer::FastTrace(v, dir, success_hit);
                    /* 这里直接continue是正确的 */
                    if (success_hit == false || np.type != BDPTVertex::Type::HIT_LIGHT_SOURCE)
                        continue;
                    Light light = Tracer::params.lights[np.materialId];
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(light, np.uv);
                    /* 把信息装到np中 */
                    init_vertex_from_lightSample(light_sample, np);
                }
                else
                {
                    /* 从光源采样 */
                    float2 uv = make_float2(x_cord, y_cord);
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(light, uv);
                    /* 把光源采样的信息装到np中 */
                    init_vertex_from_lightSample(light_sample, np);
                }

                /* 计算f(x)/p(x) */
                float pdf = (np.pdf * tracingPdf(np, path.get(0))) /
                    (np.pdf * sample_ratio + tracingPdf(np, path.get(0)) * (1 - sample_ratio));

                float continue_rate = 1 - pdf / bound;
                if (abs(continue_rate) > 1)
                {
                    bound *= 2;
                    ans = 0;
                    suc_int -= 1;
                    break;
                };

                float rr_rate = (abs(continue_rate) > 1) ? 0.5 : abs(continue_rate);
                if (rnd(seed) > rr_rate)
                    break;
                factor *= continue_rate / rr_rate;
            }
            average_accumulate += ans;
        }
        ans = average_accumulate / suc_int;
        return ans;
    }

    RT_FUNCTION float inverPdfEstimate_LS(PathContainer& path, unsigned& seed)
    {
        /* path 0是glossy 1是光 */
        Light light = Tracer::params.lights[path.get(1).materialId];
        float3 sP;
        /* 估计pdf上界 */
        float upperbound = getClosestGeometry_upperBound(
            light,
            path.get(0).position,
            path.get(0).normal,
            sP
        );
        float pdf_ref_sum = tracingPdf(path.get(1), path.get(0));
        int pdf_ref_count = 1;
        float bound = pdf_ref_sum / pdf_ref_count * 2;
        bound = upperbound;

        float ans = 0;

        float variance_accumulate = 0;
        float average_accumulate = 0;
        int suc_int = 0;

        /* 使用老方法还是用纯RR？ */
        bool RR_option = 0;
        float RR_rate = 0.8;

        /* pdf估计的核心流程 */
        for (int i = 0; i < 50; i++)
        {
            ans = 0;
            suc_int++;
            float factor = 1;
            int loop_cnt = 0;
            // 
            while (true)
            {
                loop_cnt += 1;
                if (loop_cnt > 1000) {
                    //printf("Break due to loop_cnt > 1000 \n");
                    break;
                }
                ans += factor / bound;
                BDPTVertex& v = path.get(0);
                float ratio = 0.8;
                BDPTVertex np;
                /* 使用哪种方法来采样残缺顶点？*/
                if (rnd(seed) > ratio)
                {
                    /* 从glossy顶点采样 */
                    /* 建立局部坐标系，onb代表orthonormal basis*/
                    Onb onb((dot(v.normal, path.get(1).position - v.position) > 0) ? v.normal : -v.normal);
                    float3 dir;
                    /* 半球空间采样 */
                    cosine_sample_hemisphere(rnd(seed), rnd(seed), dir);
                    onb.inverse_transform(dir);
                    /* 从glossy顶点出发进行追踪 */
                    bool success_hit;
                    np = Tracer::FastTrace(v, dir, success_hit);
                    /* 这里直接continue是正确的 */
                    if (success_hit == false || np.type != BDPTVertex::Type::HIT_LIGHT_SOURCE)
                        continue;
                    Light light = Tracer::params.lights[np.materialId];
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(light, np.uv);
                    /* 把信息装到np中 */
                    init_vertex_from_lightSample(light_sample, np);
                }
                else
                {
                    /* 从光源采样 */
                    float2 uv = make_float2(rnd(seed), rnd(seed));
                    Tracer::lightSample light_sample;
                    light_sample.ReverseSample(light, uv);
                    /* 把光源采样的信息装到np中 */
                    init_vertex_from_lightSample(light_sample, np);
                }
                /* 计算f(x)/p(x) */
                float pdf = (np.pdf * tracingPdf(np, path.get(0))) /
                    (np.pdf * ratio + tracingPdf(np, path.get(0)) * (1 - ratio));
                //float pdf = tracingPdf(np, path.get(0));

                if (RR_option) {
                    /* 试一试纯RR效果如何 */
                    if (rnd(seed) > RR_rate)
                        break;
                    factor *= (1 - pdf / bound) / RR_rate;

                }
                else {
                    /* 老方法，sfj写的 */
                    float continue_rate = 1 - pdf / bound;
                    /* 测出来continue_rate都很接近于1 */
                    // printf("c_rate: %f\n", continue_rate);
                    /* 这一段在干嘛？ */
                    if (abs(continue_rate) > 1)
                    {
                        bound *= 2;
                        ans = 0;
                        suc_int -= 1;
                        break;
                    };

                    float rr_rate = (abs(continue_rate) > 1) ? 0.5 : abs(continue_rate);
                    if (rnd(seed) > rr_rate)
                        break;
                    factor *= continue_rate / rr_rate;
                }

            }  // end while
            //printf("loop_cnt: %d\n", loop_cnt);
            variance_accumulate += ans * ans;
            average_accumulate += ans;
            /* 提前退出了 */
            if (suc_int == 1)
                break;
        }
        ans = average_accumulate / suc_int;
        variance_accumulate /= suc_int;
        variance_accumulate -= ans * ans;
        return ans;
    }

    RT_FUNCTION float inverPdfEstimate_LDS(PathContainer& path, unsigned& seed)
    {        
        /* path 0是glossy 1是diffuse 2是光*/
        Light light = Tracer::params.lights[path.get(2).materialId]; 
        float3 sP;

        /* 估计pdf上界 */
        float bound = 10;//upperbound;
        /* glossy顶点 */
        BDPTVertex& v = path.get(0);
        /* 光顶点 */
        BDPTVertex& l = path.get(2);
        float ratio = 0.5;
        BDPTVertex np;
        const int simulate_num = 50;
        for (int i = 0; i < simulate_num; i++)
        {
            /* 使用哪种方法来采样残缺顶点？*/
            if (rnd(seed) >= ratio)
            {
                /* 从glossy顶点采样 */
                /* 建立局部坐标系，onb代表orthonormal basis*/
                Onb onb(rnd(seed) > 0.5 ? v.normal : -v.normal);
                float3 dir;
                /* 半球空间采样 */
                cosine_sample_hemisphere(rnd(seed), rnd(seed), dir);
                onb.inverse_transform(dir);
                /* 从glossy顶点出发进行追踪 */
                bool success_hit;
                /* 此处np为中间的diffuse顶点 */
                np = Tracer::FastTrace(v, dir, success_hit);
                /* 这里直接continue是正确的 */
                if (success_hit == false || np.type == BDPTVertex::Type::HIT_LIGHT_SOURCE ||
                    Shift::glossy(np))
                    continue;

            }
            else
            {
                /* 从光源采样 */
                /* 建立局部坐标系，onb代表orthonormal basis*/
                Onb onb(dot(l.normal, path.get(1).position - l.position) > 0 ? l.normal : -l.normal);
                float3 dir;
                /* 半球空间采样 */
                cosine_sample_hemisphere(rnd(seed), rnd(seed), dir);
                onb.inverse_transform(dir);
                /* 从glossy顶点出发进行追踪 */
                bool success_hit;
                /* 此处np为中间的diffuse顶点 */
                np = Tracer::FastTrace(l, dir, success_hit);
                /* 这里直接continue是正确的 */
                if (success_hit == false || np.type == BDPTVertex::Type::HIT_LIGHT_SOURCE ||
                    Shift::glossy(np))
                    continue;
            }
            /* 计算f(x)/p(x) */
            MaterialData::Pbr mat = Tracer::params.materials[np.materialId];
            float pdf = l.pdf * tracingPdf(l, np)
                * Tracer::Pdf(mat, np.normal, normalize(l.position - np.position), normalize(v.position - np.position))
                * GeometryTerm(np, v)
                * Tracer::visibilityTest(Tracer::params.handle, np, v)
                / (ratio * tracingPdf(v, np) * 0.5f + (1 - ratio) * tracingPdf(l, np));
            bound = max(1.5*pdf, bound);
        }

        float ans = 0;

        float variance_accumulate = 0;
        float average_accumulate = 0;
        int suc_int = 0;

        /* 使用老方法还是用纯RR？ */
        bool RR_option = 0;
        float RR_rate = 0.5;

        /* pdf估计的核心流程 */
        for (int i = 0; i < 50; i++)
        {
            ans = 0;
            suc_int++;
            float factor = 1;
            int loop_cnt = 0;
            // 
            while (true)
            {
                loop_cnt += 1;
                if (loop_cnt > 1000) {
                    //printf("Break due to loop_cnt > 1000 \n");
                    break;
                }
                ans += factor / bound;

                /* glossy顶点 */
                BDPTVertex& v = path.get(0);
                /* 光顶点 */
                BDPTVertex& l = path.get(2);
                float ratio = 0.5;
                BDPTVertex np;
                /* 使用哪种方法来采样残缺顶点？*/
                if (rnd(seed) >= ratio)
                {
                    /* 从glossy顶点采样 */
                    /* 建立局部坐标系，onb代表orthonormal basis*/
                    Onb onb(rnd(seed) > 0.5 ? v.normal : -v.normal);
                    float3 dir;
                    /* 半球空间采样 */
                    cosine_sample_hemisphere(rnd(seed), rnd(seed), dir);
                    onb.inverse_transform(dir);
                    /* 从glossy顶点出发进行追踪 */
                    bool success_hit;
                    /* 此处np为中间的diffuse顶点 */
                    np = Tracer::FastTrace(v, dir, success_hit);
                    /* 这里直接continue是正确的 */
                    if (success_hit == false || np.type == BDPTVertex::Type::HIT_LIGHT_SOURCE ||
                        Shift::glossy(np))
                        continue;
    
                }
                else
                {
                    /* 从光源采样 */
                    /* 建立局部坐标系，onb代表orthonormal basis*/
                    Onb onb(dot(v.normal, path.get(1).position - v.position) > 0 ? l.normal : -l.normal);
                    float3 dir;
                    /* 半球空间采样 */
                    cosine_sample_hemisphere(rnd(seed), rnd(seed), dir);
                    onb.inverse_transform(dir);
                    /* 从glossy顶点出发进行追踪 */
                    bool success_hit;
                    /* 此处np为中间的diffuse顶点 */
                    np = Tracer::FastTrace(l, dir, success_hit);
                    /* 这里直接continue是正确的 */
                    if (success_hit == false || np.type == BDPTVertex::Type::HIT_LIGHT_SOURCE ||
                        Shift::glossy(np))
                        continue;
                }
                /* 计算f(x)/p(x) */
                MaterialData::Pbr mat = Tracer::params.materials[np.materialId];
                float pdf = l.pdf * tracingPdf(l, np)
                    * Tracer::Pdf(mat, np.normal, normalize(l.position - np.position), normalize(v.position - np.position))* GeometryTerm(np, v)
                    * Tracer::visibilityTest(Tracer::params.handle, np, v)
                    / (ratio * tracingPdf(v,np)*0.5 + (1 - ratio) * tracingPdf(l,np));
                //float pdf = tracingPdf(np, path.get(0));

                if (RR_option) {
                    /* 试一试纯RR效果如何 */
                    if (rnd(seed) > RR_rate)
                        break;
                    factor *= (1 - pdf / bound) / RR_rate;

                }
                else {
                    /* 老方法，sfj写的 */
                    float continue_rate = 1 - pdf / bound;
                    /* 测出来continue_rate都很接近于1 */
                    // printf("c_rate: %f\n", continue_rate);
                    /* 这一段在干嘛？ */
                    if (abs(continue_rate) > 1)
                    {
                        bound *= 2;
                        ans = 0;
                        suc_int -= 1;
                        break;
                    };

                    float rr_rate = (abs(continue_rate) > 1) ? 0.5 : abs(continue_rate);
                    if (rnd(seed) > rr_rate)
                        break;
                    factor *= continue_rate / rr_rate;
                }

            }  // end while
            // printf("loop_cnt: %d\n", loop_cnt);
            variance_accumulate += ans * ans;
            average_accumulate += ans;
            /* 提前退出了 */
            if (suc_int == 1)
                break;
        }
        ans = average_accumulate / suc_int;
        variance_accumulate /= suc_int;
        variance_accumulate -= ans * ans;
        //printf("inverpdf %f\n", ans);
        return ans;
    }


    RT_FUNCTION float inverPdfEstimate_L_S_S(PathContainer& path, unsigned& seed)
    {
        /* path 0~d-1是glossy d是光*/
        short d = path.size() - 1;
        //if (d > 2) 
            //return 100;
        Light light = Tracer::params.lights[path.get(d).materialId];

        /* 估计pdf上界 */
        float bound = 1;

        float ans = 0;
        float variance_accumulate = 0;
        float average_accumulate = 0;
        int suc_int = 0;

        /* pdf估计的核心流程 */
        for (int i = 0; i < 50; i++)
        {
            ans = 0;
            suc_int++;
            float factor = 1;
            int loop_cnt = 0;
            // 
            while (true)
            {
                loop_cnt += 1;
                if (loop_cnt > 1000) {
                    // printf("Break due to loop_cnt > 1000 \n");
                    break;
                }
                ans += factor / bound;

                /* glossy顶点 */
                BDPTVertex& v = path.get(0);
                /* 光顶点 */
                BDPTVertex& l = path.get(d);
                BDPTVertex np0, np1,np2;
                float pdf = l.pdf;
                /* 从第一个 glossy 顶点采样，用半球空间 */
                {
                    /* 建立局部坐标系，onb代表orthonormal basis*/
                    Onb onb(dot(v.normal, path.get(1).position - v.position) > 0 ? v.normal : -v.normal);
                    float3 dir;
                    /* 半球空间采样 */
                    cosine_sample_hemisphere(rnd(seed), rnd(seed), dir);
                    onb.inverse_transform(dir);
                    /* 从glossy顶点出发进行追踪 */
                    bool success_hit;
                    /* 此处np为中间的glossy顶点 */
                    np1 = Tracer::FastTrace(v, dir, success_hit);
                    /* 这里直接continue是正确的 */
                    if (success_hit == false || np1.type == BDPTVertex::Type::HIT_LIGHT_SOURCE ||
                        !Shift::glossy(np1))
                        continue;
                    pdf /= tracingPdf(v, np1);
                }
                /* 要追 d-1 个 glossy 顶点和 1 个光顶点 */
                np0 = v;
                bool retrace_state = 1;
                for (int i = 1; i < d; ++i) {
                    float3 in_dir = np0.position - np1.position;
                    MaterialData::Pbr mat = Tracer::params.materials[np1.materialId];
                    float3 out_dir = Tracer::Sample(mat, np1.normal, in_dir, seed);

                    bool success_hit;
                    np2 = Tracer::FastTrace(v, out_dir, success_hit);

                    /* 到了最后一次，追光源顶点 */
                    if (i == d - 1)
                    {
                        if (success_hit == false || np2.type != BDPTVertex::Type::HIT_LIGHT_SOURCE)
                        {
                            retrace_state = 0;
                            break;
                        }
                    }
                    else 
                    /* 之前 d-1 次，追 glossy 顶点 */
                    {
                        if (success_hit == false || np2.type == BDPTVertex::Type::HIT_LIGHT_SOURCE ||
                            !Shift::glossy(np2))
                        {
                            retrace_state = 0;
                            break;
                        }
                    }
                    pdf *= GeometryTerm(np0, np1);
                    pdf *= Tracer::Pdf(mat, np1.normal, normalize(np2.position - np1.position), normalize(np0.position - np1.position));
                    np0 = np1;
                    np1 = np2;
                }
                if (!retrace_state) continue;
                pdf *= tracingPdf(np2, np1);
                
       
                float continue_rate = 1 - pdf / bound;

                if (abs(continue_rate) > 1)
                {
                    bound *= 2;
                    ans = 0;
                    suc_int -= 1;
                    break;
                }
                float rr_rate = (abs(continue_rate) > 1) ? 0.5 : abs(continue_rate);
                if (rnd(seed) > rr_rate)
                    break;
                factor *= continue_rate / rr_rate;

            }  // end while
            // printf("loop_cnt: %d\n", loop_cnt);
            variance_accumulate += ans * ans;
            average_accumulate += ans;

            if (suc_int == 1)
                break;
        }
        ans = average_accumulate / suc_int;
        variance_accumulate /= suc_int;
        variance_accumulate -= ans * ans;
        //printf("inverpdf %f\n", ans);
        return ans;
    }

    /* 计算残缺路径的pdf */
    RT_FUNCTION float inverPdfEstimate(PathContainer& path, unsigned &seed, BDPTVertex& curVertex)
    {

        /* L - S */
        if (curVertex.path_record == 0b1)
        {
            float x1= inverPdfEstimate_LS(path, seed);
            //float x1 = another_inverPdfEstimate(path, seed);
            //printf("old:%f new%f\n", x1, x2);
            return x1;
        }
        /* L - D - S */
        else if (curVertex.depth == 2 && curVertex.path_record == 0b10)
        {
            return inverPdfEstimate_LDS(path, seed);
        }
        /* L - (S)* - S */
        else if (curVertex.depth >1 && (curVertex.path_record == (1 << curVertex.depth) - 1))
        {
            return inverPdfEstimate_L_S_S(path, seed);
        }
        return 0;
    }


    //test code 
    //can't ensure its performance
    RT_FUNCTION float get_duv_dwi(const MaterialData::Pbr& mat, const float3& N, const float3& V, const float3& L, bool is_refract)
    {
        float3 half;
        if (is_refract)
        {
            half = refract_half(V,L,N,mat.eta);

//            return 0;
        }
        else
        {
            half = normalize(V + L); 
        }
        float cosTheta = abs(dot(N, half));
        float a = max(mat.roughness, ROUGHNESS_A_LIMIT);

        float duv_dwh = Tracer::GTR2(cosTheta, a) * cosTheta;
        float dwi_dwh = is_refract ? dwh_dwi_refract(half, V, L, mat.eta) : 4 * abs(dot(half, V));
        return duv_dwh / dwi_dwh;
    }

}

#endif // !CUPROG_H