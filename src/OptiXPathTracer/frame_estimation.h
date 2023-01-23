#ifndef FRAME_ESTIMATION
#define FRAME_ESTIMATION

#include<vector>
#include<thrust/host_vector.h>
#include<string>
#include<iostream>
#include<fstream> 

#include"optixPathTracer.h"
namespace estimation
{ 
    struct estimation_status
    {
        thrust::host_vector<float4> reference;
        int ref_width;
        int ref_height;
        bool estimation_mode;
        estimation_status(std::string reference_filepath, bool old_version = false);
        float relMse_estimate(thrust::host_vector<float4> accm, const MyParams& params);
        float Mae_estimate(thrust::host_vector<float4> accm, const MyParams& params);
        float Mape_estimate(thrust::host_vector<float4> accm, const MyParams& params);
    };
    extern estimation_status es;

    struct estimation_info
    {
        std::ofstream outFile;
        std::string scene;
        std::string algorithm;
        int N_conn;
        int N_subspace;
        int M_lightvertex;
        int M_lightpath;
        int isIndirectOnly;
        int useClassifier;
        float subspaceDivTime;
        int t0_strategy;
    
        estimation_info();
        void add(int frameId,float time,float Mse,float Mae,float Mape);
        void close();
    };
    extern estimation_info es_info;
}

#endif // !FRAME_ESTIMATION
