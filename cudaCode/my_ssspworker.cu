#include <cuda_runtime.h>
#include <stdio.h>
#include "my_ssspworker.cuh"
// #include "freshman.h"
namespace tjnsssp{
    __device__ unsigned int *oeoffset_d;
    __device__ unsigned int *cur_oeoff_d;
    __device__ unsigned int *size_oe_d;
    __device__ int *oe_edata_d;
    __device__ int *deltas_d;
    __device__ int *values_d;
    __device__ unsigned int *cur_modified;
    __device__ unsigned int cur_modified_size;
    __device__ unsigned int *next_modified;
    __device__ unsigned int next_modified_size;
    
    void init(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d){
        init_real<<<1,1>>>(oeoffset_d, oe_edata_d, cur_oeoff_d, deltas_d, values_d, size_oe_d);
    }

    __global__
    void init_real(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d){
        tjnsssp::oeoffset_d = oeoffset_d;
        tjnsssp::cur_oeoff_d = cur_oeoff_d;
        tjnsssp::size_oe_d = size_oe_d;
        tjnsssp::oe_edata_d = oe_edata_d;
        tjnsssp::deltas_d = deltas_d;
        tjnsssp::values_d = values_d;
    }

    void g_function(){
        dim3 block(512);
        dim3 grid((tjnsssp::cur_modified_size - 1) / block.x + 1);
        g_function_real<<<grid, block>>>();
        cudaDeviceSynchronize();
        swap();
        // 
    }

    __global__
    void g_function_real(){

    }

    void swap(){

    }

    __global__
    void swap_real(){

    }
}