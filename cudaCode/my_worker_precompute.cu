#include <cuda_runtime.h>
#include <stdio.h>
#include "my_worker_precompute.cuh"
namespace tjnpr_precompute{
    __device__ float *values_d;
    __device__ float *deltas_d;
    __device__ unsigned int *size_oes_d;
    __device__ unsigned int *size_subgraph_d;
    __device__ unsigned int *cur_subgraph_d;
    __device__ unsigned int *subgraph_neighbor_d;
    __device__ unsigned int num;
    __device__ bool *is_active_d;
    __device__ float diff;
    void init_subgraph(float *deltas_d, float *values_d ,unsigned int *size_oes_d, unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, unsigned int *subgraph_neighbor_d, unsigned int num, bool *is_active_d){
        init_subgraph_real<<<1, 1>>>(deltas_d, values_d, size_oes_d, size_subgraph_d, cur_subgraph_d, subgraph_neighbor_d, num, is_active_d);
    }
    __global__
    void init_subgraph_real(float *deltas_d, float *values_d ,unsigned int *size_oes_d, unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, unsigned int *subgraph_neighbor_d, unsigned int num, bool *is_active_d){
        tjnpr_precompute::deltas_d = deltas_d;
        tjnpr_precompute::values_d = values_d;
        tjnpr_precompute::size_oes_d = size_oes_d;
        tjnpr_precompute::size_subgraph_d = size_subgraph_d;
        tjnpr_precompute::cur_subgraph_d = cur_subgraph_d;
        tjnpr_precompute::subgraph_neighbor_d = subgraph_neighbor_d;
        tjnpr_precompute::num = num;
        tjnpr_precompute::is_active_d = is_active_d;
        tjnpr_precompute::diff = 0;
    }
    __global__
    void delta_sum(){
        int index = threadIdx.x + blockIdx.x + blockDim.x;
        if(index < num){
            atomicAdd(&diff, deltas_d[index]);
        }
    }
    __global__
    void is_converge(double threshold, int *flag_d){
        // printf("diff is %.10f",diff);
        // printf("thresh is %.10f",threshold);
        if(diff < threshold){
            diff = 0;
            flag_d[0] = 1;
            
        }else{
            diff = 0;
            flag_d[0] = 0;
        }
    }
    // __global__
    // void clear(){
    //     int index = threadIdx.x + blockIdx.x + blockDim.x;
    //     if(index < num){
    //         is_active_d[index] = false;
    //     }
    // }
    int compute(unsigned int num, double threshold){
        int *flag = (int *)malloc(sizeof(int));
        flag[0] = 0;
        int *flag_d;
        cudaMalloc(&flag_d, sizeof(int));
        cudaMemcpy(flag_d, flag, sizeof(int), cudaMemcpyHostToDevice);

        dim3 block(512);
        dim3 grid((num - 1) / block.x + 1);
        compute_real<<<grid, block>>>();
        delta_sum<<<grid, block>>>();
        is_converge<<<1, 1>>>(threshold, flag_d);

        cudaMemcpy(flag, flag_d, sizeof(int), cudaMemcpyDeviceToHost);
        // clear<<<grid, block>>>();
        return *flag;
    }

    __device__
    void g_function(int index){
        values_d[index] += deltas_d[index];
        for(unsigned int i=cur_subgraph_d[index]; i < size_subgraph_d[index];i++){
            unsigned int dist_node = subgraph_neighbor_d[i];
            float outv = deltas_d[index] * 0.85 / size_oes_d[index];
            atomicAdd(&deltas_d[dist_node], outv);
        }
        deltas_d[index] = 0;
    }

    __global__
    void compute_real(){
        int index = threadIdx.x + blockIdx.x + blockDim.x;
        if(index < num){
            if(is_active_d[index]){
                g_function(index);
            }else{
                return ;
            }
        } 
    }
}