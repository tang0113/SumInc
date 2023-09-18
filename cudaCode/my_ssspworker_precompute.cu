#include <cuda_runtime.h>
#include <stdio.h>
#include "my_ssspworker_precompute.cuh"

namespace tjnsssp_precompute{
    __device__ unsigned int *size_subgraph_d;
    __device__ unsigned int *cur_subgraph_d;
    __device__ int *subgraph_data_d;
    __device__ unsigned int *subgraph_neighbor_d;
    __device__ unsigned int old_node_num;

    __device__ unsigned int *modified_d;
    __device__ unsigned int *modified_num_d;
    __device__ unsigned int *deltas_parent_d;
    __device__ int *deltas_d;
    __device__ int *values_d;
    void init_subgraph(unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, int *subgraph_data_d, unsigned int *subgraph_neighbor_d, unsigned int old_node_num){
        init_subgraph_real<<<1, 1>>>(size_subgraph_d, cur_subgraph_d, subgraph_data_d, subgraph_neighbor_d, old_node_num);
    }

    __global__
    void init_subgraph_real(unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, int *subgraph_data_d, unsigned int *subgraph_neighbor_d, unsigned int old_node_num){
        tjnsssp_precompute::size_subgraph_d = size_subgraph_d;
        tjnsssp_precompute::cur_subgraph_d = cur_subgraph_d;
        tjnsssp_precompute::subgraph_data_d = subgraph_data_d;
        tjnsssp_precompute::subgraph_neighbor_d = subgraph_neighbor_d;
        tjnsssp_precompute::old_node_num = old_node_num;
    }

    void init_modified(unsigned int *modified_d, int *deltas_d, unsigned int *deltas_parent_d, int *values_d, unsigned int *modified_num_d){
        init_modified_real<<<1, 1>>>(modified_d, deltas_d, deltas_parent_d, values_d, modified_num_d);
    }

    __global__
    void init_modified_real(unsigned int *modified_d, int *deltas_d, unsigned int *deltas_parent_d, int *values_d, unsigned int *modified_num_d){
        tjnsssp_precompute::modified_d = modified_d;
        tjnsssp_precompute::deltas_d = deltas_d;
        tjnsssp_precompute::deltas_parent_d = deltas_parent_d;
        tjnsssp_precompute::values_d = values_d;
        tjnsssp_precompute::modified_num_d = modified_num_d;
    }

    __global__
    void clear_all_num(){
        modified_num_d[0] = 0;
    }
    __global__
    void print_active_num(unsigned int num){
        if(modified_num_d[0] <50){
            // for(int i=0;i<num;i++){
            //     if(modified_d[i] == 1){
            //         printf("i is %d ",i);
            //     }
            // }
        }
    }
    void compute(unsigned int num){
        dim3 block(512);
        dim3 grid((num - 1) / block.x + 1);
        compute_real<<<grid, block>>>(num);
        cudaDeviceSynchronize();

        clear_all_num<<<1,1>>>();
        cudaDeviceSynchronize();

        count_active_num<<<grid, block>>>(num);
        cudaDeviceSynchronize();

        print_active_num<<<1, 1>>>(num);

        
    }

    __global__
    void compute_real(unsigned int num){
        unsigned int cur_node = threadIdx.x + blockIdx.x * blockDim.x;
        if(cur_node < num){
            if(modified_d[cur_node] == 0){
                return ;
            }else{
                
                modified_d[cur_node] = 0;
                if(values_d[cur_node] > deltas_d[cur_node]){
                    values_d[cur_node] = deltas_d[cur_node];
                    // atomicExch(&values_d[cur_node], deltas_d[cur_node]);
                    for(unsigned int i = cur_subgraph_d[cur_node];i < size_subgraph_d[cur_node] + cur_subgraph_d[cur_node]; i++){
                        
                        
                        unsigned int dist_node = subgraph_neighbor_d[i];
                        // if(subgraph_data_d[i] < 0)printf("niu");
                        // if(deltas_d[dist_node] < 0)printf("niu");  
                        if(dist_node == 233452){
                            // printf("dist deltas is %d ",deltas_d[dist_node]);
                        } 
                        int new_dist = deltas_d[cur_node] + subgraph_data_d[i];
                        bool is_update = false;
                        if(new_dist < deltas_d[dist_node]){
                            atomicMin(&deltas_d[dist_node], new_dist);
                            unsigned int e_parent = cur_node;
                            atomicExch(&deltas_parent_d[dist_node], e_parent);
                            is_update = true;
                        }
                        if(is_update && dist_node < old_node_num){
                            modified_d[dist_node] = 1;
                        }
                    }
                }
                
                    
            }
        }
    }

    __global__
    void count_active_num(unsigned int num){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num && modified_d[index]){
            atomicAdd(&modified_num_d[0], 1);
        }
    }

}