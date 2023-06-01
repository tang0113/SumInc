#include <cuda_runtime.h>
#include <stdio.h>
#include "my_worker.cuh"
#include "grape/utils/vertex_array.h"
namespace tjn{
  __device__ float *values_d;
  __device__ float *deltas_d;
  __device__ unsigned int start_d;
  __device__ unsigned int end_d;
  __device__ unsigned int *oeoffset_d;
  __device__ unsigned int *curOff_d;
  __device__ unsigned int *size_d;
  __device__ char *node_type_d;
  /**
   * @brief node type
  */
  enum NodeType {
    SingleNode = 0,
    OnlyOutNode = 1,
    OnlyInNode = 2,
    BothOutInNode = 3,
    InnerNode = 4,
    OutMaster = 5,
    BothOutInMaster = 6,
  };
  void value2last(float *last_values_d,float *values_d,unsigned int *v_d,int size){
    dim3 block(512);
    dim3 grid((size-1)/block.x + 1);
    value2last_real<<<grid,block>>>(last_values_d,values_d,v_d,size);
  }

  __global__
  void value2last_real(float *last_values_d,float *values_d,unsigned int *v_d,int size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < size){
      last_values_d[v_d[index]] = values_d[v_d[index]];
    }
    // int stride = gridDim.x * blockDim.x;
    // for(int i=index;i<size;i+=stride){
    //     last_values_d[ v_d[i] ] = values_d[ v_d[i] ];
    //     // printf("lastvalueis %f\n", last_values_d[v_d[i]]);
    //     // printf("numis %d\n", size);
    // }
  }
  void init(float *deltas_d, float *values_d, unsigned int *oeoffset_d, unsigned int *size_d, unsigned int start_d, unsigned int end_d, unsigned int *curOff_d, char *node_type_d){
    init_real<<<1,1>>>(deltas_d, values_d, oeoffset_d, size_d, start_d, end_d, curOff_d, node_type_d);
  }
  __global__
  void init_real(float *deltas_d, float *values_d, unsigned int *oeoffset_d, unsigned int *size_d, unsigned int start_d, unsigned int end_d, unsigned int *curOff_d, char *node_type_d){
    tjn::values_d = values_d;
    tjn::deltas_d = deltas_d;
    tjn::start_d = start_d;
    tjn::end_d = end_d;
    tjn::oeoffset_d = oeoffset_d;
    tjn::size_d = size_d;
    tjn::curOff_d = curOff_d;
    tjn::node_type_d = node_type_d;
  }
  void g_function_pr(unsigned int start_d, unsigned int end_d){
    dim3 block(512);
    dim3 grid((end_d - start_d - 1) / block.x + 1);
    g_function_pr_real<<<grid, block>>>();
  }

    __global__
  void g_function_pr_real(){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < end_d - start_d){
      // float delta = atomicExch(&deltas_d[index], 0);
      if(isChange_pr(deltas_d[index], end_d - start_d)){
        atomicAdd(&values_d[index], deltas_d[index]);
        // values_d[index] += deltas_d[index];
        // __syncthreads();
        // __threadfence();
        float delta = atomicExch(&deltas_d[index], 0);
        // __syncthreads();
        // __threadfence();
        // atomicAdd(&values_d[index], delta);
        // atomicAdd(&values_d[index], delta);
        // values_d[index] += delta;
        // values_d[index] += delta;
        unsigned int out_degree = max(size_d[index],1);
        // atomicExch(&deltas_d[index], 0);
        // __syncthreads();
        // __threadfence();
          float outv = delta * 0.85f / out_degree;
          for(unsigned int i=curOff_d[index];i<curOff_d[index] + size_d[index];i++){
            // deltas_d[ oeoffset_d[i] ] += outv;
            atomicAdd(&deltas_d[oeoffset_d[i]],outv);
          } 
          // __syncthreads();
          // __threadfence();
          
        
      }else{
        return ;
      }
      

    }else{
      return ;
    }
        // int row = threadIdx.y;//innervertices'order
    // int col = threadIdx.x;
    // if(index < offsize){
    //   if(isChange_pr(deltas_d[start_d + curId_d[index]], end_d - start_d)){
    //     double delta = deltas_d[start_d + curId_d[index]];
    //     unsigned int out_degree = size_d[start_d + curId_d[index]];
    //     double outv = delta * 0.85f / out_degree;
    //     // __syncthreads();
    //     // atomicAdd(&deltas_d[oeoffset_d[index]], outv);
    //     __syncthreads();
    //     __threadfence();
    //     deltas_d[curId_d[index]] = 0;
    //     __syncthreads();
    //     __threadfence();
    //     deltas_d[oeoffset_d[index]] += outv;
    //     if( index == 0 || curId_d[index] != curId_d[index-1]){
    //       // printf("curID is %d",curId_d[index]);
    //       // deltas_d[curId_d[index]] -= delta;
    //       __syncthreads();
    //     __threadfence();
    //       values_d[curId_d[index]] += deltas_d[curId_d[index]];
    //       // atomicAdd(&deltas_d[start_d + curId_d[index]], -delta);
    //       // atomicAdd(&values_d[start_d + curId_d[index]], deltas_d[start_d + curId_d[index]]);
    //     }

    //     __syncthreads();
    //     __threadfence();
    //   }
    // }
    // // if(oeoffset_d[row * blockDim.x + col] == end_d - start_d){
    //   if(size_d[start_d + row] == 0)deltas_d[row] = 0;
    //   return ;
    // }
    // if(isChange_pr(deltas_d[start_d + row], end_d - start_d)){

    //   float delta = deltas_d[start_d + row];

    //   unsigned int out_degree = size_d[start_d + row];
    //   float outv = delta * 0.85f / out_degree;
    //   __syncthreads();
    //   atomicAdd(&deltas_d[oeoffset_d[row * blockDim.x + col]], outv);
    //   if( (row*blockDim.x + col) % blockDim.x == 0){
    //     atomicAdd(&deltas_d[start_d + row], -delta);
    //     atomicAdd(&values_d[start_d + row], deltas_d[start_d + row]);
    //   }
    // }
  }

  void g_function_compr(unsigned int start_d, unsigned int end_d){
    dim3 block(512);
    dim3 grid((end_d - start_d - 1) / block.x + 1);
    g_function_compr_real<<<grid, block>>>();
  }

  __global__
  void g_function_compr_real(){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < end_d - start_d){
      switch(node_type_d[index]){
        case NodeType::SingleNode:
          {
            
          }
          break;
        case NodeType::OnlyInNode:
          {

          }
          break;
        case NodeType::OnlyOutNode:
          {

          }
          break;
        case NodeType::BothOutInNode:
          {

          }
          {

          }
          break;
        case NodeType::OutMaster:
          {

          }
          break;
        case NodeType::BothOutInMaster:
          {

          }
          {

          }
          break;
      }
    }else{
      return ;
    }
  }

  __device__
  bool isChange_pr(float delta, int verticesNum){
    if (fabs(delta) > (1e-6) / verticesNum) {
      return true;
    } else {
      return false;
    }
  }

}
