#include <cuda_runtime.h>
#include <stdio.h>
#include "my_worker.cuh"
#include "glog/src/glog/logging.h"
namespace tjn{
  __device__ float *values_d;
  __device__ float *deltas_d;
  __device__ float *spnode_datas_d;
  __device__ float *bound_node_values_d;

  __device__ unsigned int start_d;
  __device__ unsigned int end_d;

  __device__ unsigned int *oeoffset_d;
  __device__ unsigned int *iboffset_d;
  __device__ unsigned int *isoffset_d;
  __device__ unsigned int *syncoffset_d;

  __device__ unsigned int *cur_oeoff_d;
  __device__ unsigned int *cur_iboff_d;
  __device__ unsigned int *cur_isoff_d;
  __device__ unsigned int *cur_syncoff_d;

  __device__ unsigned int *size_oe_d;
  __device__ unsigned int *size_ib_d;
  __device__ unsigned int *size_is_d;
  __device__ unsigned int *size_sync_d;
  __device__ char *node_type_d;
  __device__ float *is_edata_d;

  __device__ unsigned int *all_out_mirror_d;
  __device__ unsigned int *mirrorid2vid_d;
  
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
  void init(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, 
            unsigned int *oeoffset_d, unsigned int *iboffset_d, unsigned int *isoffset_d, unsigned int *syncoffset_d, 
            unsigned int *size_oe_d, unsigned int *size_ib_d, unsigned int *size_is_d, unsigned int *size_sync_d, 
            unsigned int start_d, unsigned int end_d, 
            unsigned int *cur_oeoff_d, unsigned int *cur_iboff_d, unsigned int *cur_isoff_d, unsigned int *cur_syncoff_d, 
            char *node_type_d, float *is_edata_d, 
            unsigned int *all_out_mirror_d, unsigned int *mirrorid2vid_d){

      init_real<<<1,1>>>(spnode_datas_d, bound_node_values_d, deltas_d, values_d, 
                      oeoffset_d, iboffset_d, isoffset_d, syncoffset_d, 
                      size_oe_d, size_ib_d, size_is_d, size_sync_d, 
                      start_d, end_d, 
                      cur_oeoff_d, cur_iboff_d, cur_isoff_d, cur_syncoff_d, 
                      node_type_d, is_edata_d, 
                      all_out_mirror_d, mirrorid2vid_d);

  }

  __global__
  void init_real(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, 
            unsigned int *oeoffset_d, unsigned int *iboffset_d, unsigned int *isoffset_d, unsigned int *syncoffset_d, 
            unsigned int *size_oe_d, unsigned int *size_ib_d, unsigned int *size_is_d, unsigned int *size_sync_d, 
            unsigned int start_d, unsigned int end_d, 
            unsigned int *cur_oeoff_d, unsigned int *cur_iboff_d, unsigned int *cur_isoff_d, unsigned int *cur_syncoff_d, 
            char *node_type_d, float *is_edata_d, 
            unsigned int *all_out_mirror_d, unsigned int *mirrorid2vid_d){

        tjn::values_d = values_d;
        tjn::deltas_d = deltas_d;
        tjn::spnode_datas_d = spnode_datas_d;
        tjn::bound_node_values_d = bound_node_values_d;

        tjn::start_d = start_d;
        tjn::end_d = end_d;

        tjn::oeoffset_d = oeoffset_d;
        tjn::iboffset_d = iboffset_d;
        tjn::isoffset_d = isoffset_d;
        tjn::syncoffset_d = syncoffset_d;

        tjn::size_oe_d = size_oe_d;
        tjn::size_ib_d = size_ib_d;
        tjn::size_is_d = size_is_d;
        tjn::size_sync_d = size_sync_d;

        tjn::cur_oeoff_d = cur_oeoff_d;
        tjn::cur_iboff_d = cur_iboff_d;
        tjn::cur_isoff_d = cur_isoff_d;
        tjn::cur_syncoff_d = cur_syncoff_d;

        tjn::node_type_d = node_type_d;
        tjn::is_edata_d = is_edata_d;
        
        tjn::all_out_mirror_d = all_out_mirror_d;
        tjn::mirrorid2vid_d = mirrorid2vid_d;

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
        pr_Ingress(index);
        // atomicAdd(&values_d[index], deltas_d[index]);
        // // values_d[index] += deltas_d[index];
        // // __syncthreads();
        // // __threadfence();
        // float delta = atomicExch(&deltas_d[index], 0);
        // // __syncthreads();
        // // __threadfence();
        // // atomicAdd(&values_d[index], delta);
        // // atomicAdd(&values_d[index], delta);
        // // values_d[index] += delta;
        // // values_d[index] += delta;
        // unsigned int out_degree = max(size_oe_d[index],1);
        // // atomicExch(&deltas_d[index], 0);
        // // __syncthreads();
        // // __threadfence();
        // float outv = delta * 0.85f / out_degree;
        // for(unsigned int i=cur_oeoff_d[index];i<cur_oeoff_d[index] + size_oe_d[index];i++){
        //   // deltas_d[ oeoffset_d[i] ] += outv;
        //   atomicAdd(&deltas_d[oeoffset_d[i]],outv);
        // } 
        // // __syncthreads();
        // // __threadfence();
    }else{
      return ;
    }
        // int row = threadIdx.y;//innervertices'order
    // int col = threadIdx.x;
    // if(index < offsize){
    //   if(isChange_pr(deltas_d[start_d + curId_d[index]], end_d - start_d)){
    //     double delta = deltas_d[start_d + curId_d[index]];
    //     unsigned int out_degree = size_oe_d[start_d + curId_d[index]];
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
    //   if(size_oe_d[start_d + row] == 0)deltas_d[row] = 0;
    //   return ;
    // }
    // if(isChange_pr(deltas_d[start_d + row], end_d - start_d)){

    //   float delta = deltas_d[start_d + row];

    //   unsigned int out_degree = size_oe_d[start_d + row];
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
              pr_singleNode(index);
              
            }
            break;
          case NodeType::OnlyInNode:
            {
              pr_onlyInNode(index);
            }
            break;
          case NodeType::OnlyOutNode:
            {
              pr_onlyOutNode(index);
            }
            break;
          case NodeType::BothOutInNode:
            {
              pr_bothOutInNode(index);
            }
            break;
          case NodeType::OutMaster:
            {
              pr_outMaster(index);
            }
            break;
          case NodeType::BothOutInMaster:
            {
              pr_bothOutInMaster(index);
            }
            break;
        }
        // __syncthreads();
      }else{
        return ;
      }
  }

  void OutMirrorSyncToMaster(unsigned int size){
    dim3 block(512);
    dim3 grid((size - 1) / block.x + 1);
    OutMirrorSyncToMaster_real<<<grid, block>>>(size);
  }

  __global__
  void OutMirrorSyncToMaster_real(unsigned int size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < size && isChange_pr(bound_node_values_d[all_out_mirror_d[index]], end_d - start_d)){
      float delta = atomicExch(&bound_node_values_d[all_out_mirror_d[index]], 0);
      atomicAdd(&deltas_d[mirrorid2vid_d[index]], delta);
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

  __device__
  inline void pr_Ingress(int index){
    
    if(isChange_pr(deltas_d[index], end_d - start_d)){
      // atomicAdd(&values_d[index], deltas_d[index]);//加在此处也可

      float delta = atomicExch(&deltas_d[index], 0);

      unsigned int out_degree = max(size_oe_d[index],1);

      float outv = delta * 0.85f / out_degree;

      for(unsigned int i=cur_oeoff_d[index];i<cur_oeoff_d[index] + size_oe_d[index];i++){

        atomicAdd(&deltas_d[oeoffset_d[i]],outv);

      } 

      atomicAdd(&values_d[index], delta);

    }else{
      return ;
    }

  }


  __device__
  inline void pr_singleNode(int index){//g_function(*graph_, u, value, delta, oes)
    // printf("singleNOde");

    if(isChange_pr(deltas_d[index], end_d - start_d)){
      // atomicAdd(&values_d[index], deltas_d[index]);//加在此处也可

      float delta = atomicExch(&deltas_d[index], 0);

      unsigned int out_degree = max(size_ib_d[index],1);

      float outv = delta * 0.85f / out_degree;

      for(unsigned int i=cur_iboff_d[index];i<cur_iboff_d[index] + size_ib_d[index];i++){

        atomicAdd(&deltas_d[iboffset_d[i]],outv);

      } 

      atomicAdd(&values_d[index], delta);
    }
  }

  __device__
  inline void pr_onlyInNode(int index){//g_index_function(*graph_, u, value, delta, adj, bound_node_values)     
    // printf("onlyInNode");
    
    if(isChange_pr(deltas_d[index], end_d - start_d)){
      float delta = atomicExch(&deltas_d[index], 0);
      // unsigned int out_degree = max(size_is_d[index],1);
      // float outv = delta * 0.85f / out_degree;
      for(unsigned int i=cur_isoff_d[index];i<cur_isoff_d[index] + size_is_d[index];i++){
        atomicAdd(&bound_node_values_d[isoffset_d[i]],delta * is_edata_d[i]);
      }
      atomicAdd(&spnode_datas_d[index], delta);
    }             

  }

  __device__
  inline void pr_onlyOutNode(int index){//g_function(*graph_, u, value, delta, oes, adj)
    // printf("OnlyOutNode");
    if(isChange_pr(bound_node_values_d[index], end_d - start_d)){
      atomicAdd(&values_d[index], bound_node_values_d[index]);
      float delta = atomicExch(&bound_node_values_d[index], 0);

      unsigned int old_out_degree = size_oe_d[index];
      
      unsigned int out_degree = size_ib_d[index];
      if(out_degree > 0){
        float outv = delta * 0.85f / old_out_degree;
        for(unsigned int i=cur_iboff_d[index];i<cur_iboff_d[index] + size_ib_d[index];i++){
          atomicAdd(&deltas_d[iboffset_d[i]],outv);
        }
      }

      

    }

  }

  __device__
  inline void pr_bothOutInNode(int index){
    pr_onlyInNode(index);
    pr_onlyOutNode(index);
  }

  __device__
  inline void pr_outMaster(int index){
    // printf("outMaster");
    if(isChange_pr(bound_node_values_d[index], end_d - start_d)){

      float delta = atomicExch(&bound_node_values_d[index], 0);

      unsigned int old_out_degree = size_oe_d[index];

      unsigned int out_degree = size_ib_d[index];

      if(out_degree > 0){
        float outv = delta * 0.85f / old_out_degree;
        for(unsigned int i=cur_iboff_d[index];i<cur_iboff_d[index] + size_ib_d[index];i++){
          atomicAdd(&deltas_d[iboffset_d[i]],outv);
        }
      }

      atomicAdd(&values_d[index], delta);

      for(unsigned int i=cur_syncoff_d[index];i<cur_syncoff_d[index] + size_sync_d[index];i++){
        atomicAdd(&deltas_d[syncoffset_d[i]],delta);
        float delta = atomicExch(&deltas_d[syncoffset_d[i]],0);
        for(unsigned int j=cur_isoff_d[syncoffset_d[i]];j<cur_isoff_d[syncoffset_d[i]] + size_is_d[syncoffset_d[i]];j++){
          atomicAdd(&bound_node_values_d[isoffset_d[j]],delta * is_edata_d[j]);
        }
        atomicAdd(&spnode_datas_d[syncoffset_d[i]], delta);
      }

    }

  }

  __device__
  inline void pr_bothOutInMaster(int index){

    // printf("bothOutInMaster");
    pr_onlyInNode(index);
    pr_outMaster(index);
  }
}
