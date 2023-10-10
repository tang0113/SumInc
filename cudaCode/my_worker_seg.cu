#include <cuda_runtime.h>
#include <stdio.h>
#include "my_worker_seg.cuh"
namespace tjnpr_seg{
    __device__ float *values_d;
    __device__ float *deltas_d;
    __device__ float *spnode_datas_d;
    __device__ float *bound_node_values_d;

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

    __device__ unsigned int num;
    __device__ unsigned int oe_average_nodes;
    __device__ unsigned int is_average_nodes;
    __device__ unsigned int cur_seg;
    __device__ unsigned int seg_num;

    __device__ unsigned int *oe_edges_d;
    __device__ unsigned int *ib_edges_d;
    __device__ unsigned int *is_edges_d;
    __device__ unsigned int *sync_edges_d;

    __device__ unsigned int *isoffset_all_d;

    enum NodeType {
        SingleNode = 0,
        OnlyOutNode = 1,
        OnlyInNode = 2,
        BothOutInNode = 3,
        InnerNode = 4,
        OutMaster = 5,
        BothOutInMaster = 6,
    };

    void init(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, 
              unsigned int *oeoffset_d, unsigned int *iboffset_d, unsigned int *isoffset_d, unsigned int *syncoffset_d, 
              unsigned int *size_oe_d, unsigned int *size_ib_d, unsigned int *size_is_d, unsigned int *size_sync_d, 
              unsigned int num, unsigned int oe_average_nodes, unsigned int is_average_nodes, unsigned int cur_seg, unsigned int seg_num, 
              unsigned int *cur_oeoff_d, unsigned int *cur_iboff_d, unsigned int *cur_isoff_d, unsigned int *cur_syncoff_d, 
              char *node_type_d, float *is_edata_d, 
              unsigned int *all_out_mirror_d, unsigned int *mirrorid2vid_d, 
              unsigned int *oe_edges_d, unsigned int *ib_edges_d, unsigned int *is_edges_d, unsigned int *sync_edges_d){
        init_real<<<1, 1>>>(spnode_datas_d, bound_node_values_d, deltas_d, values_d, 
              oeoffset_d, iboffset_d, isoffset_d, syncoffset_d, 
              size_oe_d, size_ib_d, size_is_d, size_sync_d, 
              num, oe_average_nodes, is_average_nodes, cur_seg, seg_num, 
              cur_oeoff_d, cur_iboff_d, cur_isoff_d, cur_syncoff_d, 
              node_type_d, is_edata_d, 
              all_out_mirror_d, mirrorid2vid_d, 
              oe_edges_d, ib_edges_d, is_edges_d, sync_edges_d);
    }
    __global__
    void init_real(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, 
                   unsigned int *oeoffset_d, unsigned int *iboffset_d, unsigned int *isoffset_d, unsigned int *syncoffset_d, 
                   unsigned int *size_oe_d, unsigned int *size_ib_d, unsigned int *size_is_d, unsigned int *size_sync_d, 
                   unsigned int num, unsigned int oe_average_nodes, unsigned int is_average_nodes, unsigned int cur_seg, unsigned int seg_num, 
                   unsigned int *cur_oeoff_d, unsigned int *cur_iboff_d, unsigned int *cur_isoff_d, unsigned int *cur_syncoff_d, 
                   char *node_type_d, float *is_edata_d, 
                   unsigned int *all_out_mirror_d, unsigned int *mirrorid2vid_d, 
                   unsigned int *oe_edges_d, unsigned int *ib_edges_d, unsigned int *is_edges_d, unsigned int *sync_edges_d){

        tjnpr_seg::spnode_datas_d = spnode_datas_d;
        tjnpr_seg::bound_node_values_d = bound_node_values_d;
        tjnpr_seg::deltas_d = deltas_d;
        tjnpr_seg::values_d = values_d;

        tjnpr_seg::oeoffset_d = oeoffset_d;
        tjnpr_seg::iboffset_d = iboffset_d;
        tjnpr_seg::isoffset_d = isoffset_d;
        tjnpr_seg::syncoffset_d = syncoffset_d;

        tjnpr_seg::size_oe_d = size_oe_d;
        tjnpr_seg::size_ib_d = size_ib_d;
        tjnpr_seg::size_is_d = size_is_d;
        tjnpr_seg::size_sync_d = size_sync_d;

        tjnpr_seg::cur_oeoff_d = cur_oeoff_d;
        tjnpr_seg::cur_iboff_d = cur_iboff_d;
        tjnpr_seg::cur_isoff_d = cur_isoff_d;
        tjnpr_seg::cur_syncoff_d = cur_syncoff_d;

        tjnpr_seg::node_type_d = node_type_d;
        tjnpr_seg::is_edata_d = is_edata_d;
        tjnpr_seg::all_out_mirror_d = all_out_mirror_d;
        tjnpr_seg::mirrorid2vid_d = mirrorid2vid_d;
        
        tjnpr_seg::num = num;
        tjnpr_seg::oe_average_nodes = oe_average_nodes;
        tjnpr_seg::is_average_nodes = is_average_nodes;
        tjnpr_seg::cur_seg = cur_seg;
        tjnpr_seg::seg_num = seg_num;

        tjnpr_seg::oe_edges_d = oe_edges_d;
        tjnpr_seg::ib_edges_d = ib_edges_d;
        tjnpr_seg::is_edges_d = is_edges_d;
        tjnpr_seg::sync_edges_d = sync_edges_d;

        // tjnpr_seg::isoffset_all_d = isoffset_all_d;
    }

    __global__
    void deltaSum_real(float *result_d){
        // printf("tjn");
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num){
            atomicAdd(&result_d[0], deltas_d[index]);
        }
    }
    float deltaSum(unsigned int start, unsigned int end){
        dim3 block(512);
        dim3 grid((end - start - 1) / block.x + 1);
        float *result_h = (float *)malloc(sizeof(float) * 1);
        
        result_h[0] = 0;
        float *result_d;
        cudaMalloc(&result_d, sizeof(float) * 1);
        // cudaError_t err = cudaGetLastError();
        // printf("cudaFunction1:%s\n",cudaGetErrorString(err));
        cudaMemcpy(result_d, result_h, sizeof(float) * 1, cudaMemcpyHostToDevice);
        // err = cudaGetLastError();
        // printf("cudaFunction1:%s\n",cudaGetErrorString(err));
        deltaSum_real<<<grid, block>>>(result_d);
        // printf("yesno");
        cudaMemcpy(result_h, result_d, sizeof(float) * 1, cudaMemcpyDeviceToHost);
        return result_h[0];
    }
    __global__
    void g_function_pr_real(){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num){
            pr_Ingress(index);
        }else{
            return ;
        }
    }
    __global__
    void cursegChange(){
        cur_seg++;
        cur_seg %= seg_num;
    }
    void g_function_pr(unsigned int num){
        dim3 block(512);
        dim3 grid( (num - 1) / block.x + 1);
        g_function_pr_real<<<grid, block>>>();
        cursegChange<<<1, 1>>>();
    }

    __global__
    void g_function_compr_real(unsigned int *isoffset_all_d, float *isdata_all_d){
        
        int index = threadIdx.x + blockIdx.x * blockDim.x;
      if(index < num){
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
              pr_outMaster(index, isoffset_all_d, isdata_all_d);
            }
            break;
          case NodeType::BothOutInMaster:
            {
              pr_bothOutInMaster(index, isoffset_all_d, isdata_all_d);
            }
            break;
        }
        // __syncthreads();
      }else{
        return ;
      }
    }
    void g_function_compr(unsigned int num, unsigned int *isoffset_all_d, float *isdata_all_d){
        dim3 block(512);
        dim3 grid( (num-1) / block.x + 1);
        g_function_compr_real<<<grid, block>>>(isoffset_all_d, isdata_all_d);
        cursegChange<<<1,1>>>();
    }
    
    __global__
    void OutMirrorSyncToMaster_real(unsigned int size){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < size && isChange_pr(bound_node_values_d[all_out_mirror_d[index]], num)){
            float delta = atomicExch(&bound_node_values_d[all_out_mirror_d[index]], 0);
            atomicAdd(&deltas_d[mirrorid2vid_d[index]], delta);
        }else{
            return ;
        }
    }
    void OutMirrorSyncToMaster(unsigned int size){
        dim3 block(512);
        dim3 grid((size - 1) / block.x + 1);
        OutMirrorSyncToMaster_real<<<grid, block>>>(size);
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
    void pr_Ingress(int index){
        if(cur_seg == 0 && index >= oe_average_nodes){
            return ;
        }else if(cur_seg != 0 && (index < oe_average_nodes * cur_seg || index >= oe_average_nodes * (cur_seg + 1))){
            return ;
        }else if(isChange_pr(deltas_d[index], num)){
            int pre_edges = 0;
            for(int i=0;i<cur_seg;i++){
                pre_edges += oe_edges_d[i];
            }
            float delta = atomicExch(&deltas_d[index], 0);
            unsigned int out_degree = max(size_oe_d[index],1);
            float outv = delta * 0.85f / out_degree;
            for(unsigned int i=cur_oeoff_d[index];i<cur_oeoff_d[index] + size_oe_d[index];i++){
                atomicAdd(&deltas_d[oeoffset_d[i-pre_edges]],outv);
            } 
            atomicAdd(&values_d[index], delta);
        }else{
            return ;
        }
    }

    __device__
    void pr_singleNode(int index){
        if(cur_seg == 0 && index >= is_average_nodes){
            return ;
        }else if(cur_seg != 0 && (index < is_average_nodes * cur_seg || index >= is_average_nodes * (cur_seg + 1))){
            return ;
        }else if(isChange_pr(deltas_d[index], num)){
            unsigned int pre_edges = 0;
            for(int i=0;i<cur_seg;i++){
                pre_edges += ib_edges_d[i];
            }
            float delta = atomicExch(&deltas_d[index], 0);
            unsigned int out_degree = max(size_ib_d[index],1);
            float outv = delta * 0.85f / out_degree;
            for(unsigned int i=cur_iboff_d[index];i<cur_iboff_d[index] + size_ib_d[index];i++){
                // atomicAdd(&deltas_d[iboffset_d[i-pre_edges]],outv);
                // if(i < pre_edges){
                //     printf("1");
                // }
                // printf("pre edges is %u",pre_edges);
                unsigned int temp = i - pre_edges;
                atomicAdd(&deltas_d[iboffset_d[temp]],outv);
            } 
            atomicAdd(&values_d[index], delta);
        }else {
            return ;
        }
    }

    __device__
    void pr_onlyInNode(int index){
        if(cur_seg == 0 && index >= is_average_nodes){
            return ;
        }else if(cur_seg != 0 && (index < is_average_nodes * cur_seg || index >= is_average_nodes * (cur_seg + 1))){
            return ;
        }else if(isChange_pr(deltas_d[index], num)){
            int pre_edges = 0;
            for(int i=0;i<cur_seg;i++){
                pre_edges += is_edges_d[i];
            }
            float delta = atomicExch(&deltas_d[index], 0);
            for(unsigned int i=cur_isoff_d[index];i<cur_isoff_d[index] + size_is_d[index];i++){
                atomicAdd(&bound_node_values_d[isoffset_d[i-pre_edges]],delta * is_edata_d[i-pre_edges]);
            }
            atomicAdd(&spnode_datas_d[index], delta);
        }
    }

    __device__
    void pr_onlyOutNode(int index){
        if(cur_seg == 0 && index >= is_average_nodes){
            return ;
        }else if(cur_seg != 0 && (index < is_average_nodes * cur_seg || index >= is_average_nodes * (cur_seg + 1))){
            return ;
        }else if(isChange_pr(bound_node_values_d[index], num)){
            int pre_edges = 0;
            for(int i=0;i<cur_seg;i++){
                pre_edges += is_edges_d[i];
            }
            atomicAdd(&values_d[index], bound_node_values_d[index]);
            float delta = atomicExch(&bound_node_values_d[index], 0);
            unsigned int old_out_degree = size_oe_d[index];
            unsigned int out_degree = size_ib_d[index];
            if(out_degree > 0){
                float outv = delta * 0.85f / old_out_degree;
                for(unsigned int i=cur_iboff_d[index];i<cur_iboff_d[index] + size_ib_d[index];i++){
                    atomicAdd(&deltas_d[iboffset_d[i-pre_edges]],outv);
                }
            }
        }
    }

    __device__
    void pr_bothOutInNode(int index){
        pr_onlyInNode(index);
        pr_onlyOutNode(index);
    }

    __device__
    void pr_outMaster(int index, unsigned int *isoffset_all_d, float *isdata_all_d){
        bool flag0 = true;
        if(cur_seg == 0 && index >= is_average_nodes){
            flag0 = false;
        }else if(cur_seg != 0 && (index < is_average_nodes * cur_seg || index >= is_average_nodes * (cur_seg + 1))){
            flag0 = false ;
        }else if(isChange_pr(bound_node_values_d[index], num) && flag0){
            int pre_edges = 0;
            for(int i=0;i<cur_seg;i++){
                pre_edges += ib_edges_d[i];
            }
            atomicAdd(&values_d[index], bound_node_values_d[index]);
            float delta = atomicExch(&bound_node_values_d[index], 0);
            unsigned int old_out_degree = size_oe_d[index];
            unsigned int out_degree = size_ib_d[index];
            if(out_degree > 0){
                float outv = delta * 0.85f / old_out_degree;
                for(unsigned int i=cur_iboff_d[index];i<cur_iboff_d[index] + size_ib_d[index];i++){
                    atomicAdd(&deltas_d[iboffset_d[i-pre_edges]],outv);
                }
            }

            pre_edges = 0;
            int is_pre = 0;
            for(int i=0;i<cur_seg;i++){
                pre_edges += sync_edges_d[i];
                is_pre += is_edges_d[i];
            }

            for(unsigned int i=cur_syncoff_d[index];i<cur_syncoff_d[index] + size_sync_d[index];i++){
                atomicAdd(&deltas_d[syncoffset_d[i-pre_edges]],delta);
                float delta = atomicExch(&deltas_d[syncoffset_d[i-pre_edges]],0);
                for(unsigned int j=cur_isoff_d[syncoffset_d[i-pre_edges]];j<cur_isoff_d[syncoffset_d[i-pre_edges]] + size_is_d[syncoffset_d[i-pre_edges]];j++){
                    if((cur_seg == 0 && syncoffset_d[i-pre_edges] >= is_average_nodes)
                    || (cur_seg != 0 && (syncoffset_d[i-pre_edges] < is_average_nodes * cur_seg || syncoffset_d[i-pre_edges] >= is_average_nodes * (cur_seg + 1)))){
                        
                        atomicAdd(&bound_node_values_d[isoffset_all_d[j]],delta * isdata_all_d[j]);
                    
                    }else{
                        atomicAdd(&bound_node_values_d[isoffset_d[j-is_pre]],delta * is_edata_d[j-is_pre]);
                    }
                    // atomicAdd(&bound_node_values_d[isoffset_d[j-is_pre]],delta * is_edata_d[j-is_pre]);
                }
                atomicAdd(&spnode_datas_d[syncoffset_d[i-pre_edges]], delta);
            }
        }
    }

    __device__
    void pr_bothOutInMaster(int index, unsigned int *isoffset_all_d, float *isdata_all_d){
        pr_onlyInNode(index);
        pr_outMaster(index, isoffset_all_d, isdata_all_d);
    }

}