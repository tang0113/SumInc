#include <cuda_runtime.h>
#include <stdio.h>
#include "my_ssspworker_seg.cuh"
// #include "freshman.h"
namespace tjnsssp_seg{

    __device__ unsigned int *iboffset_d;
    __device__ unsigned int *isoffset_d;

    __device__ unsigned int *cur_iboff_d;
    __device__ unsigned int *cur_isoff_d;

    __device__ unsigned int *size_ib_d;
    __device__ unsigned int *size_is_d;

    __device__ int *ib_edata_d;
    __device__ int *is_edata_d;

    __device__ int *deltas_d;
    __device__ int *values_d;

    __device__ unsigned int *cur_modified_size_d;//长度为1,只记录当前修改队列的长度
    __device__ unsigned int next_modified_allsize_d;
    __device__ unsigned int *is_modified_d;//记录顶点是否被修改
    __device__ int *last_modified_d;
    __device__ unsigned int num;

    __device__ char *node_type_d;

    __device__ unsigned int *ib_seg_start_d;
    __device__ unsigned int *is_seg_start_d;

    __device__ unsigned int *ib_seg_end_d;
    __device__ unsigned int *is_seg_end_d;

    __device__ unsigned int *ib_seg_end_edges_d;
    __device__ unsigned int *is_seg_end_edges_d;

    __device__ unsigned int *cur_seg_d;
    __device__ unsigned int *seg_num_d;

    __device__ unsigned int *ib_average_edges_d;
    __device__ unsigned int *is_average_edges_d;


    void init(int *deltas_d, int *values_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d, unsigned int *cur_seg_d, unsigned int *seg_num_d, 
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, unsigned int *ib_average_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d, unsigned int *is_average_edges_d){

        init_real<<<1,1>>>(deltas_d, values_d, FLAGS_sssp_source, 
                   cur_modified_size_d, is_modified_d, last_modified_d, num, 
                   iboffset_d, ib_edata_d, cur_iboff_d, size_ib_d, 
                   isoffset_d, is_edata_d, cur_isoff_d, size_is_d, 
                   node_type_d, cur_seg_d, seg_num_d, 
                   ib_seg_start_d, ib_seg_end_d, ib_seg_end_edges_d, ib_average_edges_d, 
                   is_seg_start_d, is_seg_end_d, is_seg_end_edges_d, is_average_edges_d);

    }

    __global__
    void init_real(int *deltas_d, int *values_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d, unsigned int *cur_seg_d, unsigned int *seg_num_d,
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, unsigned int *ib_average_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d, unsigned int *is_average_edges_d){
        
        tjnsssp_seg::iboffset_d = iboffset_d;
        tjnsssp_seg::isoffset_d = isoffset_d;

        tjnsssp_seg::ib_edata_d = ib_edata_d;
        tjnsssp_seg::is_edata_d = is_edata_d;

        tjnsssp_seg::cur_iboff_d = cur_iboff_d;
        tjnsssp_seg::cur_isoff_d = cur_isoff_d;

        tjnsssp_seg::size_ib_d = size_ib_d;
        tjnsssp_seg::size_is_d = size_is_d;

        tjnsssp_seg::deltas_d = deltas_d;
        tjnsssp_seg::values_d = values_d;

        tjnsssp_seg::cur_modified_size_d = cur_modified_size_d;
        tjnsssp_seg::cur_modified_size_d[0] = 1;

        tjnsssp_seg::num = num;
        tjnsssp_seg::is_modified_d = is_modified_d;
        tjnsssp_seg::last_modified_d = last_modified_d;
        tjnsssp_seg::last_modified_d[FLAGS_sssp_source] = 1;

        tjnsssp_seg::node_type_d = node_type_d;
        if(node_type_d[FLAGS_sssp_source] == 3){
            tjnsssp_seg::last_modified_d[FLAGS_sssp_source] = 2;
        }
        tjnsssp_seg::ib_seg_start_d = ib_seg_start_d;
        tjnsssp_seg::is_seg_start_d = is_seg_start_d;

        tjnsssp_seg::ib_seg_end_d = ib_seg_end_d;
        tjnsssp_seg::is_seg_end_d = is_seg_end_d;

        tjnsssp_seg::ib_seg_end_edges_d = ib_seg_end_edges_d;
        tjnsssp_seg::is_seg_end_edges_d = is_seg_end_edges_d;

        tjnsssp_seg::ib_average_edges_d = ib_average_edges_d;
        tjnsssp_seg::is_average_edges_d = is_average_edges_d;

        tjnsssp_seg::cur_seg_d = cur_seg_d;
        tjnsssp_seg::cur_seg_d[0] = 0;

        tjnsssp_seg::seg_num_d = seg_num_d;

    }

    void g_function_compr(unsigned int cpr_num){
        dim3 block(512);
        dim3 grid((cpr_num - 1) / block.x + 1);

        clearCurSize<<<1, 1>>>();

        g_function_compr_real<<<grid, block>>>();
        cudaDeviceSynchronize();

        setCurSize<<<grid, block>>>();
        cudaDeviceSynchronize();

        setCurSeg<<<1, 1>>>();
        cudaDeviceSynchronize();

        getMaxValue<<<1, 1>>>();
    }

    __global__
    void clearCurSize(){
        cur_modified_size_d[0] = 0;
    }

    __global__
    void setCurSeg(){
        cur_seg_d[0]++;
        cur_seg_d[0] %= seg_num_d[0];
    }

    __global__
    void getMaxValue(){
        if(cur_modified_size_d[0] == 0){
            int maxnum = 0;
            int id = 0;
            for(int i=0;i<num;i++){
                if(values_d[i] > maxnum && values_d[i] != 2147483647){
                    maxnum = values_d[i];
                    id = i;
                }
            }
            printf("maxnum is %d\n",maxnum);
            printf("maxid is %d",id);
        }
    }

    __global__
    void g_function_compr_real(){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num){
            if(last_modified_d[index] <= 0){
                last_modified_d[index] = 0;
                return ;
            }else{
                // if(last_modified_d[index] == 0)printf("666");
                switch(node_type_d[index]){
                    case 0:
                        {
                            sssp_nodeTypeZeroAndOne(index);
                            break;
                        }
                    case 1:
                        {
                            sssp_nodeTypeZeroAndOne(index);
                            break;
                        }
                    case 2:
                        {
                            sssp_nodeTypeTwo(index);
                            break;
                        }
                    case 3:
                        {
                            sssp_nodeTypeThree(index);
                            break;
                        }
                }
            }
        }
    }

    __device__
    void sssp_nodeTypeZeroAndOne(int index){
        // printf("in1");
        if(index < ib_seg_start_d[cur_seg_d[0]] || index > ib_seg_end_d[cur_seg_d[0]]){
            return ;
        }
        int e_num = size_ib_d[index];
        int start = 0;
        last_modified_d[index] = 0;
        if(index == ib_seg_start_d[cur_seg_d[0]]){
            if(cur_seg_d[0] != 0 && ib_seg_start_d[cur_seg_d[0]] == ib_seg_end_d[cur_seg_d[0] - 1]){
                e_num = size_ib_d[index] - ib_seg_end_edges_d[cur_seg_d[0] - 1];
                start = ib_seg_end_edges_d[cur_seg_d[0] - 1];
            }
        }else if(index == ib_seg_end_d[cur_seg_d[0]]){
            start = 0;
            e_num = ib_seg_end_edges_d[cur_seg_d[0]];
            if(e_num < size_ib_d[index]){
                last_modified_d[index] = 1;
            }
        }else{
            
        }
        if(values_d[index] > deltas_d[index]){
            // values_d[cur_modified_node] = deltas_d[cur_modified_node];
            atomicExch(&values_d[index], deltas_d[index]);
            for(unsigned int i = 0; i < e_num; i++){
                unsigned int dist_node = iboffset_d[cur_iboff_d[index] - cur_seg_d[0] * ib_average_edges_d[0] + start + i];
                // printf("dist node is %d",dist_node);
                // printf("cur off is %d",cur_iboff_d[index] - cur_seg_d[0] * ib_average_edges_d[0] + start + i);
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[index];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&last_modified_d[dist_node], 1);
                    if(node_type_d[dist_node] == 3){
                        atomicExch(&last_modified_d[dist_node], 2);
                    }
                    
                }
            }
        }
        // printf("in1");
    }

    __device__
    void sssp_nodeTypeTwo(int index){
        // printf("in2");
        if(index < is_seg_start_d[cur_seg_d[0]] || index > is_seg_end_d[cur_seg_d[0]]){
            return ;
        }
        int e_num = size_is_d[index];
        int start = 0;
        last_modified_d[index] = 0;
        if(index == is_seg_start_d[cur_seg_d[0]]){
            if(cur_seg_d[0] != 0 && is_seg_start_d[cur_seg_d[0]] == is_seg_end_d[cur_seg_d[0] - 1]){
                e_num = size_is_d[index] - is_seg_end_edges_d[cur_seg_d[0] - 1];
                start = is_seg_end_edges_d[cur_seg_d[0] - 1];
                
            }
        }else if(index == is_seg_end_d[cur_seg_d[0]]){
            start = 0;
            e_num = is_seg_end_edges_d[cur_seg_d[0]];
            if(e_num < size_is_d[index]){
                last_modified_d[index] = 1;
            }
        }else{
            
        }
        if(values_d[index] > deltas_d[index]){
            // values_d[cur_modified_node] = deltas_d[cur_modified_node];
            atomicExch(&values_d[index], deltas_d[index]);
            for(unsigned int i = 0; i < e_num; i++){
                unsigned int dist_node = isoffset_d[cur_isoff_d[index] - cur_seg_d[0] * is_average_edges_d[0] + start + i];
                // printf("dist node is %d",dist_node);
                // printf("cur off is %d",cur_isoff_d[index] - cur_seg_d[0] * is_average_edges_d[0] + start + i);
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = is_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[index];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&last_modified_d[dist_node], 1);
                    if(node_type_d[dist_node] == 3){
                        atomicExch(&last_modified_d[dist_node], 2);
                    }
                    
                }
            }
        }
        // printf("in2");
    }

    __device__
    void sssp_nodeTypeThree(int index){
        // printf("in3");
        //这里有问题,处理了两遍,会产生负数的结果
        int temp0 = last_modified_d[index];
        bool flag = true;
        if(index < ib_seg_start_d[cur_seg_d[0]] || index > ib_seg_end_d[cur_seg_d[0]]){
            flag = false;
        }
        if(flag){
            last_modified_d[index] -= 1;
            int e_num = size_ib_d[index];
            int start = 0;
            if(index == ib_seg_start_d[cur_seg_d[0]]){
                if(cur_seg_d[0] != 0 && ib_seg_start_d[cur_seg_d[0]] == ib_seg_end_d[cur_seg_d[0] - 1]){
                    e_num = size_ib_d[index] - ib_seg_end_edges_d[cur_seg_d[0] - 1];
                    start = ib_seg_end_edges_d[cur_seg_d[0] - 1];
                }
            }else if(index == ib_seg_end_d[cur_seg_d[0]]){
                start = 0;
                e_num = ib_seg_end_edges_d[cur_seg_d[0]];
                if(e_num < size_ib_d[index]){
                    last_modified_d[index] += 1;
                }
            }else{
                
            }
            if(values_d[index] > deltas_d[index]){
                // values_d[cur_modified_node] = deltas_d[cur_modified_node];
                atomicExch(&values_d[index], deltas_d[index]);
                for(unsigned int i = 0; i < e_num; i++){
                    unsigned int dist_node = iboffset_d[cur_iboff_d[index] - cur_seg_d[0] * ib_average_edges_d[0] + start + i];
                    // printf("dist node is %d",dist_node);
                    // printf("cur off is %d",cur_iboff_d[index] - cur_seg_d[0] * ib_average_edges_d[0] + start + i);
                    // if(cur_modified_node == 9){
                    //     printf("9 dist is %d",dist_node);
                    // }
                    // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                    int new_dist = 1 + deltas_d[index];//无权测试
                    if(new_dist < deltas_d[dist_node]){
                        atomicMin(&deltas_d[dist_node], new_dist);
                        atomicExch(&last_modified_d[dist_node], 1);
                        if(node_type_d[dist_node] == 3){
                            atomicExch(&last_modified_d[dist_node], 2);
                        }
                    }
                }
            }
        }
        int temp = last_modified_d[index];
        //第二阶段
        flag = true;
        if(index < is_seg_start_d[cur_seg_d[0]] || index > is_seg_end_d[cur_seg_d[0]]){
            flag = false ;
        }
        if(flag){
            last_modified_d[index] -= 1;
            int e_num = size_is_d[index];
            int start = 0;
            if(index == is_seg_start_d[cur_seg_d[0]]){
                if(cur_seg_d[0] != 0 && is_seg_start_d[cur_seg_d[0]] == is_seg_end_d[cur_seg_d[0] - 1]){
                    e_num = size_is_d[index] - is_seg_end_edges_d[cur_seg_d[0] - 1];
                    start = is_seg_end_edges_d[cur_seg_d[0] - 1];
                }
            }else if(index == is_seg_end_d[cur_seg_d[0]]){
                start = 0;
                e_num = is_seg_end_edges_d[cur_seg_d[0]];
                if(e_num < size_is_d[index]){
                    last_modified_d[index] += 1;
                }
            }else{
                
            }
            if(values_d[index] > deltas_d[index]){
                // values_d[cur_modified_node] = deltas_d[cur_modified_node];
                atomicExch(&values_d[index], deltas_d[index]);
                for(unsigned int i = 0; i < e_num; i++){
                    unsigned int dist_node = isoffset_d[cur_isoff_d[index] - cur_seg_d[0] * is_average_edges_d[0] + start + i];
                    // printf("dist node is %d",dist_node);
                    // printf("cur off is %d",cur_isoff_d[index] - cur_seg_d[0] * is_average_edges_d[0] + start + i);
                    // if(cur_modified_node == 9){
                    //     printf("9 dist is %d",dist_node);
                    // }
                    // int new_dist = is_edata_d[i] + deltas_d[cur_modified_node];//权重图
                    int new_dist = 1 + deltas_d[index];//无权测试
                    if(new_dist < deltas_d[dist_node]){
                        atomicMin(&deltas_d[dist_node], new_dist);
                        atomicExch(&last_modified_d[dist_node], 1);
                        if(node_type_d[dist_node] == 3){
                            atomicExch(&last_modified_d[dist_node], 2);
                        }
                    }
                }
            }
        }
        // if(last_modified_d[index] < 0)
        // printf("last2 is %d,last1 is %d,last0 is %d\n",last_modified_d[index],temp,temp0);
        // printf("in3");
    }

    __global__
    void setCurSize(){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num && last_modified_d[index] > 0){
            atomicAdd(&cur_modified_size_d[0], 1);
            // printf("last is %d",last_modified_d[index]);
        }
    }


}