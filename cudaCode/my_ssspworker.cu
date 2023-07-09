#include <cuda_runtime.h>
#include <stdio.h>
#include "my_ssspworker.cuh"
// #include "freshman.h"
namespace tjnsssp{
    __device__ unsigned int *oeoffset_d;
    __device__ unsigned int *iboffset_d;
    __device__ unsigned int *isoffset_d;

    __device__ unsigned int *cur_oeoff_d;
    __device__ unsigned int *cur_iboff_d;
    __device__ unsigned int *cur_isoff_d;

    __device__ unsigned int *size_oe_d;
    __device__ unsigned int *size_ib_d;
    __device__ unsigned int *size_is_d;

    __device__ int *oe_edata_d;
    __device__ int *ib_edata_d;
    __device__ int *is_edata_d;

    __device__ int *deltas_d;
    __device__ int *values_d;

    __device__ unsigned int *cur_modified_d;
    __device__ unsigned int *cur_modified_size_d;//长度为1,只记录当前修改队列的长度
    __device__ unsigned int *next_modified_d;
    __device__ unsigned int *next_modified_size_d;//长度为num,记录每个modified顶点产生的新的modified顶点数量,这里设为num的作用是使得每个线程能正确处理当前顶点产生的新modified
    __device__ unsigned int next_modified_allsize_d;

    __device__ char *node_type_d;
    
    void init(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_d, unsigned int *next_modified_d, unsigned int *cur_modified_size_d, unsigned int *next_modified_size_d, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d){

        init_real<<<1,1>>>(oeoffset_d, oe_edata_d, cur_oeoff_d, deltas_d, values_d, size_oe_d, FLAGS_sssp_source, 
                           cur_modified_d, next_modified_d, cur_modified_size_d, next_modified_size_d, 
                           iboffset_d, ib_edata_d, cur_iboff_d, size_ib_d, 
                           isoffset_d, is_edata_d, cur_isoff_d, size_is_d, 
                           node_type_d);

    }

    __global__
    void init_real(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_d, unsigned int *next_modified_d, unsigned int *cur_modified_size_d, unsigned int *next_modified_size_d, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d){

        tjnsssp::oeoffset_d = oeoffset_d;
        tjnsssp::iboffset_d = iboffset_d;
        tjnsssp::isoffset_d = isoffset_d;

        tjnsssp::cur_oeoff_d = cur_oeoff_d;
        tjnsssp::cur_iboff_d = cur_iboff_d;
        tjnsssp::cur_isoff_d = cur_isoff_d;

        tjnsssp::size_oe_d = size_oe_d;
        tjnsssp::size_ib_d = size_ib_d;
        tjnsssp::size_is_d = size_is_d;

        tjnsssp::oe_edata_d = oe_edata_d;
        tjnsssp::ib_edata_d = ib_edata_d;

        tjnsssp::deltas_d = deltas_d;
        tjnsssp::values_d = values_d;

        tjnsssp::cur_modified_size_d = cur_modified_size_d;
        tjnsssp::cur_modified_size_d[0] = 1;

        tjnsssp::next_modified_size_d = next_modified_size_d;

        tjnsssp::cur_modified_d = cur_modified_d;
        tjnsssp::cur_modified_d[0] = FLAGS_sssp_source;
        
        tjnsssp::next_modified_d = next_modified_d;

        tjnsssp::node_type_d = node_type_d;

        // cudaFree(tjnsssp::next_modified_size);
    }

    void g_function(unsigned int *cur_modified_size_h){

        dim3 block(512);
        dim3 grid((cur_modified_size_h[0] - 1) / block.x + 1);

        // printf("cur modified size is %d", cur_modified_size_h[0]);
        g_function_real<<<grid, block>>>();
        cudaDeviceSynchronize();
        // unsigned int *next_modified_size_h = (unsigned int *)malloc(sizeof(unsigned int) * 1);
        // unsigned int *next_modified_size_d; cudaMalloc(&next_modified_size_d, sizeof(unsigned int) * 1);

        swap();
        // 
    }

    void g_function_compr(unsigned int *cur_modified_size_h){
        dim3 block(512);
        dim3 grid((cur_modified_size_h[0] - 1) / block.x + 1);
        g_function_compr_real<<<grid, block>>>();
        cudaDeviceSynchronize();
        swap();
    }

    __global__
    void g_function_real(){
        
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(index < cur_modified_size_d[0]){
            sssp_Ingress(index);
        }
    }

    __global__
    void g_function_compr_real(){

        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < cur_modified_size_d[0]){
            unsigned int cur_modified_node = cur_modified_d[index];
            switch(node_type_d[cur_modified_node]){
                case 0:
                    {
                        sssp_nodeTypeZeroAndOne(index);
                    }
                case 1:
                    {
                        sssp_nodeTypeZeroAndOne(index);
                    }
                case 2:
                    {
                        sssp_nodeTypeTwo(index);
                    }
                case 3:
                    {
                        sssp_nodeTypeThree(index);
                    }
            }
        }
    }

    
    void swap(){
        // dim3 block(512);
        // dim3 grid((next_modified_size_h[0] - 1) / block.x + 1);
        // swap_real<<<grid, block>>>();
        // cudaDeviceSynchronize();

        swap_real<<<1,1>>>();
        cudaDeviceSynchronize();
    }

    __global__
    void swap_real(){
        for(unsigned int i = 0; i < next_modified_allsize_d; i++){
            cur_modified_d[i] = next_modified_d[i];
        }
        for(unsigned int i = 0;i < cur_modified_size_d[0];i++){
            next_modified_size_d[i] = 0;
        }
        cur_modified_size_d[0] = next_modified_allsize_d;
        next_modified_allsize_d = 0;
        unsigned int temp = 0;
        int maxid = 0;
        if(cur_modified_size_d[0] == 0){
            for(int i=0;i<2400000;i++){
                if(values_d[i] < 2147483647){
                    temp = max(values_d[i], temp);
                    maxid = values_d[i] >= temp ? i : maxid;
                }
                if(i == 1539735){
                    printf("values1539735 is %u,",values_d[i]);
                }
                if(i == 2321775){
                    printf("values2321775 is %u,",values_d[i]);
                }
            }
            printf("values max is %u, i max is %d\n\n",temp,maxid);
        }
        
    }

    __device__
    void sssp_Ingress(int index){
        unsigned int cur_modified_node = cur_modified_d[index];
        if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
            // values_d[cur_modified_node] = deltas_d[cur_modified_node];
            atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
            for(unsigned int i = cur_oeoff_d[cur_modified_node]; i < size_oe_d[cur_modified_node] + cur_oeoff_d[cur_modified_node]; i++){
                unsigned int dist_node = oeoffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = oe_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicAdd(&next_modified_size_d[index], 1);
                    atomicAdd(&next_modified_allsize_d, 1);
                }
            }
            __syncthreads();
            __threadfence();
            unsigned int cur_pos = 0;
            unsigned int offset = 0;
            for(int i=0;i<index;i++){
                offset += next_modified_size_d[i];
            }
            __syncthreads();
            __threadfence();
            for(unsigned int i = cur_oeoff_d[cur_modified_node]; i < size_oe_d[cur_modified_node] + cur_oeoff_d[cur_modified_node]; i++){
                unsigned int dist_node = oeoffset_d[i];
                // int new_dist = oe_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&next_modified_d[offset + cur_pos], dist_node);
                    cur_pos++;
                    
                }
            }
            __syncthreads();
            __threadfence();
        }
        /**
         *  以下写法是错误的
         * 假如有3个线程,执行完a代码行之后同时执行b代码行,那么相当于对全局内存依次修改(原子操作),全部执行b操作之后再执行c
         * 而c行的目的是执行完b之后马上执行自增,这显然是不能办到的
        */
        // if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
        //     // values_d[cur_modified_node] = deltas_d[cur_modified_node];
        //     atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
        //     for(unsigned int i = cur_oeoff_d[cur_modified_node]; i < size_oe_d[cur_modified_node] + cur_oeoff_d[cur_modified_node]; i++){
        //         unsigned int dist_node = oeoffset_d[i];
        //         // int new_dist = oe_edata_d[i] + deltas_d[cur_modified_node];//权重图
        //         int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
        //         if(new_dist < deltas_d[dist_node]){
        //             atomicMin(&deltas_d[dist_node], new_dist);//a
        //             atomicExch(&next_modified_d[0], dist_node);//b
        //             atomicAdd(&next_modified_d[0],1);//c
        //         }
        //     }
        // }
        __syncthreads();
        __threadfence();
    }

    __device__
    void sssp_nodeTypeZeroAndOne(int index){
        unsigned int cur_modified_node = cur_modified_d[index];
        if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
            // values_d[cur_modified_node] = deltas_d[cur_modified_node];
            atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
            for(unsigned int i = cur_iboff_d[cur_modified_node]; i < size_ib_d[cur_modified_node] + cur_iboff_d[cur_modified_node]; i++){
                unsigned int dist_node = iboffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicAdd(&next_modified_size_d[index], 1);
                    atomicAdd(&next_modified_allsize_d, 1);
                }
            }
            __syncthreads();
            __threadfence();
            unsigned int cur_pos = 0;
            unsigned int offset = 0;
            for(int i=0;i<index;i++){
                offset += next_modified_size_d[i];
            }
            __syncthreads();
            __threadfence();
            for(unsigned int i = cur_iboff_d[cur_modified_node]; i < size_ib_d[cur_modified_node] + cur_iboff_d[cur_modified_node]; i++){
                unsigned int dist_node = iboffset_d[i];
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&next_modified_d[offset + cur_pos], dist_node);
                    cur_pos++;
                    
                }
            }
            __syncthreads();
            __threadfence();
        }
        /**
         *  以下写法是错误的
         * 假如有3个线程,执行完a代码行之后同时执行b代码行,那么相当于对全局内存依次修改(原子操作),全部执行b操作之后再执行c
         * 而c行的目的是执行完b之后马上执行自增,这显然是不能办到的
        */
        // if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
        //     // values_d[cur_modified_node] = deltas_d[cur_modified_node];
        //     atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
        //     for(unsigned int i = cur_oeoff_d[cur_modified_node]; i < size_oe_d[cur_modified_node] + cur_oeoff_d[cur_modified_node]; i++){
        //         unsigned int dist_node = oeoffset_d[i];
        //         // int new_dist = oe_edata_d[i] + deltas_d[cur_modified_node];//权重图
        //         int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
        //         if(new_dist < deltas_d[dist_node]){
        //             atomicMin(&deltas_d[dist_node], new_dist);//a
        //             atomicExch(&next_modified_d[0], dist_node);//b
        //             atomicAdd(&next_modified_d[0],1);//c
        //         }
        //     }
        // }
        __syncthreads();
        __threadfence();
    }

    __device__
    void sssp_nodeTypeTwo(int index){
        unsigned int cur_modified_node = cur_modified_d[index];
        if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
            // values_d[cur_modified_node] = deltas_d[cur_modified_node];
            atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
            for(unsigned int i = cur_isoff_d[cur_modified_node]; i < size_is_d[cur_modified_node] + cur_isoff_d[cur_modified_node]; i++){
                unsigned int dist_node = isoffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = is_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicAdd(&next_modified_size_d[index], 1);
                    atomicAdd(&next_modified_allsize_d, 1);
                }
            }
            __syncthreads();
            __threadfence();
            unsigned int cur_pos = 0;
            unsigned int offset = 0;
            for(int i=0;i<index;i++){
                offset += next_modified_size_d[i];
            }
            __syncthreads();
            __threadfence();
            for(unsigned int i = cur_isoff_d[cur_modified_node]; i < size_is_d[cur_modified_node] + cur_isoff_d[cur_modified_node]; i++){
                unsigned int dist_node = isoffset_d[i];
                // int new_dist = is_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&next_modified_d[offset + cur_pos], dist_node);
                    cur_pos++;
                    
                }
            }
            __syncthreads();
            __threadfence();
        }
        /**
         *  以下写法是错误的
         * 假如有3个线程,执行完a代码行之后同时执行b代码行,那么相当于对全局内存依次修改(原子操作),全部执行b操作之后再执行c
         * 而c行的目的是执行完b之后马上执行自增,这显然是不能办到的
        */
        // if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
        //     // values_d[cur_modified_node] = deltas_d[cur_modified_node];
        //     atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
        //     for(unsigned int i = cur_oeoff_d[cur_modified_node]; i < size_oe_d[cur_modified_node] + cur_oeoff_d[cur_modified_node]; i++){
        //         unsigned int dist_node = oeoffset_d[i];
        //         // int new_dist = oe_edata_d[i] + deltas_d[cur_modified_node];//权重图
        //         int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
        //         if(new_dist < deltas_d[dist_node]){
        //             atomicMin(&deltas_d[dist_node], new_dist);//a
        //             atomicExch(&next_modified_d[0], dist_node);//b
        //             atomicAdd(&next_modified_d[0],1);//c
        //         }
        //     }
        // }
        __syncthreads();
        __threadfence();
    }

    __device__
    void sssp_nodeTypeThree(int index){
        unsigned int cur_modified_node = cur_modified_d[index];
        if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
            // values_d[cur_modified_node] = deltas_d[cur_modified_node];
            atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
            //第一阶段
            for(unsigned int i = cur_iboff_d[cur_modified_node]; i < size_ib_d[cur_modified_node] + cur_iboff_d[cur_modified_node]; i++){
                unsigned int dist_node = iboffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicAdd(&next_modified_size_d[index], 1);
                    atomicAdd(&next_modified_allsize_d, 1);
                }
            }
            __syncthreads();
            __threadfence();
            unsigned int cur_pos = 0;
            unsigned int offset = 0;
            for(int i=0;i<index;i++){
                offset += next_modified_size_d[i];
            }
            __syncthreads();
            __threadfence();
            for(unsigned int i = cur_iboff_d[cur_modified_node]; i < size_ib_d[cur_modified_node] + cur_iboff_d[cur_modified_node]; i++){
                unsigned int dist_node = iboffset_d[i];
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&next_modified_d[offset + cur_pos], dist_node);
                    cur_pos++;
                    
                }
            }
            atomicExch(&next_modified_size_d[index], 0);
            unsigned int allstart = next_modified_allsize_d;
            __syncthreads();
            __threadfence();
            //第二阶段
            for(unsigned int i = cur_isoff_d[cur_modified_node]; i < size_is_d[cur_modified_node] + cur_isoff_d[cur_modified_node]; i++){
                unsigned int dist_node = isoffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicAdd(&next_modified_size_d[index], 1);
                    atomicAdd(&next_modified_allsize_d, 1);
                }
            }
            __syncthreads();
            __threadfence();
            offset = 0;
            cur_pos = 0;
            for(int i=0;i<index;i++){
                offset += next_modified_size_d[i];
            }
            __syncthreads();
            __threadfence();
            for(unsigned int i = cur_isoff_d[cur_modified_node]; i < size_is_d[cur_modified_node] + cur_isoff_d[cur_modified_node]; i++){
                unsigned int dist_node = isoffset_d[i];
                // int new_dist = is_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&next_modified_d[allstart + offset + cur_pos], dist_node);
                    cur_pos++;
                }
            }
            __syncthreads();
            __threadfence();
        }
        __syncthreads();
        __threadfence();
    }

}