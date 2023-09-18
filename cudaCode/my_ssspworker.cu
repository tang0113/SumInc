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
    __device__ unsigned int *is_eparent_d;
    __device__ unsigned int *deltas_parent_d;

    __device__ int *deltas_d;
    __device__ int *values_d;

    // __device__ unsigned int *cur_modified_d;
    __device__ unsigned int *cur_modified_size_d;//长度为1,只记录当前修改队列的长度
    // __device__ unsigned int *next_modified_d;
    // __device__ unsigned int *next_modified_size_d;//长度为num,记录每个modified顶点产生的新的modified顶点数量,这里设为num的作用是使得每个线程能正确处理当前顶点产生的新modified
    __device__ unsigned int next_modified_allsize_d;
    __device__ unsigned int *is_modified_d;//记录顶点是否被修改
    __device__ unsigned int *last_modified_d;
    __device__ unsigned int num;

    __device__ char *node_type_d;

    // __device__ int *sem;
    __device__ int curpos = 0;
    
    void init(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, unsigned int *is_eparent_d, unsigned int *deltas_parent_d, 
              char *node_type_d){

                // int *sem_d;
                // cudaMalloc(&sem_d, sizeof(int) *1);

        init_real<<<1,1>>>(oeoffset_d, oe_edata_d, cur_oeoff_d, deltas_d, values_d, size_oe_d, FLAGS_sssp_source, 
                           cur_modified_size_d, is_modified_d, last_modified_d, num, 
                           iboffset_d, ib_edata_d, cur_iboff_d, size_ib_d, 
                           isoffset_d, is_edata_d, cur_isoff_d, size_is_d, is_eparent_d, deltas_parent_d, 
                           node_type_d);

    }

    __global__
    void init_real(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, unsigned int *is_eparent_d, unsigned int *deltas_parent_d, 
              char *node_type_d){

        tjnsssp::oeoffset_d = oeoffset_d;
        tjnsssp::iboffset_d = iboffset_d;
        tjnsssp::isoffset_d = isoffset_d;
        tjnsssp::is_eparent_d = is_eparent_d;
        tjnsssp::deltas_parent_d = deltas_parent_d;

        tjnsssp::cur_oeoff_d = cur_oeoff_d;
        tjnsssp::cur_iboff_d = cur_iboff_d;
        tjnsssp::cur_isoff_d = cur_isoff_d;

        tjnsssp::size_oe_d = size_oe_d;
        tjnsssp::size_ib_d = size_ib_d;
        tjnsssp::size_is_d = size_is_d;

        tjnsssp::oe_edata_d = oe_edata_d;
        tjnsssp::ib_edata_d = ib_edata_d;
        tjnsssp::is_edata_d = is_edata_d;

        tjnsssp::deltas_d = deltas_d;
        tjnsssp::values_d = values_d;

        tjnsssp::cur_modified_size_d = cur_modified_size_d;
        tjnsssp::cur_modified_size_d[0] = 1;

        // tjnsssp::cur_modified_d = cur_modified_d;
        // tjnsssp::cur_modified_d[0] = FLAGS_sssp_source;
        
        tjnsssp::num = num;
        tjnsssp::is_modified_d = is_modified_d;
        tjnsssp::last_modified_d = last_modified_d;
        // tjnsssp::last_modified_d[FLAGS_sssp_source] = 1;

        tjnsssp::node_type_d = node_type_d;
        

        // cudaFree(tjnsssp::next_modified_size);
    }

    void g_function(unsigned int *cur_modified_size_h, unsigned int num){
        dim3 block(512);
        dim3 grid((num - 1) / block.x + 1);

        // printf("cur modified size is %d", cur_modified_size_h[0]);
        g_function_real<<<grid, block>>>();
        

        cudaDeviceSynchronize();
        // unsigned int *next_modified_size_h = (unsigned int *)malloc(sizeof(unsigned int) * 1);
        // unsigned int *next_modified_size_d; cudaMalloc(&next_modified_size_d, sizeof(unsigned int) * 1);


        dim3 block1(512);
        dim3 grid1((num-1) / block1.x + 1);
        setNextSize<<<grid1, block1>>>();
        cudaDeviceSynchronize();

        swap(num);

        clear(num);
    }

    void g_function_compr(unsigned int *cur_modified_size_h, unsigned int cpr_num){
        dim3 block(512);
        dim3 grid((cpr_num - 1) / block.x + 1);
        g_function_compr_real<<<grid, block>>>();
        cudaDeviceSynchronize();

        dim3 block1(512);
        dim3 grid1((cpr_num-1) / block1.x + 1);
        setNextSize<<<grid1, block1>>>();
        cudaDeviceSynchronize();

        swap(cpr_num);

        clear(cpr_num);
        // swap();
    }

    __global__
    void g_function_real(){
        
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(index < num){
            sssp_Ingress(index);
        }
    }

    __global__
    void g_function_compr_real(){
        
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num){
            if(last_modified_d[index] == 0){
                return ;
            }else{
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

    
    void swap(unsigned int num){
        cudaEvent_t startCuda, stopCuda;  //declare
        cudaEventCreate(&startCuda);      //set up 
        cudaEventCreate(&stopCuda);       //set up
        cudaEventRecord(startCuda,0);    //start
        
        

        getAllSize<<<1, 1>>>();
        

        dim3 block(512);
        dim3 grid((num - 1) / block.x + 1);

        swap_real<<<grid, block>>>();

        cudaEventRecord(stopCuda,0);     //finish
        cudaEventSynchronize(stopCuda);
        cudaDeviceSynchronize();
        float eTime;
        cudaEventElapsedTime(&eTime, startCuda, stopCuda);  
        //eTime = stoptime - starttime
        printf("time is %f\n",eTime);

    }

    __global__
    void swap_real(){
        //分块策略,保留is_modified和last_modified长度为顶点总size,其余的都为当前块的顶点size
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num){
            last_modified_d[index] = 0;
            if(is_modified_d[index]){
                last_modified_d[index] = 1;
                // acquire_semaphore();
                // for (int i = 0; i < 32; i++) {
                //     // Check if it is this thread's turn
                //     if (index % 32 != i)
                //         continue;

                //     // Lock
                //     while (atomicExch(sem, 1) == 1);
                //     // Work
                //     cur_modified_d[curpos] = index;
                //     curpos++;
                //     // Unlock
                //     *sem = 0;
                // }
                // while(true){
                //     // printf("sem = %d",sem);
                //     if(atomicExch(sem,1) != 1){
                //         printf("sem = %d",*sem);
                //         cur_modified_d[curpos] = index;
                //         curpos++;
                //         printf("sem = %d",*sem);
                //         *sem = 0;printf("sem = %d",*sem);
                //         break;
                //     }
                // }
                
                // cur_modified_d[curpos] = index;
                // curpos++;
                // printf("111");
                // sem = 0;
                // __syncthreads();
            }
        }
        
        // if(index < next_modified_allsize_d){
        //     int number = index + 1;
        //     for(unsigned int i = 0; i < num; i++){
        //         if(is_modified_d[i]){
        //             number--;
        //         }
        //         if(number == 0){
        //             cur_modified_d[index] = i;
        //             break;
        //         }
        //     }
        //     cur_modified_size_d[0] = next_modified_allsize_d;
        // }
        // if(next_modified_allsize_d == 0 && index == 0){
        //     cur_modified_size_d[0] = 0;
        //     int maxid = 0;
        //     int maxvalue = 0;
        //     for(int i=0;i<num;i++){
        //         if(values_d[i] !=  2147483647 && values_d[i] > maxvalue){
        //             maxvalue = values_d[i];
        //             maxid = i;
        //         }
        //         // if(i == 266023 || i == 266024){
        //         //     printf("values is %d",values_d[i]);
        //         // }
        //     }
        //     printf("max values[%d] is %d",maxid,values_d[maxid]);
        //     printf("-------------------------------end-------------------------------");
        // }
    }

    void clear(unsigned int num){
        dim3 block(512);
        dim3 grid((num-1) / block.x + 1);
        clear_real<<<grid, block>>>();
    }

    __global__
    void clear_real(){

        int index = threadIdx.x + blockIdx.x * blockDim.x;
        

        if(index < num)
            is_modified_d[index] = 0;

        next_modified_allsize_d = 0;
        curpos = 0;

    }

    __global__
    void setNextSize(){
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        if(index < num && is_modified_d[index]){
            atomicAdd(&next_modified_allsize_d, 1);
        }
    }

    __global__
    void getAllSize(){
        cur_modified_size_d[0] = next_modified_allsize_d;
        printf("next size is %d",next_modified_allsize_d);
    }

    __device__ 
    void acquire_semaphore(){
        // int index = threadIdx.x + blockIdx.x * blockDim.x;
        // int temp = atomicCAS(&sem, 0, 1);
        // printf("temp hhh is %d",temp);
        // if(temp == 0){
        //     if(index == 2)printf("no");
        //     return ;
        // }
        // while (temp != 0){
        //     temp = atomicCAS(&sem, 0, 1);
        //     // printf("temp while is %d index is %d sem is %d\n",temp,index,sem);
        // }
        
    }

    __device__ 
    void release_sem(){
        // sem = 0;
        // printf("sem is 0");
    }

    __device__
    void sssp_Ingress(int index){
        if(last_modified_d[index] == 0){
            return ;
        }
        else{
            unsigned int cur_modified_node = index;
            if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
                values_d[cur_modified_node] = deltas_d[cur_modified_node];
                // atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
                for(unsigned int i = cur_oeoff_d[cur_modified_node]; i < size_oe_d[cur_modified_node] + cur_oeoff_d[cur_modified_node]; i++){
                    unsigned int dist_node = oeoffset_d[i];
                    // if(cur_modified_node == 9){
                    //     printf("9 dist is %d",dist_node);
                    // }
                    // int new_dist = oe_edata_d[i] + deltas_d[cur_modified_node];//权重图
                    int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                    if(new_dist < deltas_d[dist_node]){
                        atomicMin(&deltas_d[dist_node], new_dist);
                        atomicExch(&is_modified_d[dist_node], 1);
                    }
                }
            }
            __syncthreads();
            __threadfence();
        }
        
    }

    __device__
    void sssp_nodeTypeZeroAndOne(int index){
        unsigned int cur_modified_node = index;
        if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
            values_d[cur_modified_node] = deltas_d[cur_modified_node];
            // atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
            for(unsigned int i = cur_iboff_d[cur_modified_node]; i < size_ib_d[cur_modified_node] + cur_iboff_d[cur_modified_node]; i++){
                unsigned int dist_node = iboffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&is_modified_d[dist_node], 1);
                    unsigned int e_parent = cur_modified_node;
                    atomicExch(&deltas_parent_d[dist_node], e_parent);
                }
            }
        }
    }

    __device__
    void sssp_nodeTypeTwo(int index){
        unsigned int cur_modified_node = index;
        if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
            values_d[cur_modified_node] = deltas_d[cur_modified_node];
            // atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
            for(unsigned int i = cur_isoff_d[cur_modified_node]; i < size_is_d[cur_modified_node] + cur_isoff_d[cur_modified_node]; i++){
                unsigned int dist_node = isoffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = is_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&is_modified_d[dist_node], 1);
                    unsigned int e_parent = is_eparent_d[i];
                    atomicExch(&deltas_parent_d[dist_node], e_parent);
                }
            }
        }
    }

    __device__
    void sssp_nodeTypeThree(int index){
        unsigned int cur_modified_node = index;
        if(values_d[cur_modified_node] > deltas_d[cur_modified_node]){
            values_d[cur_modified_node] = deltas_d[cur_modified_node];
            // atomicExch(&values_d[cur_modified_node], deltas_d[cur_modified_node]);
            //第一阶段
            for(unsigned int i = cur_iboff_d[cur_modified_node]; i < size_ib_d[cur_modified_node] + cur_iboff_d[cur_modified_node]; i++){
                unsigned int dist_node = iboffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&is_modified_d[dist_node], 1);
                    unsigned int e_parent = cur_modified_node;
                    atomicExch(&deltas_parent_d[dist_node], e_parent);
                }
            }
            //第二阶段
            for(unsigned int i = cur_isoff_d[cur_modified_node]; i < size_is_d[cur_modified_node] + cur_isoff_d[cur_modified_node]; i++){
                unsigned int dist_node = isoffset_d[i];
                // if(cur_modified_node == 9){
                //     printf("9 dist is %d",dist_node);
                // }
                // int new_dist = ib_edata_d[i] + deltas_d[cur_modified_node];//权重图
                int new_dist = 1 + deltas_d[cur_modified_node];//无权测试
                if(new_dist < deltas_d[dist_node]){
                    atomicMin(&deltas_d[dist_node], new_dist);
                    atomicExch(&is_modified_d[dist_node], 1);
                    unsigned int e_parent = is_eparent_d[i];
                    atomicExch(&deltas_parent_d[dist_node], e_parent);
                }
            }
        }
    }

}