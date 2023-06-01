#ifndef MY_WORKER_CUH
#define MY_WORKER_CUH

namespace tjn{
    /**
     * @author tjn
     * @brief for sum_sync_iter_worker.h Query(),arg4 'size' is v_d.size.
    */
    void value2last(float *last_values_d,float *values_d,unsigned int *v_d,int size);

    /**
     * @author tjn
     * @brief call from value2last,which calls from sum_sync_iter_worker to compute last_values in GPU.
    */
    __global__
    void value2last_real(float *last_values_d,float *values_d,unsigned int *v_d,int size);

    
    void init(float *deltas_d, float *values_d, unsigned int *oeoffset_d, unsigned int *size_d, unsigned int start_d, unsigned int end_d, unsigned int *curOff_d, char *node_type_d);
    
    /**
     * @brief 初始化操作
    */
    __global__
    void init_real(float *deltas_d, float *values_d, unsigned int *oeoffset_d, unsigned int *size_d, unsigned int start_d, unsigned int end_d, unsigned int *curOff_d, char *node_type_d);


    /**
     * @author tjn
     * @brief computation part of pagerank
    */
    void g_function_pr(unsigned int start_d, unsigned int end_d);

    /**
     * @author tjn
     * @brief call from g_function_pr , computes in GPU.
    */
    __global__
    void g_function_pr_real();

    /**
     * @brief 压缩时候的pagerank，此时为sumInc，此函数调用运行在GPU上的global函数
    */
    void g_function_compr(unsigned int start_d, unsigned int end_d);

    /**
     * @brief 此函数为运行在GPU上的函数
    */
    __global__
    void g_function_compr_real();

    /**
     * @author tjn
     * @brief delta change
    */
    __device__
    bool isChange_pr(float delta, int verticesNum);
}
#endif