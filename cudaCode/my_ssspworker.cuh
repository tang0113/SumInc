#ifndef MY_SSSPWORKER_CUH
#define MY_SSSPWORKER_CUH
namespace tjnsssp{

    /**
     * @brief 调用init_real来进行初始化操作
    */
    
    void init(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, unsigned int *is_eparent_d, unsigned int *deltas_parent_d, 
              char *node_type_d);

    __global__
    void init_real(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d, int FLAGS_sssp_source, 
                   unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
                   unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
                   unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, unsigned int *is_eparent_d, unsigned int *deltas_parent_d, 
                   char *node_type_d);

    /**
     * @brief 调用g_function_real来实现sssp的过程,Ingress
    */
    void g_function(unsigned int *cur_modified_size_h, unsigned int num);

    __global__
    void g_function_real();

    void g_function_compr(unsigned int *cur_modified_size_h, unsigned int cpr_num);

    __global__
    void g_function_compr_real();

    /**
     * @brief 根据ismodified计算下一次要修改的顶点数量
    */
    __global__
    void setNextSize();

    /**
     * @brief 获取下一次要修改顶点数量
    */
    __global__
    void getAllSize();

    void swap(unsigned int num);

    __global__
    void swap_real();

    void clear(unsigned int num);

    __global__
    void clear_real();

    __device__
    void sssp_Ingress(int index);

    __device__
    void sssp_nodeTypeZeroAndOne(int index);

    __device__
    void sssp_nodeTypeTwo(int index);

    __device__
    void sssp_nodeTypeThree(int index);

    __device__ 
    void acquire_semaphore();

    __device__ 
    void release_sem();
}
#endif