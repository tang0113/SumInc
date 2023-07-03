#ifndef MY_SSSPWORKER_CUH
#define MY_SSSPWORKER_CUH
namespace tjnsssp{

    /**
     * @brief 调用init_real来进行初始化操作
    */
    void init(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d);

    __global__
    void init_real(unsigned int *oeoffset_d, int *oe_edata_d, unsigned int *cur_oeoff_d, int *deltas_d, int *values_d, unsigned int *size_oe_d);

    /**
     * @brief 调用g_function_real来实现sssp的过程
    */
    void g_function();

    __global__
    void g_function_real();

    void swap();

    __global__
    void swap_real();
}
#endif