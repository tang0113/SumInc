#ifndef MY_SSSPWORKER_SEG_CUH
#define MY_SSSPWORKER_SEG_CUH
namespace tjnsssp_seg{
    void init(int *deltas_d, int *values_d, int FLAGS_sssp_source, unsigned int *deltas_parent_d, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, unsigned int *is_eparent_d, 
              char *node_type_d, unsigned int *cur_seg_d, unsigned int *seg_num_d, 
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, unsigned int *ib_average_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d, unsigned int *is_average_edges_d);
    
    __global__
    void init_real(int *deltas_d, int *values_d, int FLAGS_sssp_source, unsigned int *deltas_parent_d, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, unsigned int *is_eparent_d, 
              char *node_type_d, unsigned int *cur_seg_d, unsigned int *seg_num_d, 
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, unsigned int *ib_average_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d, unsigned int *is_average_edges_d);

    void g_function_compr(unsigned int cpr_num);

    __global__
    void g_function_compr_real();

    __global__
    void clearCurSize();

    __global__
    void setCurSize();

    __device__
    void sssp_nodeTypeZeroAndOne(int index);

    __device__
    void sssp_nodeTypeTwo(int index);

    __device__
    void sssp_nodeTypeThree(int index);

    __global__
    void setCurSeg();

    __global__
    void getMaxValue();
}
#endif