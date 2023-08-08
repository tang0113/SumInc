#ifndef MY_SSSPWORKER_SEG_CUH
#define MY_SSSPWORKER_SEG_CUH
namespace tjnsssp_seg{
    void init(int *deltas_d, int *values_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d, 
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d);
    
    __global__
    void init_real(int *deltas_d, int *values_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d, 
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d);
}
#endif