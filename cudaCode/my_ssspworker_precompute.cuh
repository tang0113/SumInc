#ifndef MY_SSSPWORKER_PRECOMPUTE_CUH
#define MY_SSSPWORKER_PRECOMPUTE_CUH
namespace tjnsssp_precompute{

    void init_subgraph(unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, int *subgraph_data_d, unsigned int *subgraph_neighbor_d, unsigned int old_node_num);
    
    __global__
    void init_subgraph_real(unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, int *subgraph_data_d, unsigned int *subgraph_neighbor_d, unsigned int old_node_num);

    void init_modified(unsigned int *modified_d, int *deltas_d, unsigned int *deltas_parent_d, int *values_d, unsigned int *modified_num_d);

    __global__
    void init_modified_real(unsigned int *modified_d, int *deltas_d, unsigned int *deltas_parent_d, int *values_d, unsigned int *modified_num_d);

    void compute(unsigned int num);

    __global__
    void compute_real(unsigned int num);

    __global__
    void count_active_num(unsigned int num);
}
#endif