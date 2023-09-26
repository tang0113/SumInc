#ifndef MY_WORKER_PRECOMPUTE_CUH
#define MY_WORKER_PRECOMPUTE_CUH
namespace tjnpr_precompute{
    void init_subgraph(float *deltas_d, float *values_d ,unsigned int *size_oes_d, unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, unsigned int *subgraph_neighbor_d, unsigned int num, bool *is_active_d);
    
    __global__
    void init_subgraph_real(float *deltas_d, float *values_d ,unsigned int *size_oes_d, unsigned int *size_subgraph_d, unsigned int *cur_subgraph_d, unsigned int *subgraph_neighbor_d, unsigned int num, bool *is_active_d);
    
    int compute(unsigned int num, double threshold);

    __global__
    void compute_real();
}
#endif