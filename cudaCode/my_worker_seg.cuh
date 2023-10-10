#ifndef MY_WORKER_SEG_CUH
#define MY_WORKER_SEG_CUH
namespace tjnpr_seg{
    void init(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, 
              unsigned int *oeoffset_d, unsigned int *iboffset_d, unsigned int *isoffset_d, unsigned int *syncoffset_d, 
              unsigned int *size_oe_d, unsigned int *size_ib_d, unsigned int *size_is_d, unsigned int *size_sync_d, 
              unsigned int num, unsigned int oe_average_nodes, unsigned int is_average_nodes, unsigned int cur_seg, unsigned int seg_num, 
              unsigned int *cur_oeoff_d, unsigned int *cur_iboff_d, unsigned int *cur_isoff_d, unsigned int *cur_syncoff_d, 
              char *node_type_d, float *is_edata_d, 
              unsigned int *all_out_mirror_d, unsigned int *mirrorid2vid_d, 
              unsigned int *oe_edges_d, unsigned int *ib_edges_d, unsigned int *is_edges_d, unsigned int *sync_edges_d);

    __global__
    void init_real(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, 
              unsigned int *oeoffset_d, unsigned int *iboffset_d, unsigned int *isoffset_d, unsigned int *syncoffset_d, 
              unsigned int *size_oe_d, unsigned int *size_ib_d, unsigned int *size_is_d, unsigned int *size_sync_d, 
              unsigned int num, unsigned int oe_average_nodes, unsigned int is_average_nodes, unsigned int cur_seg, unsigned int seg_num, 
              unsigned int *cur_oeoff_d, unsigned int *cur_iboff_d, unsigned int *cur_isoff_d, unsigned int *cur_syncoff_d, 
              char *node_type_d, float *is_edata_d, 
              unsigned int *all_out_mirror_d, unsigned int *mirrorid2vid_d, 
              unsigned int *oe_edges_d, unsigned int *ib_edges_d, unsigned int *is_edges_d, unsigned int *sync_edges_d);

    void g_function_pr(unsigned int num);
    void g_function_compr(unsigned int num, unsigned int *isoffset_all_d, float *isdata_all_d);
    void OutMirrorSyncToMaster(unsigned int size);
    float deltaSum(unsigned int start, unsigned int end);
    __global__
    void deltaSum_real(float *result);
    __device__
    bool isChange_pr(float delta, int verticesNum);
    __device__
    void pr_Ingress(int index);
    __device__
    void pr_singleNode(int index);
    __device__
    void pr_onlyInNode(int index);
    __device__
    void pr_onlyOutNode(int index);
    __device__
    void pr_bothOutInNode(int index);
    __device__
    void pr_outMaster(int index, unsigned int *isoffset_all_d, float *isdata_all_d);
    __device__
    void pr_bothOutInMaster(int index, unsigned int *isoffset_all_d, float *isdata_all_d);
}
#endif