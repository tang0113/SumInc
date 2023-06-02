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

    /**
     * @param spnode_datas_d:SumInc,所有超点的值
     * @param bound_node_values_d:SumInc,所有边界点的值
     * @param oeoffset_d:Ingress,所有顶点的邻接表合并成的一个链表
     * @param cur_oeoff_d:Ingress,所有顶点在oeoffset中的起始偏移量形成的链表
     * @param size_oe_d:Ingress,所有顶点的邻居size形成的链表 
     * @param node_type_d:SumInc,顶点类型
    */
    void init(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, unsigned int *oeoffset_d, unsigned int *size_oe_d, unsigned int start_d, unsigned int end_d, unsigned int *cur_oeoff_d, char *node_type_d);
    
    /**
     * @brief 初始化操作
    */
    __global__
    void init_real(float *spnode_datas_d, float *bound_node_values_d, float *deltas_d, float *values_d, unsigned int *oeoffset_d, unsigned int *size_oe_d, unsigned int start_d, unsigned int end_d, unsigned int *cur_oeoff_d, char *node_type_d);


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

    /**
    * @brief 顶点类型为SingleNode时，为Ingress时也使用此函数
    * @param index:表示当前线程的索引，也可理解为当前顶点的索引
    */
    __device__
    void pr_singleNode(int index);

    /**
     * @brief 顶点类型为OnlyInNode
    */
    __device__
    void pr_onlyInNode(int index);

    /**
     * @brief 顶点类型为OnlyInNode
     */
    __device__
    void pr_onlyOutNode(int index);

    __device__
    void pr_bothOutInNode(int index);

    __device__
    void pr_outMaster(int index);

    __device__
    void pr_bothOutInMaster(int index);
}
#endif