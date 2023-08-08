#include <cuda_runtime.h>
#include <stdio.h>
#include "my_ssspworker_seg.cuh"
// #include "freshman.h"
namespace tjnsssp_seg{

    __device__ unsigned int *iboffset_d;
    __device__ unsigned int *isoffset_d;

    __device__ unsigned int *cur_iboff_d;
    __device__ unsigned int *cur_isoff_d;

    __device__ unsigned int *size_ib_d;
    __device__ unsigned int *size_is_d;

    __device__ int *ib_edata_d;
    __device__ int *is_edata_d;

    __device__ int *deltas_d;
    __device__ int *values_d;

    __device__ unsigned int *cur_modified_size_d;//长度为1,只记录当前修改队列的长度
    __device__ unsigned int next_modified_allsize_d;
    __device__ unsigned int *is_modified_d;//记录顶点是否被修改
    __device__ unsigned int *last_modified_d;
    __device__ unsigned int num;

    __device__ char *node_type_d;

    __device__ unsigned int *ib_seg_start_d;
    __device__ unsigned int *is_seg_start_d;

    __device__ unsigned int *ib_seg_end_d;
    __device__ unsigned int *is_seg_end_d;

    __device__ unsigned int *ib_seg_end_edges_d;
    __device__ unsigned int *is_seg_end_edges_d;


    void init(int *deltas_d, int *values_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d, 
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d){

        init_real<<<1,1>>>(deltas_d, values_d, FLAGS_sssp_source, 
                   cur_modified_size_d, is_modified_d, last_modified_d, num, 
                   iboffset_d, ib_edata_d, cur_iboff_d, size_ib_d, 
                   isoffset_d, is_edata_d, cur_isoff_d, size_is_d, 
                   node_type_d, 
                   ib_seg_start_d, ib_seg_end_d, ib_seg_end_edges_d, 
                   is_seg_start_d, is_seg_end_d, is_seg_end_edges_d);

    }

    __global__
    void init_real(int *deltas_d, int *values_d, int FLAGS_sssp_source, 
              unsigned int *cur_modified_size_d, unsigned int *is_modified_d, unsigned int *last_modified_d, unsigned int num, 
              unsigned int *iboffset_d, int *ib_edata_d, unsigned int *cur_iboff_d, unsigned int *size_ib_d, 
              unsigned int *isoffset_d, int *is_edata_d, unsigned int *cur_isoff_d, unsigned int *size_is_d, 
              char *node_type_d, 
              unsigned int *ib_seg_start_d, unsigned int *ib_seg_end_d, unsigned int *ib_seg_end_edges_d, 
              unsigned int *is_seg_start_d, unsigned int *is_seg_end_d, unsigned int *is_seg_end_edges_d){
        
        tjnsssp_seg::iboffset_d = iboffset_d;
        tjnsssp_seg::isoffset_d = isoffset_d;

        tjnsssp_seg::ib_edata_d = ib_edata_d;
        tjnsssp_seg::is_edata_d = is_edata_d;

        tjnsssp_seg::cur_iboff_d = cur_iboff_d;
        tjnsssp_seg::cur_isoff_d = cur_isoff_d;

        tjnsssp_seg::size_ib_d = size_ib_d;
        tjnsssp_seg::size_is_d = size_is_d;

        tjnsssp_seg::deltas_d = deltas_d;
        tjnsssp_seg::values_d = values_d;

        tjnsssp_seg::cur_modified_size_d = cur_modified_size_d;
        tjnsssp_seg::cur_modified_size_d[0] = 1;

        tjnsssp_seg::num = num;
        tjnsssp_seg::is_modified_d = is_modified_d;
        tjnsssp_seg::last_modified_d = last_modified_d;
        tjnsssp_seg::last_modified_d[FLAGS_sssp_source] = 1;

        tjnsssp_seg::node_type_d = node_type_d;

        tjnsssp_seg::ib_seg_start_d = ib_seg_start_d;
        tjnsssp_seg::is_seg_start_d = is_seg_start_d;

        tjnsssp_seg::ib_seg_end_d = ib_seg_end_d;
        tjnsssp_seg::is_seg_end_d = is_seg_end_d;

        tjnsssp_seg::ib_seg_end_edges_d = ib_seg_end_edges_d;
        tjnsssp_seg::is_seg_end_edges_d = is_seg_end_edges_d;

    }


}