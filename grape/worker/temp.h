
#include <grape/fragment/loader.h>

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>

#include "flags.h"
#include "grape/app/traversal_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/parallel_engine.h"
#include "grape/parallel/parallel_message_manager.h"
#include "timer.h"
#include "grape/fragment/trav_compressor.h"
#include <cuda_runtime.h>
#include "freshman.h"
#include "my_ssspworker.cuh"

if(FLAGS_segment){
    vid_t num = fragment_->InnerVertices().size();

    vid_t *size_ib_d, *size_ib_h = (vid_t *)malloc(sizeof(vid_t) * num);//SumInc,node type:SingleNode
    vid_t *size_is_d, *size_is_h = (vid_t *)malloc(sizeof(vid_t) * num);//SumInc,node type:OnlyInNode

    vid_t *cur_iboff_d, *cur_iboff_h = (vid_t *)malloc(sizeof(vid_t) * num);
    vid_t *cur_isoff_d, *cur_isoff_h = (vid_t *)malloc(sizeof(vid_t) * num);

    vid_t *cur_modified_size_d;
    vid_t *cur_modified_size_h = (vid_t *)malloc(sizeof(vid_t) * 1);
    vid_t *is_modified_d;//判断当前顶点是否被修改
    vid_t *last_modified_d;

    unsigned int ib_offsize = 0;
    for(int i = 0;i < num;i++){//SumInc
        cur_iboff_h[i] = ib_offsize;
        ib_offsize += ib_e_offset_[i+1] - ib_e_offset_[i];
        size_ib_h[i] = ib_e_offset_[i+1] - ib_e_offset_[i];
    }
    
    // LOG(INFO) << "ibsize is "<<ib_e_offset_.size()<<" num is "<<num<<"delta size is"<<app_->deltas_.size();
    unsigned int is_offsize = 0;
    for(int i=0;i < num;i++){
        cur_isoff_h[i] = is_offsize;
        is_offsize += is_e_offset_[i+1] - is_e_offset_[i];
        size_is_h[i] = is_e_offset_[i+1] - is_e_offset_[i];
    }

    auto &values = app_->values_;
    auto &deltas = app_->deltas_;

    value_t *deltas_d, *deltas_h = (value_t *)malloc(sizeof(value_t) * cpr_->all_node_num);
    value_t *values_d;

    char *node_type_d, *node_type_h = (char *)malloc(sizeof(char) * num);//SumInc,记录每个顶点的类型

    vid_t ib_average_edges = (ib_offsize + FLAGS_seg_num - 1) / FLAGS_seg_num;
    vid_t is_average_edges = (is_offsize + FLAGS_seg_num - 1) / FLAGS_seg_num;

    //二维数组,分段赋给gpu
    vid_t *iboffset_d, **iboffset_h = (vid_t **)malloc(sizeof(vid_t *) * FLAGS_seg_num);//SumInc
    vid_t *isoffset_d, **isoffset_h = (vid_t **)malloc(sizeof(vid_t *) * FLAGS_seg_num);//SumInc

    for(int i=0;i<FLAGS_seg_num;i++){
        iboffset_h[i] = (vid_t *)malloc(sizeof(vid_t) * ib_average_edges);
        isoffset_h[i] = (vid_t *)malloc(sizeof(vid_t) * is_average_edges);
    }
    
    value_t *ib_edata_d, **ib_edata_h = (value_t **)malloc(sizeof(value_t) * FLAGS_seg_num);
    value_t *is_edata_d, **is_edata_h = (value_t **)malloc(sizeof(value_t) * FLAGS_seg_num);

    for(int i=0;i<FLAGS_seg_num;i++){
        ib_edata_h[i] = (vid_t *)malloc(sizeof(vid_t) * ib_average_edges);
        is_edata_h[i] = (vid_t *)malloc(sizeof(vid_t) * is_average_edges);
    }

    vid_t *ib_seg_start = (vid_t *)malloc(sizeof(vid_t) * FLAGS_seg_num);//起始顶点
    vid_t *ib_seg_end = (vid_t *)malloc(sizeof(vid_t) * FLAGS_seg_num);//结束顶点
    vid_t *ib_seg_end_edges = (vid_t *)malloc(sizeof(vid_t) * FLAGS_seg_num);//最后一个顶点分得的边数量
    vid_t *is_seg_start = (vid_t *)malloc(sizeof(vid_t) * FLAGS_seg_num);
    vid_t *is_seg_end = (vid_t *)malloc(sizeof(vid_t) * FLAGS_seg_num);
    vid_t *is_seg_end_edges = (vid_t *)malloc(sizeof(vid_t) * FLAGS_seg_num);

    vid_t ib_edge_num = 0, is_edge_num = 0;//记录分段时的边,判断是否超出平均
    vid_t ib_cur_node = 0, is_cur_node = 0;//记录分段时的点遍历到何处
    for(int i=0;i<FLAGS_seg_num;i++){
        ib_seg_start[i] = ib_cur_node;
        is_seg_start[i] = is_cur_node;
        if(i!=0){
            if(ib_seg_end_edges[i-1] < size_ib_h[ib_seg_end[i-1]]){
                ib_edge_num += size_ib_h[ib_seg_end[i-1]] - ib_seg_end_edges[i-1];
                if(ib_edge_num < ib_average_edges){
                    ib_cur_node++;
                }
            }
            if(is_seg_end_edges[i-1] < size_is_h[is_seg_end[i-1]]){
                is_edge_num += size_is_h[is_seg_end[i-1]] - is_seg_end_edges[i-1];
                if(is_edge_num < is_average_edges){
                    is_cur_node++;
                }
            }
        }
        //边的分配
        while(ib_edge_num < ib_average_edges && ib_cur_node < num){
            ib_edge_num += size_ib_h[ib_cur_node];
            if(ib_edge_num >= ib_average_edges){
                break;
            }
            ib_cur_node++;
            if(ib_cur_node == num){
                break;
            }
        }
        while(is_edge_num < is_average_edges && is_cur_node < num){
            is_edge_num += size_is_h[is_cur_node];
            if(is_edge_num >= is_average_edges){
                break;
            }
            is_cur_node++;
            if(is_cur_node == num){
                break;
            }
        }

        //确定结束顶点
        if(ib_cur_node == num){
            ib_seg_end[i] = num - 1;
        }else{
            ib_seg_end[i] = ib_cur_node;
        }
        if(is_cur_node == num){
            is_seg_end[i] = num - 1;
        }else{
            is_seg_end[i] = is_cur_node;
        }

        //确定结束顶点分配的边数量
        if(ib_edge_num > ib_average_edges){
            ib_seg_end_edges[i] = size_ib_h[ib_seg_end[i]] - ib_edge_num + ib_average_edges;
        }else{
            ib_seg_end_edges[i] = size_ib_h[ib_seg_end[i]];
            ib_cur_node++;
        }
        if(is_edge_num > is_average_edges){
            is_seg_end_edges[i] = size_is_h[is_seg_end[i]] - is_edge_num + is_average_edges;
        }else{
            is_seg_end_edges[i] = size_is_h[is_seg_end[i]];
            is_cur_node++;
        }

        ib_edge_num = 0;
        is_edge_num = 0;
    }

    cudaSetDevice(0);
    cudaMalloc(&deltas_d, sizeof(value_t) * cpr_->all_node_num);
    cudaMalloc(&values_d, sizeof(value_t) * cpr_->all_node_num);
    check();

    //邻居大列表,所有点的邻接表拼接而成
    cudaMalloc(&iboffset_d, sizeof(vid_t) * ib_average_edges);
    cudaMalloc(&isoffset_d, sizeof(vid_t) * is_average_edges);

    //边数据
    cudaMalloc(&ib_edata_d, sizeof(value_t) * ib_average_edges);
    cudaMalloc(&is_edata_d, sizeof(value_t) * is_average_edges);

    //记录每个点的邻接表在其邻居大列表中的起始位置
    cudaMalloc(&cur_iboff_d, sizeof(vid_t) * num);
    cudaMalloc(&cur_isoff_d, sizeof(vid_t) * num);
    check();

    //记录每个点的邻居数量
    cudaMalloc(&size_ib_d, sizeof(vid_t) * num);
    cudaMalloc(&size_is_d, sizeof(vid_t) * num);
    check();
    //顶点类型
    cudaMalloc(&node_type_d, sizeof(char) * num);   
    //当前要修改的点数量
    cudaMalloc(&cur_modified_size_d, sizeof(vid_t) * 1);
    //下一次每个顶点要加入修改的目的顶点数量,设置为num目的是使用GPU时防止多个线程对全局变量同时进行修改
    cudaMalloc(&is_modified_d, sizeof(vid_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    cudaMalloc(&last_modified_d, sizeof(vid_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    check();

    for(int i=0;i<FLAGS_seg_num;i++){
        unsigned int ib_curIndex = 0, is_curIndex = 0;
        for(int j=ib_seg_start[i];j<ib_seg_end[i];j++){
            //分段之后有两种可能,一种是顶点所指向的边完全被分割,另一种是两段都有同一个点的邻居消息.
            if(j == ib_seg_start[i] && i!=0 && ib_seg_start[i] == ib_seg_end[i-1]){
                for(int k=ib_seg_end_edges[i-1]; k<size_ib_h[ib_seg_end[i-1]]; k++){
                    value_t* temp = reinterpret_cast<value_t*>(&ib_e_offset_[j][k].data);//强制转换,原类型为empty不能直接用
                    ib_edata_h[i][ib_curIndex] = *temp;
                    ib_offset_h[i][ib_curIndex++] = ib_e_offset_[i][j].neighbor.GetValue();
                }
            }else{
                for(int k=0;k<size_ib_h[j];k++){
                    value_t* temp = reinterpret_cast<value_t*>(&ib_e_offset_[j][k].data);//强制转换,原类型为empty不能直接用
                    ib_edata_h[i][ib_curIndex] = *temp;
                    ib_offset_h[i][ib_curIndex++] = ib_e_offset_[i][j].neighbor.GetValue();
                }
            }
        }
        //处理最后一个顶点,在极端情况下可能会出错,如一个顶点邻居数量特别多足以分两端,
        //此时k不应从0开始,且之前分段时设置的ib_seg_end_edges[i]也是错误的.
        //或者分段数量很多,导致一个顶点的边能分成两段,此时也易出错.
        for(int k=0;k<ib_seg_end_edges[i];k++){
            value_t* temp = reinterpret_cast<value_t*>(&ib_e_offset_[ib_seg_end[i]][k].data);//强制转换,原类型为empty不能直接用
            ib_edata_h[i][ib_curIndex] = *temp;
            ib_offset_h[i][ib_curIndex++] = ib_e_offset_[i][j].neighbor.GetValue();
        }

        for(int j=is_seg_start[i];j<is_seg_end[i];j++){
            //分段之后有两种可能,一种是顶点所指向的边完全被分割,另一种是两段都有同一个点的邻居消息.
            if(j == is_seg_start[i] && i!=0 && is_seg_start[i] == is_seg_end[i-1]){
                for(int k=is_seg_end_edges[i-1]; k<size_is_h[is_seg_end[i-1]]; k++){
                    value_t* temp = reinterpret_cast<value_t*>(&is_e_offset_[j][k].data);//强制转换,原类型为empty不能直接用
                    is_edata_h[i][is_curIndex] = *temp;
                    is_offset_h[i][is_curIndex++] = is_e_offset_[i][j].neighbor.GetValue();
                }
            }else{
                for(int k=0;k<size_is_h[j];k++){
                    value_t* temp = reinterpret_cast<value_t*>(&is_e_offset_[j][k].data);//强制转换,原类型为empty不能直接用
                    is_edata_h[i][is_curIndex] = *temp;
                    is_offset_h[i][is_curIndex++] = is_e_offset_[i][j].neighbor.GetValue();
                }
            }
        }
        
        for(int k=0;k<is_seg_end_edges[i];k++){
            value_t* temp = reinterpret_cast<value_t*>(&is_e_offset_[is_seg_end[i]][k].data);//强制转换,原类型为empty不能直接用
            is_edata_h[i][is_curIndex] = *temp;
            is_offset_h[i][is_curIndex++] = is_e_offset_[i][j].neighbor.GetValue();
        }
        
    }

    values.fake2buffer();
    deltas.fake2buffer();
    for(int i=0;i<num;i++){
        deltas_h[i] = deltas.data_buffer[i].value;
    }

    cudaMemcpy(cur_iboff_d, cur_iboff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_isoff_d, cur_isoff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);

    cudaMemcpy(deltas_d, deltas_h, sizeof(value_t) * (FLAGS_compress ? cpr_->all_node_num : num), cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * (FLAGS_compress ? cpr_->all_node_num : num), cudaMemcpyHostToDevice);

    cudaMemcpy(size_ib_d, size_ib_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(size_is_d, size_is_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    check();

    unsigned int sssp_source = FLAGS_sssp_source;
    if(!app_->curr_modified_.Exist(vertex_t(sssp_source))){
        for(int i = 0;i<(FLAGS_compress ? cpr_->all_node_num : num);i++){
          // vertex_t u(sssp_source+i);
          // vertex_t v(sssp_source-i);
          vertex_t temp(i);
          if(app_->curr_modified_.Exist(temp)){
            LOG(INFO) << "sssp source is "<< temp.GetValue();
            sssp_source = i;
            break;
          }
        }
    }

    vid_t *ib_seg_start_d, *ib_seg_end_d, *ib_seg_end_edges_d;
    vid_t *is_seg_start_d, *is_seg_end_d, *is_seg_end_edges_d;
    cudaMalloc(&ib_seg_start_d,sizeof(vid_t)*FLAGS_seg_num);
    cudaMalloc(&ib_seg_end_d,sizeof(vid_t)*FLAGS_seg_num);
    cudaMalloc(&ib_seg_end_edges_d,sizeof(vid_t)*FLAGS_seg_num);
    
    cudaMalloc(&is_seg_start_d,sizeof(vid_t)*FLAGS_seg_num);
    cudaMalloc(&is_seg_end_d,sizeof(vid_t)*FLAGS_seg_num);
    cudaMalloc(&is_seg_end_edges_d,sizeof(vid_t)*FLAGS_seg_num);

    cudaMemcpy(ib_seg_start_d, ib_seg_start, sizeof(vid_t) * FLAGS_seg_num, cudaMemcpyHostToDevice);
    cudaMemcpy(ib_seg_end_d, ib_seg_end, sizeof(vid_t) * FLAGS_seg_num, cudaMemcpyHostToDevice);
    cudaMemcpy(ib_seg_end_edges_d, ib_seg_end_edges, sizeof(vid_t) * FLAGS_seg_num, cudaMemcpyHostToDevice);
    
    cudaMemcpy(is_seg_start_d, is_seg_start, sizeof(vid_t) * FLAGS_seg_num, cudaMemcpyHostToDevice);
    cudaMemcpy(is_seg_end_d, is_seg_end, sizeof(vid_t) * FLAGS_seg_num, cudaMemcpyHostToDevice);
    cudaMemcpy(is_seg_end_edges_d, is_seg_end_edges, sizeof(vid_t) * FLAGS_seg_num, cudaMemcpyHostToDevice);
    
    tjnsssp_seg::init(deltas_d, values_d, sssp_source, 
                    cur_modified_size_d, is_modified_d, last_modified_d, (FLAGS_compress ? cpr_->all_node_num : num),
                    iboffset_d, ib_edata_d, cur_iboff_d, size_ib_d, 
                    isoffset_d, is_edata_d, cur_isoff_d, size_is_d, 
                    node_type_d, 
                    ib_seg_start_d, ib_seg_end_d, ib_seg_end_edges_d, 
                    is_seg_start_d, is_seg_end_d, is_seg_end_edges_d);
    double whileTime = 0;
    whileTime = GetCurrentTime();
    int cur_seg = 0;
    while(true){
        cur_seg = cur_seg % FLAGS_seg_num;
        LOG(INFO) << "step=" << step << " curr_modified_.size()=" << app_->curr_modified_.Count();
        exec_time -= GetCurrentTime();
        ++step;

        auto inner_vertices = fragment_->InnerVertices();
        auto outer_vertices = fragment_->OuterVertices();

        messages_.StartARound();
        // clean_bitset_time -= GetCurrentTime();
        // app_->next_modified_.ParallelClear(thread_num()); // 对于压缩图清理的范围可以缩小， 直接初始化为小区间！！！！
        app_->next_modified_.ParallelClear(thread_num()); // 对于压缩图清理的范围可以缩小， 直接初始化为小区间！！！！
        // clean_bitset_time += GetCurrentTime();
        {
          messages_.ParallelProcess<fragment_t, DependencyData<vid_t, value_t>>(
              thread_num(), *fragment_,
              [this](int tid, vertex_t v,
                    const DependencyData<vid_t, value_t>& msg) {
                if (app_->AccumulateToAtomic(v, msg)) {
                  app_->curr_modified_.Insert(v); // 换成小的bitset好像可能会报错
                }
              });
        }

        if (FLAGS_cilk) {
          if(compr_stage == false){
            // ForEachCilk(
            if(!FLAGS_gpu_start){
              
            }

            if(FLAGS_gpu_start){
            //   cudaMemcpy(cur_modified_size_h, cur_modified_size_d, sizeof(vid_t) * 1, cudaMemcpyDeviceToHost);
            //   // check();
            //   tjnsssp_seg::g_function(cur_modified_size_h, num);
            //   // check();
            //   cudaMemcpy(cur_modified_size_h, cur_modified_size_d, sizeof(vid_t) * 1, cudaMemcpyDeviceToHost);
            }
            
          }
          if (compr_stage) {
            if(!FLAGS_gpu_start){
              
            }
            if(FLAGS_gpu_start){
            //   cudaMemcpy(cur_modified_size_h, cur_modified_size_d, sizeof(vid_t) * 1, cudaMemcpyDeviceToHost);
              cudaMemcpy(iboffset_d, iboffset_h[cur_seg], sizeof(vid_t) * ib_average_edges, cudaMemcpyHostToDevice);
              cudaMemcpy(isoffset_d, isoffset_h[cur_seg], sizeof(vid_t) * is_average_edges, cudaMemcpyHostToDevice);
              cudaMemcpy(ib_edata_d, ib_edata_h[cur_seg], sizeof(vid_t) * ib_average_edges, cudaMemcpyHostToDevice);
              cudaMemcpy(is_edata_d, is_edata_h[cur_seg], sizeof(vid_t) * is_average_edges, cudaMemcpyHostToDevice);

              tjnsssp_seg::g_function_compr(cur_modified_size_h, cpr_->all_node_num);
              cudaMemcpy(cur_modified_size_h, cur_modified_size_d, sizeof(vid_t) * 1, cudaMemcpyDeviceToHost);
              cur_seg++;
            }
            
          }
        }

        auto& channels = messages_.Channels();
        ForEach(app_->next_modified_, outer_vertices,
                [&channels, vm_ptr, fid, this](int tid, vertex_t v) {
                  auto& delta_to_send = app_->deltas_[v];

                  if (delta_to_send.value != app_->GetIdentityElement()) {
                    vid_t& v_parent_gid = delta_to_send.parent_gid;
                    fid_t v_fid = vm_ptr->GetFidFromGid(v_parent_gid);
                    if (v_fid == fid) {
                      v_parent_gid = newGid2oldGid[v_parent_gid];
                    }
                    channels[tid].SyncStateOnOuterVertex(*fragment_, v,
                                                        delta_to_send);
                  }
                });
        if (!app_->next_modified_.PartialEmpty(0, fragment_->GetInnerVerticesNum())) {
          messages_.ForceContinue();
        }
         VLOG(1) << "[Worker " << comm_spec_.worker_id()
                << "]: Finished IterateKernel - " << step;
        messages_.FinishARound();

        // app_->next_modified_.Swap(app_->curr_modified_);

        exec_time += GetCurrentTime();
        bool terminate = messages_.ToTerminate();
        if ( (terminate && !FLAGS_gpu_start) || (!cur_modified_size_h[0] && FLAGS_gpu_start) || step > 100 * FLAGS_seg_num) {
            if(compr_stage){
            LOG(INFO) << "start correct...";
            // check_result("correct before");
            timer_next("correct deviation");
            print_active_edge("#globalCompt");
            compr_stage = false;
            corr_time -= GetCurrentTime();

            // supernode send by inner_delta
            LOG(INFO) << "cpr_->supernodes_num=" << cpr_->supernodes_num;
            double send_time = GetCurrentTime();
            vertex_t source;
            bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
            // #pragma cilk grainsize = 16
            parallel_for(vid_t j = 0; j < cpr_->supernodes_num; j++){
              supernode_t &spnode = cpr_->supernodes[j];
              auto u = spnode.id;
              bool is_mirror = false;
              if (u.GetValue() >= cpr_->old_node_num) {
                is_mirror = true;
                u = cpr_->mirrorid2vid[u];
              }
              auto& value = app_->values_[u];
              if (value != app_->GetIdentityElement()) { // right, 这是其实不能判断是否是被更新的！老的其实不用发！
                auto& delta = app_->deltas_[u];
                vid_t spid = cpr_->id2spids[u];
                vertex_t p;
                fragment_->Gid2Vertex(delta.parent_gid, p);
                // 下面的if优化，为了调试暂时关闭！！！！最后需要打开！！！
                if (is_mirror == true || spid != cpr_->id2spids[p] || (native_source && source == p)) { // Only nodes that depend on external nodes need to send
                  auto& oes = spnode.inner_delta;
                  app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_); // 没有完全重编号，所以只能用原来的bitset
                }
              }
            }
            LOG(INFO) << "  send_time: " << (GetCurrentTime() - send_time);
            

            // check_result("corr before");
            // print_result("校正后:");
            corr_time += GetCurrentTime();
            LOG(INFO) << "#first iter step: " << step;
            LOG(INFO) << "#first exec_time: " << exec_time;
            LOG(INFO) << "#corr_time: " << corr_time;
            print_active_edge("#localAss");
            // print_result();
            app_->next_modified_.Swap(app_->curr_modified_); // 校正完应该收敛，无须交换吧！！！
            // continue;  // Unnecessary!!!
            // break;
          }
          if (batch_stage) {
            batch_stage = false;

            if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
              // print_result();
              LOG(INFO) << "#iter step: " << step;
              LOG(INFO) << "#Batch time: " << exec_time;
              LOG(INFO) << "#for_time: " << for_time;
              LOG(INFO) << "#clean_bitset_time: " << clean_bitset_time;
              print_active_edge("#Batch");
              for_time = 0;
            }
            exec_time = 0;
            corr_time = 0;
            step = 1;

            if (!FLAGS_efile_update.empty()) {
              LOG(INFO) << "-------------------------------------------------------------------";
              LOG(INFO) << "--------------------------INC COMPUTE------------------------------";
              LOG(INFO) << "-------------------------------------------------------------------";
              // FLAGS_compress = false; // 测试
              compr_stage = FLAGS_compress; // use supernode
              timer_next("reloadGraph");
              deltaCompute();  // reload graph
              // compr_stage = false; // 测试
              LOG(INFO) << "\n-----load graph finish, app_->next_modified_.size=" << app_->next_modified_.ParallelCount(8);
              timer_next("inc algorithm");

              // 新版本重排序
              if (compr_stage == true) {
                // app_->next_modified_.Swap(app_->curr_modified_);
                first_step(values_temp, deltas_temp, exec_time, true);
              }
              continue; // 已经将活跃点放入curr_modified_中了..

            } else {
              LOG(ERROR) << "Missing efile_update or efile_updated";
              break;
            }
          } else {
            if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
              LOG(INFO) << "#Inc iter step: " << step;
              LOG(INFO) << "#Inc time: " << exec_time << " sec";
              LOG(INFO) << "#for_time_inc: " << for_time;
              print_active_edge("#curr");
              // print_result();
              for_time = 0;

              LOG(INFO) << "step=" << step << " f_send_delta_num=" << app_->f_send_delta_num;
              LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
              app_->f_send_delta_num = 0;
              app_->node_update_num = 0;
              app_->touch_nodes.ParallelClear(8);
            }
            break;
          }
        }

        app_->next_modified_.Swap(app_->curr_modified_); // 针对Ingress做动态时, 用这个 
    }
    whileTime = GetCurrentTime() - whileTime;
}