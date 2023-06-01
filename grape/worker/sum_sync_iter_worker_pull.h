#ifndef GRAPE_WORKER_SUM_SYNC_ITER_WORKER_PULL_H_
#define GRAPE_WORKER_SUM_SYNC_ITER_WORKER_PULL_H_

#include <grape/fragment/loader.h>

#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "flags.h"
#include "grape/app/ingress_app_base.h"
#include "grape/communication/communicator.h"
#include "grape/communication/sync_comm.h"
#include "grape/fragment/inc_fragment_builder.h"
#include "grape/graph/adj_list.h"
#include "grape/parallel/default_message_manager.h"
#include "grape/parallel/parallel.h"
#include "grape/parallel/parallel_engine.h"
#include "timer.h"
#include "grape/fragment/iter_compressor.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class IterateKernel;

/**
 * @brief A Worker manages the computation cycle. DefaultWorker is a kind of
 * worker for apps derived from AppBase.
 *
 * @tparam APP_T
 */
template <typename APP_T>
class SumSyncIterWorkerPull : public ParallelEngine {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "SumSyncIterWorkerPull should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = ParallelMessageManager;
  using vid_t = typename APP_T::vid_t;
  using supernode_t = grape::SuperNodeForIter<vertex_t, value_t, vid_t>;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  using nbr_t = typename fragment_t::nbr_t;
  using nbr_index_t = Nbr<vid_t, value_t>;
  using adj_list_index_t = AdjList<vid_t, value_t>;

  SumSyncIterWorkerPull(std::shared_ptr<APP_T> app,
                        std::shared_ptr<fragment_t>& graph)
      : app_(app), graph_(graph) {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    graph_->PrepareToRunApp(APP_T::message_strategy, APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
    messages_.Init(comm_spec_.comm());
    messages_.InitChannels(thread_num());
    communicator_.InitCommunicator(comm_spec.comm());
    terminate_checking_time_ = 0;

    InitParallelEngine(pe_spec);
    LOG(INFO) << "Thread num: " << thread_num();

    app_->Init(comm_spec_, *graph_, false);
    app_->iterate_begin(*graph_);

    // init compressor
    if(FLAGS_compress){
      cpr_ = new IterCompressor<APP_T, supernode_t>(app_, graph_);
      cpr_->init(comm_spec_, communicator_, pe_spec);
      cpr_->run();
      timer_next("statistic");
      cpr_->statistic();
    }
  }

  /**
   * 通过采样确定阈值来筛选数据
   * sample_size: 采样大小
   * return 阈值
   */
   value_t Scheduled(int sample_size) {
  //    auto& priority = app_->priority_;
     vid_t all_size = graph_->GetInnerVerticesNum();
     if (all_size <= sample_size) {
       return 0;
     } else {
       std::unordered_set<int> id_set;
       // random number generator
       std::mt19937 gen(time(0));
       std::uniform_int_distribution<> dis(0, all_size - 1);  // 给定范围 // 构造符合要求的随机数生成器
       // sample random pos, the sample reflect the whole data set more or less 
       std::vector<value_t> sample; 
       int i;
       for (i = 0; i < sample_size; i++) {
         int rand_pos = dis(gen);
         while (id_set.find(rand_pos) != id_set.end()) {
           rand_pos = dis(gen);
         }
         id_set.insert(rand_pos);
         vertex_t u(rand_pos);
         value_t pri;
         app_->priority(pri, app_->values_[u], app_->deltas_[u]);
         sample.emplace_back(fabs(pri));
       }
  
       sort(sample.begin(), sample.end());
       int cut_index = sample_size * (1 - FLAGS_portion);  // 选择阈值位置 
       return sample[cut_index];
     }
   }

  /**
   * 用于图变换前后值的修正
   * type: -1表示回收(在旧图上)， 1表示重发(在新图上)
   *
   */
  void AmendValue(int type) {
    MPI_Barrier(comm_spec_.comm());

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    {
      messages_.StartARound();
      ForEach(inner_vertices, [this, type, &values](int tid, vertex_t u) {
        auto& value = values[u];
        auto delta = type * value;  // 发送回收值/发送补发值
        auto oes = graph_->GetOutgoingAdjList(u);

        app_->g_function(*graph_, u, value, delta, oes);
      });

      auto& channels = messages_.Channels();

      ForEach(outer_vertices, [this, &deltas, &channels](int tid, vertex_t v) {
        auto& delta_to_send = deltas[v];

        if (delta_to_send != app_->default_v()) {
          channels[tid].template SyncStateOnOuterVertex<fragment_t, value_t>(
              *graph_, v, delta_to_send);
          delta_to_send = app_->default_v();
        }
      });
      messages_.FinishARound();

      messages_.StartARound();
      messages_.template ParallelProcess<fragment_t, value_t>(
          thread_num(), *graph_,
          [this](int tid, vertex_t v, value_t received_delta) {
            app_->accumulate_atomic(app_->deltas_[v], received_delta);
          });
      // default_work,同步一轮
      messages_.FinishARound();
    }
    MPI_Barrier(comm_spec_.comm());  // 同步
  }

  /**
   * 用于重新加载图，并完成图变换前后值的校正
   *
   */
  void reloadGraph() {
    // recycled value on the old graph
    AmendValue(-1);

    VertexArray<value_t, vid_t> values, deltas;
    auto iv = graph_->InnerVertices();
    {
      // Backup values on old graph
      values.Init(iv);
      deltas.Init(iv);

      for (auto v : iv) {
        values[v] = app_->values_[v];
        deltas[v] = app_->deltas_[v];
      }
    }

    IncFragmentBuilder<fragment_t> inc_fragment_builder(graph_);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Parsing update file";
    }
    inc_fragment_builder.Init(FLAGS_efile_update);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Building new graph" << " old_graph_edgeNum=" << (graph_->GetEdgeNum()/2);
    }

    auto deleted_edges = inc_fragment_builder.GetDeletedEdgesGid();
    auto added_edges = inc_fragment_builder.GetAddedEdgesGid();
    LOG(INFO) << "deleted_edges_num=" << deleted_edges.size() << " added_edges_num=" << added_edges.size();
    
    // graph_ = inc_fragment_builder.Build();
    app_->rebuild_graph(*graph_);
    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    app_->iterate_begin(*graph_);
    app_->iterate_begin(*new_graph);

    if(FLAGS_compress){
      cpr_->inc_run(deleted_edges, added_edges, new_graph);
    }
    graph_ = new_graph;

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "New graph loaded" << " new_graph_edgeNum=" << (graph_->GetEdgeNum()/2);
    }
    app_->Init(comm_spec_, *graph_, false);
    {
      // Copy values to new graph
      for (auto v : iv) {
        app_->values_[v] = values[v];
        app_->deltas_[v] = deltas[v];
      }
    }
    // app_->iterate_begin(*graph_);
    // reissue value on the new graph
    AmendValue(1);

    if(FLAGS_compress){
      cpr_->inc_precompute_supernode();
    }
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());
    // app_->Init(comm_spec_, *graph_, false);

    if (FLAGS_debug) {
      volatile int i = 0;
      char hostname[256];
      gethostname(hostname, sizeof(hostname));
      printf("PID %d on %s ready for attach\n", getpid(), hostname);
      fflush(stdout);
      while (0 == i) {
        sleep(1);
      }
    }

    LOG(INFO) << "compr_stage=" << FLAGS_compress;

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    auto& index_values = app_->index_values_;
    VertexArray<value_t, vid_t> bound_deltas;
    //    auto& prioritys = app_->priority_;
    VertexArray<value_t, vid_t> last_values;


    int step = 1;
    bool batch_stage = true;
    bool compr_stage = FLAGS_compress; // true: supernode send


    if(compr_stage){
      #ifdef DEBUG
        value_t init_value_sum = 0;
        value_t init_delta_sum = 0;
        for(auto v : graph_->Vertices()){
          init_delta_sum += deltas[v];
          init_value_sum += values[v];
        }
        LOG(INFO) << "init_value_sum=" << init_value_sum << " init_delta_sum=" << init_delta_sum;
      #endif

      /* vertex classification */
      double node_type_time = GetCurrentTime();
      vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
      node_type.clear();
      node_type.resize(inner_node_num, std::numeric_limits<char>::max());
      bound_node_values.clear();
      bound_node_values.Init(inner_vertices, app_->default_v());
      all_nodes.resize(5);
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
          if(cpr_->Fc[u] == cpr_->FC_default_value){
            node_type[i] = 0; // out node
          }
          else if(cpr_->Fc[u] >= 0){
            node_type[i] = 2; // only source node
          }
          else if(!cpr_->supernode_out_bound[i]){
            node_type[i] = 4; // inner node
          }
          if(cpr_->supernode_out_bound[i]){
            node_type[i] = 1; // only bound node
            if(cpr_->Fc[u] >= 0){
              node_type[i] = 3; // source node + bound node
            }
          }
      }
      for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
          all_nodes[node_type[i]].emplace_back(vertex_t(i));
      }
      LOG(INFO) << "node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

      // debug
      {
        vid_t all_node_size = 0;
        for(int i = 0; i <= 4; i++){
          LOG(INFO) << "----node_type=" << i << " size=" << all_nodes[i].size();
          all_node_size += all_nodes[i].size();
        }
        LOG(INFO) << "all_node_size=" << all_node_size << " inner_node_num=" << inner_node_num;
      }

      double  transfer_csr_time = GetCurrentTime();

      /* get in-adj list */
      std::vector<std::vector<nbr_index_t> > iindex_delta_vec(inner_node_num); // all delta index
      std::vector<std::vector<nbr_index_t> > iindex_value_vec(inner_node_num); // all value index

      vid_t index_num1 = 0;
      vid_t index_num2 = 0;
      for(vid_t i = 0; i < cpr_->supernodes_num; i++) {
        supernode_t &spnode = cpr_->supernodes[i];
        vertex_t v(spnode.id);
        for(auto oe : spnode.bound_delta){
          iindex_value_vec[oe.first.GetValue()].emplace_back(nbr_index_t(v, oe.second));
          index_num1++;
        }
        for(auto oe : spnode.inner_value){
          iindex_value_vec[oe.first.GetValue()].emplace_back(nbr_index_t(v, oe.second));
          index_num1++;
        }
        for(auto oe : spnode.inner_delta){
          iindex_delta_vec[oe.first.GetValue()].emplace_back(nbr_index_t(v, oe.second));
          index_num2++;
        }
      }
      LOG(INFO) << "index_value_num=" << index_num1 << " index_delta_num=" << index_num2;

      double  init_time_1 = GetCurrentTime();
      spnode_datas.Init(inner_vertices, 0);
      is_ie_.clear();
      is_ie_offset_.clear();
      iindex_delta_.clear();
      iindex_delta_offset_.clear();
      iindex_value_.clear();
      iindex_value_offset_.clear();
      size_t source_ie_num = 0;
      size_t iindex_delta_num = 0;
      size_t iindex_value_num = 0;
      std::vector<size_t> is_ie_degree(inner_node_num+1, 0);
      std::vector<size_t> iindex_delta_degree(inner_node_num+1, 0);
      std::vector<size_t> iindex_value_degree(inner_node_num+1, 0);
      LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        char type = node_type[i];
        // if(type == 1 || type == 4){
        if(type != 0){ // not out node
          iindex_delta_degree[i+1] = iindex_delta_vec[i].size();
          iindex_value_degree[i+1] = iindex_value_vec[i].size();
          atomic_add(iindex_delta_num, iindex_delta_vec[i].size());
          atomic_add(iindex_value_num, iindex_value_vec[i].size());
        }
        if(type == 2 || type == 3){ // source + bound
          auto ies = graph_->GetIncomingAdjList(u);
          auto it = ies.begin();
          auto in_degree = ies.Size();
          vid_t ids_id = cpr_->id2spids[u];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < in_degree; j++){
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              // bound_e_num += 1;
              temp_cnt += 1;
            }
          }
          is_ie_degree[i+1] += temp_cnt;
          atomic_add(source_ie_num, temp_cnt);
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987
      LOG(INFO) << "iindex_delta_num=" << iindex_delta_num;
      LOG(INFO) << "iindex_value_num=" << iindex_value_num;
      LOG(INFO) << "source_ie_num=" << source_ie_num;

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i < inner_node_num; i++) {
        is_ie_degree[i] += is_ie_degree[i-1];
        iindex_delta_degree[i] += iindex_delta_degree[i-1];
        iindex_value_degree[i] += iindex_value_degree[i-1];
      }
      LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

      double init_time_2 = GetCurrentTime();
      iindex_delta_.resize(iindex_delta_num);
      iindex_value_.resize(iindex_value_num);
      is_ie_.resize(source_ie_num);
      iindex_delta_offset_.resize(inner_node_num+1);
      iindex_value_offset_.resize(inner_node_num+1);
      is_ie_offset_.resize(inner_node_num+1);
      LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      double csr_time_2 = GetCurrentTime();
      parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
      // for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* delta/value index */
        vid_t index_delta = iindex_delta_degree[i];
        vid_t index_value = iindex_value_degree[i];
        iindex_delta_offset_[i] = &iindex_delta_[index_delta];
        iindex_value_offset_[i] = &iindex_value_[index_value];
        char type = node_type[i];
        if(type != 0){
          for(auto& ie : iindex_delta_vec[i]){
            iindex_delta_[index_delta] = ie;
            index_delta++;
          }
          for(auto& ie : iindex_value_vec[i]){
            iindex_value_[index_value] = ie;
            index_value++;
          } 
        }
        /* in-edge of source node and s_b node */
        vid_t index_s = is_ie_degree[i];
        is_ie_offset_[i] = &is_ie_[index_s];
        if(type == 2 || type == 3){
          auto ies = graph_->GetIncomingAdjList(u);
          auto it = ies.begin();
          auto in_degree = ies.Size();
          vid_t ids_id = cpr_->id2spids[u];
          for(vid_t j = 0; j < in_degree; j++){
            auto& ie = *(it + j);
            if(ids_id != cpr_->id2spids[ie.neighbor]){
              is_ie_[index_s] = ie;
              index_s++;
            }
          }
        }
      }
      iindex_delta_offset_[inner_node_num] = &iindex_delta_[iindex_delta_num-1] + 1;
      iindex_value_offset_[inner_node_num] = &iindex_value_[iindex_value_num-1] + 1;
      is_ie_offset_[inner_node_num] = &is_ie_[source_ie_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      // just deal with bound + source nodes
      const std::vector<vertex_t>& nodes_3 = all_nodes[3];
      vid_t node_3_size = nodes_3.size();
      vid_t ie_3_size = 0;
      vid_t max_ie_3 = 0;
      for(vid_t i = 0; i < node_3_size; i++){
        vertex_t u = nodes_3[i];
        adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
        ie_3_size += ies.Size();
        max_ie_3 = std::max(int(max_ie_3), int(ies.Size()));
      }
      LOG(INFO) << "---max_ie_3=" << max_ie_3;
      is_ie_3.resize(ie_3_size);
      is_ie_offset_3.resize(node_3_size+1);
      vid_t is_ie_3_index = 0;
      for(vid_t i = 0; i < node_3_size; i++){
        vertex_t u = nodes_3[i];
        is_ie_offset_3[i] = &is_ie_3[is_ie_3_index];
        adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
        for(auto ie : ies){
          is_ie_3[is_ie_3_index++] = ie;
        }
      }
      is_ie_offset_3[node_3_size] = &is_ie_3[ie_3_size-1] + 1;
      CHECK_EQ(ie_3_size, is_ie_3_index);
      LOG(INFO) << "---ie_3_size=" << ie_3_size;
      //debug
      {
        const std::vector<vertex_t>& nodes_0 = all_nodes[0];
        vid_t node_0_size = nodes_0.size();
        vid_t max_ie_0 = 0;
        for(vid_t i = 0; i < node_0_size; i++){
          vertex_t u = nodes_0[i];
          auto ies = graph_->GetIncomingAdjList(u);
          max_ie_0 = std::max(int(max_ie_0), int(ies.Size()));
        }
        LOG(INFO) << "---max_ie_0=" << max_ie_0;
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149
    }

    double exec_time = 0;
    double one_step_time = 0;
    value_t pri = 0;

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    app_->g_num = 0;

// debug
// #define DEBUG

    /* save bound delta */
    bound_deltas.Init(inner_vertices);
    last_values.Init(inner_vertices);

    parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
      vertex_t v(i);
      auto& value = values[v];
      auto& delta = deltas[v];
      last_values[v] = value;
      if(compr_stage){
        if(node_type[i] == 0){
          app_->accumulate(value, delta); // 增量阶段可能也需要先聚合一次
        }
        else if(node_type[i] == 1 || node_type[i] == 4){
          bound_deltas[v] = delta; // 对于bound node copy一份后应该清空
          delta = app_->default_v();
        }
        else if(node_type[i] == 2 || node_type[i] == 3){
          app_->accumulate(index_values[v], delta);
          app_->accumulate(spnode_datas[v], delta);
          delta = app_->default_v();
        }
      }
      else{
        app_->accumulate(value, delta);
      }
    }

    #ifdef DEBUG
      double time_sum_0 = 0;
      double time_sum_1 = 0;
      double time_sum_2 = 0;
      double time_sum_3 = 0;
      double time_sum_4 = 0;
    #endif

    while (true) {
      exec_time -= GetCurrentTime();
      #ifdef DEBUG
        one_step_time = GetCurrentTime();
        double exchange_delta_time = 0;
        double send_time_0 = 0;
        double send_time_1 = 0;
        double send_time_2 = 0;
        double send_time_3 = 0;
        double send_time_4 = 0;
        size_t n_edge = 0;
      #endif

      messages_.StartARound();
      auto& channels = messages_.Channels();

      {
        #ifdef DEBUG
          auto begin = GetCurrentTime();
        #endif
        messages_.ParallelProcess<fragment_t, value_t>(
            thread_num(), *graph_,
            [this](int tid, vertex_t v, value_t received_delta) {
              app_->accumulate_atomic(app_->deltas_[v], received_delta);
            });
        #ifdef DEBUG
          VLOG(1) << "Process time: " << GetCurrentTime() - begin;
        #endif
        // LOG(INFO) << "Process time: " << GetCurrentTime() - begin;
      }

      {
        #ifdef DEBUG
          auto begin = GetCurrentTime();
        #endif
        // long long last_f = (app_->f_send_num + app_->f_send_value_num + app_->f_send_delta_num);
        if (FLAGS_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
          if(compr_stage == false){
            // copy delta to last delta
            #ifdef DEBUG
              exchange_delta_time -= GetCurrentTime();
            #endif
            app_->get_last_delta(*graph_);
            #ifdef DEBUG
              exchange_delta_time += GetCurrentTime();
            #endif

            #ifdef DEBUG
              send_time_0 -= GetCurrentTime();
              vid_t last_n_edge = n_edge;
            #endif
            //#pragma cilk grainsize = 1
            parallel_for(vid_t i = inner_vertices.begin().GetValue();
                      i < inner_vertices.end().GetValue(); i++) {
              vertex_t u(i);
              auto& value = values[u];
              auto& delta = deltas[u];
              auto ies = graph_->GetIncomingAdjList(u);
              app_->g_function_pull(*graph_, u, value, delta, ies); // pull
              app_->accumulate(value, delta);
              #ifdef DEBUG
                //n_edge += ies.Size();
                //atomic_add(n_edge, ies.Size());
              #endif
            }
            #ifdef DEBUG
              send_time_0 += GetCurrentTime();
              time_sum_0 += send_time_0;
              LOG(INFO) << "---time0/edges=" << (send_time_0/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge);
              // std::cout << "N edge: " << n_edge << std::endl;
            #endif
          }

          if(0){
            // copy delta to last delta
            // LOG(INFO) << "copy last delta..";
            #ifdef DEBUG
              exchange_delta_time -= GetCurrentTime();
            #endif
            app_->get_last_delta(*graph_, all_nodes);
            #ifdef DEBUG
              exchange_delta_time += GetCurrentTime();
            #endif

            /* 0. out node */
            const std::vector<vertex_t>& nodes_0 = all_nodes[0];
            vid_t node_0_size = nodes_0.size();
            #ifdef DEBUG
              send_time_0 -= GetCurrentTime();
            #endif
            parallel_for(vid_t i = 0; i < node_0_size; i++){
              vertex_t u = nodes_0[i];
              auto& value = values[u];
              auto& delta = deltas[u];
              auto ies = graph_->GetIncomingAdjList(u);
              app_->g_function_pull(*graph_, u, value, delta, ies); // pull
              app_->accumulate(value, delta);
              #ifdef DEBUG
                //n_edge += ies.Size();
              #endif
            }
            #ifdef DEBUG
              send_time_0 += GetCurrentTime();
            #endif

            /* 1. only bound node */
            const std::vector<vertex_t>& nodes_1 = all_nodes[1];
            vid_t node_1_size = nodes_1.size();
            #ifdef DEBUG
              send_time_1 -= GetCurrentTime();
            #endif
            parallel_for(vid_t i = 0; i < node_1_size; i++){
              vertex_t u = nodes_1[i];
              auto& value = values[u];
              auto& delta = deltas[u];
              // auto ies = graph_->GetIncomingAdjList(u);
              adj_list_index_t iindexes = adj_list_index_t(iindex_value_offset_[u.GetValue()], iindex_value_offset_[u.GetValue()+1]);
              app_->g_function_pull_by_index(*graph_, u, value, delta, iindexes); // pull
              app_->accumulate(value, delta);
              #ifdef DEBUG
                //n_edge += iindexes.Size();
              #endif
            }
            #ifdef DEBUG
              send_time_1 += GetCurrentTime();
            #endif

            /* 2. only source node */
            const std::vector<vertex_t>& nodes_2 = all_nodes[2];
            vid_t node_2_size = nodes_2.size();
            #ifdef DEBUG
              send_time_2 -= GetCurrentTime();
            #endif
            parallel_for(vid_t i = 0; i < node_2_size; i++){
              vertex_t u = nodes_2[i];
              auto& value = values[u];
              // auto& delta = deltas[u];
              adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
              value_t outv = app_->default_v();
              app_->g_function_pull(*graph_, u, value, outv, ies); // pull
              app_->accumulate(index_values[u], outv);
              app_->accumulate(spnode_datas[u], outv);
              #ifdef DEBUG
                //n_edge += ies.Size();
              #endif
            }
            #ifdef DEBUG
              send_time_2 += GetCurrentTime();
            #endif

            /* 3. source+bound node */
            const std::vector<vertex_t>& nodes_3 = all_nodes[3];
            vid_t node_3_size = nodes_3.size();
            #ifdef DEBUG
              send_time_3 -= GetCurrentTime();
            #endif
            parallel_for(vid_t i = 0; i < node_3_size; i++){
              vertex_t u = nodes_3[i];
              auto& value = values[u];
              auto& delta = deltas[u];
              // 3.1 as source
              adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
              value_t outv = app_->default_v();
              app_->g_function_pull(*graph_, u, value, outv, ies); // pull delta
              app_->accumulate(index_values[u], outv);
              app_->accumulate(spnode_datas[u], outv);
              // 3.2 as bound
              adj_list_index_t iindexes = adj_list_index_t(iindex_value_offset_[u.GetValue()], iindex_value_offset_[u.GetValue()+1]);
              value_t outv_b = app_->default_v();
              app_->g_function_pull_by_index(*graph_, u, value, outv_b, iindexes); // pull value
              app_->accumulate(value, outv_b);
              app_->accumulate(delta, outv_b);
              #ifdef DEBUG
                //n_edge += ies.Size();
                //n_edge += iindexes.Size();
              #endif
            }

            #ifdef DEBUG
              send_time_3 += GetCurrentTime();
              std::cout << "N edge: " << n_edge << std::endl;
            #endif
          }

          if(compr_stage){
            // copy delta to last delta
            // LOG(INFO) << "copy last delta..";
            #ifdef DEBUG
              exchange_delta_time -= GetCurrentTime();
            #endif
            app_->get_last_delta(*graph_, all_nodes);
            #ifdef DEBUG
              exchange_delta_time += GetCurrentTime();
            #endif

            /* 0. out node */
            const std::vector<vertex_t>& nodes_0 = all_nodes[0];
            vid_t node_0_size = nodes_0.size();
            #ifdef DEBUG
              send_time_0 -= GetCurrentTime();
              vid_t last_n_edge = n_edge;
            #endif
            //#pragma cilk grainsize = 1
            parallel_for(vid_t i = 0; i < node_0_size; i++){
              vertex_t u = nodes_0[i];
              auto& value = values[u];
              auto& delta = deltas[u];
              auto ies = graph_->GetIncomingAdjList(u);
              app_->g_function_pull(*graph_, u, value, delta, ies); // pull
              app_->accumulate(value, delta);
              #ifdef DEBUG
                //n_edge += ies.Size();
                atomic_add(n_edge, ies.Size());
              #endif
            }
            // std::atomic<vid_t> node_id(0);
            // int thread_num = FLAGS_app_concurrency;
            // ForEach(node_0_size, [this, &values, &deltas, &nodes_0, &node_id, &node_0_size](int tid) {
            //     int i = 0, cnt = 0, step = 1024;  // step need to be adjusted
            //     double thread_time = GetCurrentTime();
            //     while(i < node_0_size){
            //         i = node_id.fetch_add(step);
            //         for(int j = i; j < i + step && j < node_0_size; j++){
            //           vertex_t u = nodes_0[j];
            //           auto& value = values[u];
            //           auto& delta = deltas[u];
            //           auto ies = graph_->GetIncomingAdjList(u);
            //           app_->g_function_pull(*graph_, u, value, delta, ies); // pull
            //           app_->accumulate(value, delta);
            //         }
            //     }
            //     LOG(INFO) << "thread_id=" << tid << " time=" << (GetCurrentTime()-thread_time);
            //   }, thread_num
            // );
            #ifdef DEBUG
              send_time_0 += GetCurrentTime();
              time_sum_0 += send_time_0;
              LOG(INFO) << "---time0/edges=" << (send_time_0/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_0_size;
            #endif

            /* 1. only bound node */
            const std::vector<vertex_t>& nodes_1 = all_nodes[1];
            vid_t node_1_size = nodes_1.size();
            #ifdef DEBUG
              send_time_1 -= GetCurrentTime();
              last_n_edge = n_edge;
            #endif
            //#pragma cilk grainsize = 1
            parallel_for(vid_t i = 0; i < node_1_size; i++){
              vertex_t u = nodes_1[i];
              auto& value = values[u];
              auto& delta = deltas[u];
              // auto ies = graph_->GetIncomingAdjList(u);
              adj_list_index_t iindexes = adj_list_index_t(iindex_value_offset_[u.GetValue()], iindex_value_offset_[u.GetValue()+1]);
              app_->g_function_pull_by_index(*graph_, u, value, delta, iindexes); // pull
              app_->accumulate(value, delta);
              #ifdef DEBUG
                //n_edge += iindexes.Size();
                atomic_add(n_edge, iindexes.Size());
              #endif
            }
            #ifdef DEBUG
              send_time_1 += GetCurrentTime();
              time_sum_1 += send_time_1;
              LOG(INFO) << "---time1/edges=" << (send_time_1/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_1_size;
            #endif

            /* 2. only source node */
            const std::vector<vertex_t>& nodes_2 = all_nodes[2];
            vid_t node_2_size = nodes_2.size();
            #ifdef DEBUG
              send_time_2 -= GetCurrentTime();
              last_n_edge = n_edge;
            #endif
            //#pragma cilk grainsize = 1
            parallel_for(vid_t i = 0; i < node_2_size; i++){
              vertex_t u = nodes_2[i];
              auto& value = values[u];
              // auto& delta = deltas[u];
              adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
              value_t outv = app_->default_v();
              app_->g_function_pull(*graph_, u, value, outv, ies); // pull
              app_->accumulate(index_values[u], outv);
              app_->accumulate(spnode_datas[u], outv);
              #ifdef DEBUG
                //n_edge += ies.Size();
                atomic_add(n_edge, ies.Size());
              #endif
            }
            #ifdef DEBUG
              send_time_2 += GetCurrentTime();
              time_sum_2 += send_time_2;
              LOG(INFO) << "---time2/edges=" << (send_time_2/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_2_size;
            #endif

            /* 3. source+bound node */
            const std::vector<vertex_t>& nodes_3 = all_nodes[3];
            vid_t node_3_size = nodes_3.size();
            // #ifdef DEBUG
            //   send_time_3 -= GetCurrentTime();
            //   last_n_edge = n_edge;
            // #endif
            // parallel_for(vid_t i = 0; i < node_3_size; i++){
            //   vertex_t u = nodes_3[i];
            //   auto& value = values[u];
            //   auto& delta = deltas[u];
            //   // 3.1 as source
            //   adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
            //   value_t outv = app_->default_v();
            //   app_->g_function_pull(*graph_, u, value, outv, ies); // pull delta
            //   app_->accumulate(index_values[u], outv);
            //   app_->accumulate(spnode_datas[u], outv);
            //   // 3.2 as bound
            //   adj_list_index_t iindexes = adj_list_index_t(iindex_value_offset_[u.GetValue()], iindex_value_offset_[u.GetValue()+1]);
            //   value_t outv_b = app_->default_v();
            //   app_->g_function_pull_by_index(*graph_, u, value, outv_b, iindexes); // pull value
            //   app_->accumulate(value, outv_b);
            //   app_->accumulate(delta, outv_b);
            //   #ifdef DEBUG
            //     //n_edge += ies.Size();
            //     //n_edge += iindexes.Size();
            //     //atomic_add(n_edge, ies.Size());
            //     //atomic_add(n_edge, iindexes.Size());
            //   #endif
            // }
            // #ifdef DEBUG
            //   send_time_3 += GetCurrentTime();
            //   time_sum_3 += send_time_3;
            //   LOG(INFO) << "---time3/edges=" << (send_time_3/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge);
            //   // std::cout << "N edge: " << n_edge << std::endl;
            // #endif
          
            #ifdef DEBUG
              send_time_3 -= GetCurrentTime();
              last_n_edge = n_edge;
            #endif
            // #pragma cilk grainsize = min(512, N / (8*p)) // =min(512, 7816)
            #pragma cilk grainsize = 1 // uk-2005 开
            parallel_for(vid_t i = 0; i < node_3_size; i++){
              vertex_t u = nodes_3[i];
              auto& value = values[u];
              // auto& delta = deltas[u]; // 分开了就不需要
              // 3.1 as source
              adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
              // adj_list_t ies = adj_list_t(is_ie_offset_3[i], is_ie_offset_3[i+1]); // 增量部分没有预处理
              value_t outv = app_->default_v();
              app_->g_function_pull(*graph_, u, value, outv, ies); // pull delta
              app_->accumulate(index_values[u], outv);
              app_->accumulate(spnode_datas[u], outv);
              #ifdef DEBUG
                //n_edge += ies.Size();
                atomic_add(n_edge, ies.Size());
              #endif
            }
            // std::atomic<vid_t> node_id(0);
            // int thread_num = FLAGS_app_concurrency;
            // ForEach(node_3_size, [this, &values, &deltas, &nodes_3, &node_id, &node_3_size, &index_values](int tid) {
            //     int i = 0, cnt = 0, step = 1;  // step need to be adjusted
            //     // double thread_time = GetCurrentTime();
            //     while(i < node_3_size){
            //         i = node_id.fetch_add(step);
            //         for(int j = i; j < i + step && j < node_3_size; j++){
            //           vertex_t u = nodes_3[i];
            //           auto& value = values[u];
            //           // auto& delta = deltas[u]; // 分开了就不需要
            //           // 3.1 as source
            //           adj_list_t ies = adj_list_t(is_ie_offset_[u.GetValue()], is_ie_offset_[u.GetValue()+1]);
            //           // adj_list_t ies = adj_list_t(is_ie_offset_3[i], is_ie_offset_3[i+1]);
            //           value_t outv = app_->default_v();
            //           app_->g_function_pull(*graph_, u, value, outv, ies); // pull delta
            //           app_->accumulate(index_values[u], outv);
            //           app_->accumulate(spnode_datas[u], outv);
            //         }
            //     }
            //     // LOG(INFO) << "----thread_id=" << tid << " time=" << (GetCurrentTime()-thread_time);
            //   }, thread_num
            // );
            #ifdef DEBUG
              send_time_3 += GetCurrentTime();
              time_sum_3 += send_time_3;
              LOG(INFO) << "---time3/edges=" << (send_time_3/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_3_size;
            #endif

            #ifdef DEBUG
              send_time_4 -= GetCurrentTime();
              last_n_edge = n_edge;
            #endif
            //#pragma cilk grainsize = 1
            parallel_for(vid_t i = 0; i < node_3_size; i++){
              vertex_t u = nodes_3[i];
              auto& value = values[u];
              auto& delta = deltas[u];
              // 3.2 as bound
              adj_list_index_t iindexes = adj_list_index_t(iindex_value_offset_[u.GetValue()], iindex_value_offset_[u.GetValue()+1]);
              value_t outv_b = app_->default_v();
              app_->g_function_pull_by_index(*graph_, u, value, outv_b, iindexes); // pull value
              app_->accumulate(value, outv_b);
              app_->accumulate(delta, outv_b);
              #ifdef DEBUG
                //n_edge += iindexes.Size();
                atomic_add(n_edge, iindexes.Size());
              #endif
            }
            #ifdef DEBUG
              send_time_4 += GetCurrentTime();
              time_sum_4 += send_time_4;
              LOG(INFO) << "---time3/edges=" << (send_time_3/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_3_size;
            #endif
          }

        } else {
          ForEach(inner_vertices,
                  [this, &values, &deltas, &compr_stage, &pri](int tid, vertex_t u) {
                      //   // all nodes send or normal node send
                      //   auto& value = values[u];
                      //   auto delta = atomic_exch(deltas[u], app_->default_v());
                      //   auto oes = graph_->GetOutgoingAdjList(u);

                      //   app_->g_function(*graph_, u, value, delta, oes);
                      //   app_->accumulate_atomic(value, delta);
                  });
        }
        #ifdef DEBUG
          VLOG(1) << "Iter time: " << GetCurrentTime() - begin;
        #endif
      }

      {
        #ifdef DEBUG
          auto begin = GetCurrentTime();
        #endif
        /* remote nodes pull delta */
        ForEach(outer_vertices, [this, &deltas, &channels](int tid,
                                                           vertex_t u) {
          value_t value = app_->default_v();
          auto& delta = deltas[u];
          auto ies = graph_->GetIncomingAdjList(u); // 不知道这里会pull到不在这个work的点不?
          app_->g_function_pull(*graph_, u, value, delta, ies); // pull
          app_->accumulate(value, delta);
        });
        // send local delta to remote
        ForEach(outer_vertices, [this, &deltas, &channels](int tid,
                                                           vertex_t v) {
          auto& delta_to_send = deltas[v];

          if (delta_to_send != app_->default_v()) {
            channels[tid].template SyncStateOnOuterVertex<fragment_t, value_t>(
                *graph_, v, delta_to_send);
            delta_to_send = app_->default_v();
          }
        });
        #ifdef DEBUG
          VLOG(1) << "Send time: " << GetCurrentTime() - begin;
        #endif
        // LOG(INFO) << "Send time: " << GetCurrentTime() - begin;
      }

      #ifdef DEBUG
        VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      #endif
      // default_work,同步一轮
      messages_.FinishARound();

      exec_time += GetCurrentTime();
      #ifdef DEBUG
        LOG(INFO) << "step=" << step << " one_step_time=" << (GetCurrentTime() - one_step_time);
        LOG(INFO) << "time_sum_0=" << time_sum_0 << " ave_time=" << (time_sum_0/step);
        LOG(INFO) << "time_sum_1=" << time_sum_1 << " ave_time=" << (time_sum_1/step);
        LOG(INFO) << "time_sum_2=" << time_sum_2 << " ave_time=" << (time_sum_2/step);
        LOG(INFO) << "time_sum_3=" << time_sum_3 << " ave_time=" << (time_sum_3/step);
        LOG(INFO) << "time_sum_4=" << time_sum_4 << " ave_time=" << (time_sum_4/step);
        LOG(INFO) << "exchange_delta_time=" << exchange_delta_time;
        LOG(INFO) << "send_time_0=" << send_time_0 << " send_time_1=" << send_time_1 << " send_time_2=" << send_time_2 << " send_time_3=" << send_time_3;
      #endif

      if (termCheck(last_values, values) || step > 200) {
        if(compr_stage){
          LOG(INFO) << "step=" << step;
          LOG(INFO) << "start correct deviation...";
          timer_next("correct deviation");
          compr_stage = false;
          double corr_time = GetCurrentTime();

          // recover bound delta
          // value_t bound_delta_sum = 0; // debug
          // value_t bound_delta_sum_2 = 0; // debug
          // value_t bound_delta_sum_in = 0; // debug
          const std::vector<vertex_t>& nodes_1 = all_nodes[1];
          vid_t node_1_size = nodes_1.size();
          parallel_for(vid_t i = 0; i < node_1_size; i++){
            vertex_t u = nodes_1[i];
            // app_->accumulate_atomic(bound_delta_sum_2, deltas[u]); // debug
            deltas[u] = bound_deltas[u];
            app_->accumulate(values[u], deltas[u]);
            // app_->accumulate_atomic(bound_delta_sum, deltas[u]); // debug
          }
          const std::vector<vertex_t>& nodes_4 = all_nodes[4];
          vid_t node_4_size = nodes_4.size(); // 可以与下面的合并
          parallel_for(vid_t i = 0; i < node_4_size; i++){
            vertex_t u = nodes_4[i];
            // app_->accumulate_atomic(bound_delta_sum_2, deltas[u]); // debug
            deltas[u] = bound_deltas[u];
            app_->accumulate(values[u], deltas[u]);
            // app_->accumulate_atomic(bound_delta_sum_in, deltas[u]); // debug
          }

          /* 2/4: pull value */
          const std::vector<vertex_t>& nodes_2 = all_nodes[2];
          vid_t node_2_size = nodes_2.size();
          parallel_for(vid_t i = 0; i < node_2_size; i++){
            vertex_t u = nodes_2[i];
            auto& value = values[u];
            value_t outv = app_->default_v();
            adj_list_index_t iindexes = adj_list_index_t(iindex_value_offset_[u.GetValue()], iindex_value_offset_[u.GetValue()+1]);
            app_->g_function_pull_spnode_datas_by_index(*graph_, u, value, outv, iindexes, spnode_datas); // pull
            app_->accumulate(value, outv);
          }
          // vid_t node_4_size = all_nodes[4].size();
          parallel_for(vid_t i = 0; i < node_4_size; i++){
            vertex_t u = nodes_4[i];
            auto& value = values[u];
            value_t outv = app_->default_v();
            adj_list_index_t iindexes = adj_list_index_t(iindex_value_offset_[u.GetValue()], iindex_value_offset_[u.GetValue()+1]);
            app_->g_function_pull_spnode_datas_by_index(*graph_, u, value, outv, iindexes, spnode_datas); // pull
            app_->accumulate(value, outv);
          }
          /* 1/2/3/4: pull delta */
          value_t wc_delta_sum = 0; //debug
          value_t all_delta_sum = 0; //
          value_t all_value_sum = 0; //
          vid_t size = 0;
          for(vid_t j = 1; j <= 4; j++){
          // for(vid_t j = 0; j <= 4; j++){ // debug: j=0-4
            std::vector<vertex_t>& nodes = all_nodes[j];
            vid_t node_size = nodes.size();
            parallel_for(vid_t i = 0; i < node_size; i++){
              vertex_t u = nodes[i];
              auto& value = values[u];
              auto& delta = deltas[u];
              value_t outv = app_->default_v();
              adj_list_index_t iindexes = adj_list_index_t(iindex_delta_offset_[u.GetValue()], iindex_delta_offset_[u.GetValue()+1]);
              app_->g_function_pull_spnode_datas_by_index(*graph_, u, value, outv, iindexes, spnode_datas); // pull
              app_->accumulate(value, outv);
              app_->accumulate(delta, outv);

              // app_->accumulate_atomic(wc_delta_sum, outv);
              // app_->accumulate_atomic(all_delta_sum, delta);
              // app_->accumulate_atomic(all_value_sum, value);
            }
            // size += node_size;
          }
          
          // termCheck(last_values, values); //debug
          // LOG(INFO) << "------all node size=" << size;
          // LOG(INFO) << "------be cover bound_delta_sum_2=" << bound_delta_sum_2;
          // LOG(INFO) << "------wc_delta_sum=" << wc_delta_sum;
          // LOG(INFO) << "------all_delta_sum=" << all_delta_sum;
          // LOG(INFO) << "------bound_delta_sum=" << bound_delta_sum;
          // LOG(INFO) << "------bound_delta_sum_in=" << bound_delta_sum_in;
          // LOG(INFO) << "------all_value_sum=" << all_value_sum;
          // LOG(INFO) << "correct deviation in supernode";
          // LOG(INFO) << "#1st_step: " << step;

          LOG(INFO) << "#corr_time: " << (GetCurrentTime()-corr_time);
          continue;
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#iter step: " << step;
            LOG(INFO) << "#Batch time: " << exec_time << " sec";
          }
          exec_time = 0;
          step = 1;

          if (!FLAGS_efile_update.empty()) {
            compr_stage = FLAGS_compress; // use supernode
            timer_next("reloadGraph");
            reloadGraph();
            LOG(INFO) << "start inc...";
            timer_next("inc algorithm");
            CHECK_EQ(inner_vertices.size(), graph_->InnerVertices().size());
            inner_vertices = graph_->InnerVertices();
            outer_vertices = graph_->OuterVertices();
            CHECK_EQ(values.size(), app_->values_.size());
            CHECK_EQ(deltas.size(), app_->deltas_.size());
            values = app_->values_;
            deltas = app_->deltas_;

            if(compr_stage){
              #ifdef DEBUG
                value_t init_value_sum = 0;
                value_t init_delta_sum = 0;
                for(auto v : graph_->Vertices()){
                  init_delta_sum += deltas[v];
                  init_value_sum += values[v];
                }
                LOG(INFO) << "init_value_sum=" << init_value_sum << " init_delta_sum=" << init_delta_sum;
              #endif

              /* vertex classification */
              double node_type_time = GetCurrentTime();
              vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
              node_type.clear();
              node_type.resize(inner_node_num, std::numeric_limits<char>::max());
              bound_node_values.clear();
              bound_node_values.Init(inner_vertices, app_->default_v());
              all_nodes.clear();
              all_nodes.resize(5);
              parallel_for(vid_t i = inner_vertices.begin().GetValue();
                  i < inner_vertices.end().GetValue(); i++) {
                    vertex_t u(i);
                  if(cpr_->Fc[u] == cpr_->FC_default_value){
                    node_type[i] = 0; // out node
                  }
                  else if(cpr_->Fc[u] >= 0){
                    node_type[i] = 2; // only source node
                  }
                  else if(!cpr_->supernode_out_bound[i]){
                    node_type[i] = 4; // inner node
                  }
                  if(cpr_->supernode_out_bound[i]){
                    node_type[i] = 1; // only bound node
                    if(cpr_->Fc[u] >= 0){
                      node_type[i] = 3; // source node + bound node
                    }
                  }
              }
              for(vid_t i = inner_vertices.begin().GetValue();
                  i < inner_vertices.end().GetValue(); i++) {
                  all_nodes[node_type[i]].emplace_back(vertex_t(i));
              }
              LOG(INFO) << "node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

              // debug
              {
                vid_t all_node_size = 0;
                for(int i = 0; i <= 4; i++){
                  LOG(INFO) << "----node_type=" << i << " size=" << all_nodes[i].size();
                  all_node_size += all_nodes[i].size();
                }
                LOG(INFO) << "all_node_size=" << all_node_size << " inner_node_num=" << inner_node_num;
              }

              double  transfer_csr_time = GetCurrentTime();

              /* get in-adj list */
              std::vector<std::vector<nbr_index_t> > iindex_delta_vec(inner_node_num); // all delta index
              std::vector<std::vector<nbr_index_t> > iindex_value_vec(inner_node_num); // all value index

              vid_t index_num1 = 0;
              vid_t index_num2 = 0;
              for(vid_t i = 0; i < cpr_->supernodes_num; i++) {
                supernode_t &spnode = cpr_->supernodes[i];
                vertex_t v(spnode.id);
                for(auto oe : spnode.bound_delta){
                  iindex_value_vec[oe.first.GetValue()].emplace_back(nbr_index_t(v, oe.second));
                  index_num1++;
                }
                for(auto oe : spnode.inner_value){
                  iindex_value_vec[oe.first.GetValue()].emplace_back(nbr_index_t(v, oe.second));
                  index_num1++;
                }
                for(auto oe : spnode.inner_delta){
                  iindex_delta_vec[oe.first.GetValue()].emplace_back(nbr_index_t(v, oe.second));
                  index_num2++;
                }
              }
              LOG(INFO) << "index_value_num=" << index_num1 << " index_delta_num=" << index_num2;

              double  init_time_1 = GetCurrentTime();
              spnode_datas.Init(inner_vertices, 0);
              index_values.Init(inner_vertices, 0);
              is_ie_.clear();
              is_ie_offset_.clear();
              iindex_delta_.clear();
              iindex_delta_offset_.clear();
              iindex_value_.clear();
              iindex_value_offset_.clear();
              size_t source_ie_num = 0;
              size_t iindex_delta_num = 0;
              size_t iindex_value_num = 0;
              std::vector<size_t> is_ie_degree(inner_node_num+1, 0);
              std::vector<size_t> iindex_delta_degree(inner_node_num+1, 0);
              std::vector<size_t> iindex_value_degree(inner_node_num+1, 0);
              LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


              double csr_time_1 = GetCurrentTime();
              // for(auto u : inner_vertices){
              parallel_for(vid_t i = inner_vertices.begin().GetValue();
                  i < inner_vertices.end().GetValue(); i++) {
                vertex_t u(i);
                char type = node_type[i];
                // if(type == 1 || type == 4){
                if(type != 0){ // not out node
                  iindex_delta_degree[i+1] = iindex_delta_vec[i].size();
                  iindex_value_degree[i+1] = iindex_value_vec[i].size();
                  atomic_add(iindex_delta_num, iindex_delta_vec[i].size());
                  atomic_add(iindex_value_num, iindex_value_vec[i].size());
                }
                if(type == 2 || type == 3){ // source + bound
                  auto ies = graph_->GetIncomingAdjList(u);
                  auto it = ies.begin();
                  auto in_degree = ies.Size();
                  vid_t ids_id = cpr_->id2spids[u];
                  size_t temp_cnt = 0;
                  for(vid_t j = 0; j < in_degree; j++){
                    auto& e = *(it + j);
                    if(ids_id != cpr_->id2spids[e.neighbor]){
                      // bound_e_num += 1;
                      temp_cnt += 1;
                    }
                  }
                  is_ie_degree[i+1] += temp_cnt;
                  atomic_add(source_ie_num, temp_cnt);
                }
              }
              LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987
              LOG(INFO) << "iindex_delta_num=" << iindex_delta_num;
              LOG(INFO) << "iindex_value_num=" << iindex_value_num;
              LOG(INFO) << "source_ie_num=" << source_ie_num;

              /* get index start */
              double index_time = GetCurrentTime();
              for(vid_t i = 1; i < inner_node_num; i++) {
                is_ie_degree[i] += is_ie_degree[i-1];
                iindex_delta_degree[i] += iindex_delta_degree[i-1];
                iindex_value_degree[i] += iindex_value_degree[i-1];
              }
              LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

              LOG(INFO) << "inner_node_num=" << inner_node_num;
              LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

              double init_time_2 = GetCurrentTime();
              iindex_delta_.resize(iindex_delta_num);
              iindex_value_.resize(iindex_value_num);
              is_ie_.resize(source_ie_num);
              iindex_delta_offset_.resize(inner_node_num+1);
              iindex_value_offset_.resize(inner_node_num+1);
              is_ie_offset_.resize(inner_node_num+1);
              LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

              double csr_time_2 = GetCurrentTime();
              parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
              // for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
                vertex_t u(i);
                /* delta/value index */
                vid_t index_delta = iindex_delta_degree[i];
                vid_t index_value = iindex_value_degree[i];
                iindex_delta_offset_[i] = &iindex_delta_[index_delta];
                iindex_value_offset_[i] = &iindex_value_[index_value];
                char type = node_type[i];
                if(type != 0){
                  for(auto& ie : iindex_delta_vec[i]){
                    iindex_delta_[index_delta] = ie;
                    index_delta++;
                  }
                  for(auto& ie : iindex_value_vec[i]){
                    iindex_value_[index_value] = ie;
                    index_value++;
                  } 
                }
                /* in-edge of source node and s_b node */
                vid_t index_s = is_ie_degree[i];
                is_ie_offset_[i] = &is_ie_[index_s];
                if(type == 2 || type == 3){
                  auto ies = graph_->GetIncomingAdjList(u);
                  auto it = ies.begin();
                  auto in_degree = ies.Size();
                  vid_t ids_id = cpr_->id2spids[u];
                  for(vid_t j = 0; j < in_degree; j++){
                    auto& ie = *(it + j);
                    if(ids_id != cpr_->id2spids[ie.neighbor]){
                      is_ie_[index_s] = ie;
                      index_s++;
                    }
                  }
                }
              }
              iindex_delta_offset_[inner_node_num] = &iindex_delta_[iindex_delta_num-1] + 1;
              iindex_value_offset_[inner_node_num] = &iindex_value_[iindex_value_num-1] + 1;
              is_ie_offset_[inner_node_num] = &is_ie_[source_ie_num-1] + 1;
              LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

              LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149
            }
            
            parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
              vertex_t v(i);
              auto& value = values[v];
              auto& delta = deltas[v];
              last_values[v] = value;
              if(compr_stage){
                if(node_type[i] == 0){
                  app_->accumulate(value, delta); // 增量阶段可能也需要先聚合一次
                }
                else if(node_type[i] == 1 || node_type[i] == 4){
                  bound_deltas[v] = delta; // 对于bound node copy一份后应该清空
                  delta = app_->default_v();
                }
                else if(node_type[i] == 2 || node_type[i] == 3){
                  app_->accumulate(index_values[v], delta); // delta没有被bound点用掉，下次就清空了
                  app_->accumulate(spnode_datas[v], delta);
                  delta = app_->default_v();
                }
              }
              else{
                app_->accumulate(value, delta);
              }
            }
            continue;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#Inc iter step: " << step;
            LOG(INFO) << "#Inc time: " << exec_time << " sec";
          }
          break;
        }
      }
      ++step;
    }

    // Analysis result
    double d_sum = 0;
    vertex_t source;
    bool native_source =
        graph_->GetInnerVertex(FLAGS_sssp_source, source);
    vid_t max_id = native_source ? source.GetValue() : 0;
    for (auto v : graph_->InnerVertices()) {
      d_sum += app_->values_[v];
      if (app_->values_[v] > app_->values_[vertex_t(max_id)]) {
        max_id = v.GetValue();
      }
    }
    LOG(INFO) << "max_d[" << graph_->GetId(vertex_t(max_id)) << "]=" << app_->values_[vertex_t(max_id)];
    LOG(INFO) << "d_sum=" << d_sum;

    MPI_Barrier(comm_spec_.comm());
    if(compr_stage){
      delete cpr_;
    }
  }

  void Output(std::ostream& os) {
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;

    for (auto v : inner_vertices) {
      os << graph_->GetId(v) << " " << values[v] << std::endl;
    }
  }

  void Finalize() { messages_.Finalize(); }

 private:
  bool termCheck(VertexArray<value_t, vid_t>& last_values,
                 VertexArray<value_t, vid_t>& values) {
    // terminate_checking_time_ -= GetCurrentTime();
    auto vertices = graph_->InnerVertices();
    double diff_sum = 0, global_diff_sum;
    // double value_sum_1 = 0; // debug
    // double value_sum_2 = 0; // debug

    // for (auto u : vertices) {
    //   // value_sum_1 += last_values[u];
    //   // value_sum_2 += values[u];
    //   diff_sum += fabs(last_values[u] - values[u]);
    //   last_values[u] = values[u];
    // }

    cilk::reducer_opadd<value_t> total(0);
    // #pragma cilk grainsize = 10000
    parallel_for(vid_t j = 0; j < vertices.end().GetValue(); j++){ // 粒度应该设的大
      vertex_t u(j);
      total += fabs(last_values[u] - values[u]);
      last_values[u] = values[u];
    }
    diff_sum = total.get_value();

    communicator_.template Sum(diff_sum, global_diff_sum);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Diff: " << global_diff_sum;
      // LOG(INFO) << "value_sum_1=" << value_sum_1 << " value_sum_2=" << value_sum_2;
    }

    terminate_checking_time_ += GetCurrentTime();

    return global_diff_sum < FLAGS_termcheck_threshold;
  }

  // VertexArray<vid_t, vid_t> spnode_ids;
  std::vector<char> node_type; // all node's types, 0:out node, 1:bound node, 2:source node, 3:belong 1 and 2 at the same time, 127:inner node that needn't send message.
  VertexArray<value_t, vid_t> bound_node_values;
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t>& graph_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
  double terminate_checking_time_;
  IterCompressor<APP_T, supernode_t>* cpr_;
  // std::vector<value_t> spnode_datas;
  VertexArray<value_t, vid_t> spnode_datas{};;
  /* source to in_bound_node, i.e, oindex */
  Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
  /* in_bound_node to out_bound_node */
  Array<nbr_t, Allocator<nbr_t>> ib_e_;
  Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
  /* pull: in-index of all inner node*/
  Array<nbr_index_t, Allocator<nbr_index_t>> iindex_delta_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> iindex_delta_offset_;
  Array<nbr_index_t, Allocator<nbr_index_t>> iindex_value_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> iindex_value_offset_;
  /* pull: in-edge of source node that filter inner edge in supernode */
  Array<nbr_t, Allocator<nbr_t>> is_ie_;
  Array<nbr_t*, Allocator<nbr_t*>> is_ie_offset_;
  Array<nbr_t, Allocator<nbr_t>> is_ie_3; // type 3: bound + source node
  Array<nbr_t*, Allocator<nbr_t*>> is_ie_offset_3;
  /* each type of vertices */
  std::vector<std::vector<vertex_t>> all_nodes;

  class compare_priority {
   public:
    VertexArray<value_t, vid_t>& parent;

    explicit compare_priority(VertexArray<value_t, vid_t>& inparent)
        : parent(inparent) {}

    bool operator()(const vid_t a, const vid_t b) {
      return abs(parent[Vertex<unsigned int>(a)]) >
             abs(parent[Vertex<unsigned int>(b)]);
    }
  };
};

}  // namespace grape

#endif  // GRAPE_WORKER_SUM_SYNC_ITER_WORKER_PULL_H_

