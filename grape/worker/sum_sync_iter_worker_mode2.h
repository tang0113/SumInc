#ifndef GRAPE_WORKER_SUM_SYNC_ITER_WORKER_H_
#define GRAPE_WORKER_SUM_SYNC_ITER_WORKER_H_

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
class SumSyncIterWorker : public ParallelEngine {
  static_assert(std::is_base_of<IterateKernel<typename APP_T::fragment_t,
                                              typename APP_T::value_t>,
                                APP_T>::value,
                "SumSyncIterWorker should work with App");

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

  SumSyncIterWorker(std::shared_ptr<APP_T> app,
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

    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    //    auto& prioritys = app_->priority_;
    VertexArray<value_t, vid_t> last_values;


    int step = 1;
    bool batch_stage = true;
    bool compr_stage = FLAGS_compress; // true: supernode send


    last_values.Init(inner_vertices);
    value_t init_value_sum = 0;
    value_t init_delta_sum = 0;

    // for (auto v : inner_vertices) {
    parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
      vertex_t v(i);
      last_values[v] = values[v];
    }

    if(compr_stage){
      #ifdef DEBUG
        for(auto v : graph_->Vertices()){
          init_delta_sum += deltas[v];
          init_value_sum += values[v];
        }
        LOG(INFO) << "init_value_sum=" << init_value_sum << " init_delta_sum=" << init_delta_sum;
      #endif

      double node_type_time = GetCurrentTime();
      vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
      node_type.clear();
      node_type.resize(inner_node_num, std::numeric_limits<char>::max());
      bound_node_values.clear();
      bound_node_values.Init(inner_vertices, app_->default_v());
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
          if(cpr_->Fc[u] == cpr_->FC_default_value){
            node_type[i] = 0; // out node
          }
          else if(cpr_->Fc[u] >= 0){
            node_type[i] = 2; // source node
          }
          else if(!cpr_->supernode_out_bound[i]){
            node_type[i] = 4; // inner node
          }
          if(cpr_->supernode_out_bound[i]){
            node_type[i] = 1; // bound node
            if(cpr_->Fc[u] >= 0){
              node_type[i] = 3; // source node + bound node
            }
          }
      }
      all_nodes.clear();
      all_nodes.resize(5);
      for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
          all_nodes[node_type[i]].emplace_back(vertex_t(i));
      }
      LOG(INFO) << "node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

      // debug
      {
        LOG(INFO) << "---node1_num=" << all_nodes[1].size();
        all_nodes[1].clear();
        size_t spn_ids_num = cpr_->supernode_ids.size();
        for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[j]; 
          for(auto v : node_set){
            if(node_type[v.GetValue()] == 1){
              all_nodes[1].emplace_back(v);
            }
          }
        }
        LOG(INFO) << "---node1_num=" << all_nodes[1].size();
        // 统计入度
        std::vector<size_t> node_0_indegree(inner_node_num, 0);
        std::vector<size_t> node_1_indegree(inner_node_num, 0);
        {
          const std::vector<vertex_t>& nodes_0 = all_nodes[0];
          vid_t node_0_size = nodes_0.size();
          for(vid_t i = 0; i < node_0_size; i++){
            vertex_t u = nodes_0[i];
            auto oes = graph_->GetOutgoingAdjList(u);
            for(auto e : oes){
              node_0_indegree[e.neighbor.GetValue()]++;
            }
          }
          size_t max_in_degree_0 = 0;
          for(vid_t i = 0; i < inner_node_num; i++){
            max_in_degree_0 = std::max(max_in_degree_0, node_0_indegree[i]);
          }
          LOG(INFO) << " ----max_in_degree_0=" << max_in_degree_0;
        }
        {
          const std::vector<vertex_t>& nodes_1 = all_nodes[1];
          vid_t node_1_size = nodes_1.size();
          for(vid_t i = 0; i < node_1_size; i++){
            vertex_t u = nodes_1[i];
            auto oes = graph_->GetOutgoingAdjList(u);
            for(auto e : oes){
              node_1_indegree[e.neighbor.GetValue()]++;
            }
          }
          size_t max_in_degree_1 = 0;
          for(vid_t i = 0; i < inner_node_num; i++){
            max_in_degree_1 = std::max(max_in_degree_1, node_1_indegree[i]);
          }
          LOG(INFO) << " ----max_in_degree_1=" << max_in_degree_1;
        }
      }

      double  transfer_csr_time = GetCurrentTime();

      /* source to in_bound_node */
      // spnode_datas.resize(cpr_->supernodes_num);

      double  init_time_1 = GetCurrentTime();
      spnode_datas.Init(inner_vertices, 0);
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == 2 || type == 3){
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          // source_e_num += spnode.bound_delta.size();
          is_e_degree[i+1] = spnode.bound_delta.size();
          atomic_add(source_e_num, spnode.bound_delta.size());
        }
        if(type == 1 || type == 3){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){  // 关闭多线程
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              // bound_e_num += 1;
              temp_cnt += 1;
            }
          }
          ib_e_degree[i+1] += temp_cnt;
          atomic_add(bound_e_num, temp_cnt);
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i < inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      // vid_t index_s = 0;
      // vid_t index_b = 0;
      // build edge
      double csr_time_2 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        vid_t index_s = is_e_degree[i];
        is_e_offset_[u.GetValue()] = &is_e_[index_s];
        char type = node_type[u.GetValue()];
        if(type == 2 || type == 3){
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          for(auto& oe : spnode.bound_delta){
            // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
            is_e_[index_s].neighbor = oe.first;
            is_e_[index_s].data = oe.second;
            index_s++;
          }
        }
        /* inner_bound node */
        vid_t index_b = ib_e_degree[i];
        ib_e_offset_[u.GetValue()] = &ib_e_[index_b];
        if(type == 1 || type == 3){
          auto oes = graph_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              ib_e_[index_b] = e;
              index_b++;
            }
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      //debug
      {
        const std::vector<vertex_t>& nodes_0 = all_nodes[0];
        vid_t node_0_size = nodes_0.size();
        size_t max_edge_0_num = 0;
        size_t edge_0_num = 0;
        for(vid_t i = 0; i < node_0_size; i++){
          vertex_t u(nodes_0[i]);
          auto oes = graph_->GetOutgoingAdjList(u);
          max_edge_0_num = std::max(max_edge_0_num, oes.Size());
          edge_0_num += oes.Size();
        }
        LOG(INFO) << "---max_edge_0_num=" << max_edge_0_num << " edge_0_num=" << edge_0_num << " node_0_size=" << node_0_size;
        const std::vector<vertex_t>& nodes_1 = all_nodes[1];
        vid_t node_1_size = nodes_1.size();
        size_t max_edge_1_num = 0;
        size_t edge_1_num = 0;
        for(vid_t i = 0; i < node_1_size; i++){
          vertex_t u(nodes_1[i]);
          adj_list_t adj = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          max_edge_1_num = std::max(max_edge_1_num, adj.Size());
          edge_1_num += adj.Size();
        }
        LOG(INFO) << "---max_edge_1_num=" << max_edge_1_num << " edge_1_num=" << edge_1_num << " node_1_size=" << node_1_size;
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149
    }

    LOG(INFO) << "compr_stage=" << compr_stage;

    double exec_time = 0;
    double corr_time = 0;
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
// LOG(INFO) << "\nopen debug...";

    #ifdef DEBUG
      double time_sum_0 = 0;
      double time_sum_1 = 0;
      double time_sum_2 = 0;
      double time_sum_3 = 0;
      double time_sum_4 = 0;
    #endif

    while (true) {
      LOG(INFO) << "step=" << step << " f_send_value_num=" << app_->f_send_value_num << " f_send_delta_num=" << app_->f_send_delta_num;
      app_->f_send_value_num = 0;
      app_->f_send_delta_num = 0;
      LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.size();
      app_->node_update_num = 0;

      exec_time -= GetCurrentTime();
      #ifdef DEBUG
        one_step_time = GetCurrentTime();
        double send_time_0 = 0;
        double send_time_1 = 0;
        double send_time_2 = 0;
        double send_time_3 = 0;
        double send_time_4 = 0;
        size_t n_edge = 0;
        vid_t last_n_edge = 0;

        double out_node_send_time = 0;
        double bound_send_time = 0;
        double source_send_time = 0;
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

            // value_t priority;
            // if(FLAGS_portion < 1){
            //   priority = Scheduled(1000);
            // }

            #ifdef DEBUG
              send_time_0 -= GetCurrentTime();
            #endif
            parallel_for(vid_t i = inner_vertices.begin().GetValue();
                       i < inner_vertices.end().GetValue(); i++) {
              vertex_t u(i);
              value_t& old_delta = deltas[u];
              // if(FLAGS_portion == 1 || old_delta >= priority){
                auto& value = values[u];
                auto delta = atomic_exch(deltas[u], app_->default_v());
                auto oes = graph_->GetOutgoingAdjList(u);

                app_->g_function(*graph_, u, value, delta, oes);
                app_->accumulate_atomic(value, delta);
                #ifdef DEBUG
                  //n_edge += oes.Size();
                #endif
              // }
            }
            #ifdef DEBUG
              send_time_0 += GetCurrentTime();
              time_sum_0 += send_time_0;
              //LOG(INFO) << "time0/edges=" << (send_time_0/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_0_size;
              std::cout << "N edge: " << n_edge << std::endl;
            #endif
          }

          if(compr_stage){
            // value_t priority;
            // if(FLAGS_portion < 1){
            //   priority = Scheduled(1000);
            // }
            if(1){
            parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
              vertex_t u(i);
              switch (node_type[i]){
              case 0:
                /* 1. out node */
                {
                  #ifdef DEBUG
                    out_node_send_time -= GetCurrentTime();
                  #endif
                  value_t& old_delta = deltas[u];
                  // if(FLAGS_portion == 1 || old_delta >= priority){
                    auto delta = atomic_exch(old_delta, app_->default_v());
                    auto& value = values[u];
                    auto oes = graph_->GetOutgoingAdjList(u);
                    #ifdef DEBUG
                      //n_edge += oes.Size();
                    #endif
                    app_->g_function(*graph_, u, value, delta, oes);
                    app_->accumulate_atomic(value, delta);
                  // }
                  #ifdef DEBUG
                    out_node_send_time += GetCurrentTime();
                  #endif
                }
                break;
              case 2:
                /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                {
                  #ifdef DEBUG
                    source_send_time -= GetCurrentTime();
                  #endif
                  value_t& old_delta = deltas[u];
                  // if(FLAGS_portion == 1 || old_delta >= priority){
                    // vid_t index_s = source_edge_index[j];
                    auto delta = atomic_exch(old_delta, app_->default_v());
                    auto& value = values[u];
                    adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], is_e_offset_[i+1]);
                    app_->g_index_function(*graph_, u, value, delta, adj, bound_node_values);
                    app_->accumulate_atomic(spnode_datas[u], delta);
                    #ifdef DEBUG
                      //n_edge += adj.Size();
                    #endif
                  // }
                  #ifdef DEBUG
                    source_send_time += GetCurrentTime();
                  #endif
                }
                break;
              case 1:
                /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                {
                  #ifdef DEBUG
                    bound_send_time -= GetCurrentTime();
                  #endif
                  value_t& old_delta = bound_node_values[u];
                  // if(FLAGS_portion == 1 || old_delta >= priority){
                    auto delta = atomic_exch(old_delta, app_->default_v());
                    auto& value = values[u];
                    auto oes = graph_->GetOutgoingAdjList(u);
                    adj_list_t adj = adj_list_t(ib_e_offset_[i], ib_e_offset_[i+1]);
                    app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                    app_->accumulate_atomic(value, delta);
                    #ifdef DEBUG
                      //n_edge += adj.Size();
                    #endif
                  // }
                  #ifdef DEBUG
                    bound_send_time += GetCurrentTime();
                  #endif
                }
                break;
              case 3:
                /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                {
                  #ifdef DEBUG
                    source_send_time -= GetCurrentTime();
                  #endif
                  value_t& old_delta = deltas[u];
                  // if(FLAGS_portion == 1 || old_delta >= priority){
                    // vid_t index_s = source_edge_index[j];
                    auto delta = atomic_exch(old_delta, app_->default_v());
                    auto& value = values[u];
                    adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], is_e_offset_[i+1]);
                    app_->g_index_function(*graph_, u, value, delta, adj, bound_node_values);
                    app_->accumulate_atomic(spnode_datas[u], delta);
                    #ifdef DEBUG
                      //n_edge += adj.Size();
                    #endif
                  // }
                  #ifdef DEBUG
                    source_send_time += GetCurrentTime();
                  #endif
                }
                /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                {
                  #ifdef DEBUG
                    bound_send_time -= GetCurrentTime();
                  #endif
                  value_t& old_delta = bound_node_values[u];
                  // if(FLAGS_portion == 1 || old_delta >= priority){
                    auto delta = atomic_exch(old_delta, app_->default_v());
                    auto& value = values[u];
                    auto oes = graph_->GetOutgoingAdjList(u);
                    adj_list_t adj = adj_list_t(ib_e_offset_[i], ib_e_offset_[i+1]);
                    app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                    app_->accumulate_atomic(value, delta);
                    #ifdef DEBUG
                      //n_edge += adj.Size();
                      //atomic_add(n_edge, oes.Size());
                    #endif
                  // }
                  #ifdef DEBUG
                    bound_send_time += GetCurrentTime();
                  #endif
                }
                break;
              }
            }
            #ifdef DEBUG
              LOG(INFO) << "N edge: " << n_edge << std::endl;
            #endif
            }

            if(0){
              /* 0. out node */
              const std::vector<vertex_t>& nodes_0 = all_nodes[0];
              vid_t node_0_size = nodes_0.size();
              #ifdef DEBUG
                send_time_0 -= GetCurrentTime();
                last_n_edge = n_edge;
              #endif
              parallel_for(vid_t i = 0; i < node_0_size; i++){
                vertex_t u = nodes_0[i];
                value_t& old_delta = deltas[u];
                auto delta = atomic_exch(old_delta, app_->default_v());
                auto& value = values[u];
                auto oes = graph_->GetOutgoingAdjList(u);
                app_->g_function(*graph_, u, value, delta, oes);
                app_->accumulate_atomic(value, delta);
                #ifdef DEBUG
                  //n_edge += oes.Size();
                  //atomic_add(n_edge, oes.Size());
                #endif
              }
              #ifdef DEBUG
                send_time_0 += GetCurrentTime();
                time_sum_0 += send_time_0;
                //LOG(INFO) << "time0/edges=" << (send_time_0/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_0_size;
              #endif

              /* 2. source node: source send message to inner_bound_node by inner_bound_index */
              const std::vector<vertex_t>& nodes_2 = all_nodes[2];
              vid_t node_2_size = nodes_2.size();
              #ifdef DEBUG
                send_time_2 -= GetCurrentTime();
                last_n_edge = n_edge;
              #endif
              parallel_for(vid_t i = 0; i < node_2_size; i++){
                vertex_t u = nodes_2[i];
                value_t& old_delta = deltas[u];
                auto delta = atomic_exch(old_delta, app_->default_v());
                auto& value = values[u];
                adj_list_index_t adj = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                app_->g_index_function(*graph_, u, value, delta, adj, bound_node_values);
                app_->accumulate_atomic(spnode_datas[u], delta);
                #ifdef DEBUG
                  //n_edge += adj.Size();
                  //atomic_add(n_edge, adj.Size());
                #endif
              }
              #ifdef DEBUG
                send_time_2 += GetCurrentTime();
                time_sum_2 += send_time_2;
                //LOG(INFO) << "time2/edges=" << (send_time_2/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_2_size;
              #endif

              /* 1. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
              const std::vector<vertex_t>& nodes_1 = all_nodes[1];
              vid_t node_1_size = nodes_1.size();
              #ifdef DEBUG
                send_time_1 -= GetCurrentTime();
                last_n_edge = n_edge;
              #endif
              // #pragma cilk grainsize = 1024
              parallel_for(vid_t i = 0; i < node_1_size; i++){
// #pragma omp parallel for num_threads(FLAGS_app_concurrency) //schedule(guided, 1)
//               for(vid_t i = 0; i < node_1_size; i++){
                vertex_t u = nodes_1[i];
                value_t& old_delta = bound_node_values[u];
                auto delta = atomic_exch(old_delta, app_->default_v());
                auto& value = values[u];
                auto oes = graph_->GetOutgoingAdjList(u);
                adj_list_t adj = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                app_->accumulate_atomic(value, delta);
                #ifdef DEBUG
                  //n_edge += adj.Size();
                  // if(delta != app_->default_v()){
                  //   atomic_add(n_edge, adj.Size());
                  // }
                #endif
              }

              // std::atomic<vid_t> node_id(0);
              // int thread_num = FLAGS_app_concurrency;
              // ForEach(node_1_size, [this, &values, &deltas, &nodes_1, &node_id, &node_1_size](int tid) {
              //     int i = 0, cnt = 0, step = 1;  // step need to be adjusted
              //     double thread_time = GetCurrentTime();
              //     while(i < node_1_size){
              //         i = node_id.fetch_add(step);
              //         for(int j = i; j < i + step && j < node_1_size; j++){
              //           vertex_t u = nodes_1[j];
              //           value_t& old_delta = bound_node_values[u];
              //           auto delta = atomic_exch(old_delta, app_->default_v());
              //           auto& value = values[u];
              //           auto oes = graph_->GetOutgoingAdjList(u);
              //           adj_list_t adj = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
              //           app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
              //           app_->accumulate_atomic(value, delta);
              //         }
              //     }
              //     LOG(INFO) << "thread_id=" << tid << " time=" << (GetCurrentTime()-thread_time);
              //   }, thread_num
              // );

              #ifdef DEBUG
                send_time_1 += GetCurrentTime();
                time_sum_1 += send_time_1;
                LOG(INFO) << "time1/edges=" << (send_time_1/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_1_size;
              #endif

              /* 3. source node + bound node */
              // 3.1 source send
              const std::vector<vertex_t>& nodes_3 = all_nodes[3];
              vid_t node_3_size = nodes_3.size();
              #ifdef DEBUG
                send_time_3 -= GetCurrentTime();
                last_n_edge = n_edge;
              #endif
              parallel_for(vid_t i = 0; i < node_3_size; i++){
                vertex_t u = nodes_3[i];
                value_t& old_delta = deltas[u];
                auto delta = atomic_exch(old_delta, app_->default_v());
                auto& value = values[u];
                adj_list_index_t adj = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                app_->g_index_function(*graph_, u, value, delta, adj, bound_node_values);
                app_->accumulate_atomic(spnode_datas[u], delta);
                #ifdef DEBUG
                  //n_edge += adj.Size();
                  //atomic_add(n_edge, adj.Size());
                #endif
              }
              #ifdef DEBUG
                send_time_3 += GetCurrentTime();
                time_sum_3 += send_time_3;
                //LOG(INFO) << "time3/edges=" << (send_time_3/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_3_size;
              #endif
              // 3.2 bound send
              #ifdef DEBUG
                send_time_4 -= GetCurrentTime();
                last_n_edge = n_edge;
              #endif
              parallel_for(vid_t i = 0; i < node_3_size; i++){
                vertex_t u = nodes_3[i];
                value_t& old_delta = bound_node_values[u];
                auto delta = atomic_exch(old_delta, app_->default_v());
                auto& value = values[u];
                auto oes = graph_->GetOutgoingAdjList(u);
                adj_list_t adj = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                app_->accumulate_atomic(value, delta);
                #ifdef DEBUG
                  //n_edge += adj.Size();
                  //atomic_add(n_edge, adj.Size());
                #endif
              }
              #ifdef DEBUG
                send_time_4 += GetCurrentTime();
                time_sum_4 += send_time_4;
                //LOG(INFO) << "time4/edges=" << (send_time_4/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_3_size;
              #endif
            }
            #ifdef DEBUG
              LOG(INFO) << "N edge: " << n_edge << std::endl;
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
      }

      {
        #ifdef DEBUG
          auto begin = GetCurrentTime();
        #endif
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
        LOG(INFO) << "send_time_0=" << send_time_0 << " send_time_1=" << send_time_1 << " send_time_2=" << send_time_2 << " send_time_3=" << send_time_3;
      #endif

      if (termCheck(last_values, values, compr_stage) || step > 200) {
        app_->touch_nodes.clear();
        if(compr_stage){
          timer_next("correct deviation");
          corr_time -= GetCurrentTime();
          // supernode send by inner_delta and inner_value
          parallel_for(vid_t i = 0; i < cpr_->supernodes_num; i++){
            supernode_t &spnode = cpr_->supernodes[i];
            auto& oes_d = spnode.inner_delta;
            auto& oes_v = spnode.inner_value;
            auto& value = values[spnode.id];
            // auto delta = atomic_exch(spnode_datas[spnode.id], app_->default_v()); // csr
            auto& delta = spnode_datas[spnode.id];
            /* filter useless delta */
            if(delta != app_->default_v()){
              app_->g_index_func_delta(*graph_, spnode.id, value, delta, oes_d); //If the threshold is small enough when calculating the index, it can be omitted here
              app_->g_index_func_value(*graph_, spnode.id, value, delta, oes_v);
            }
            delta = app_->default_v();
          }
          #ifdef DEBUG
            LOG(INFO) << "one_step_time=" << one_step_time;
          #endif
          corr_time += GetCurrentTime();
          LOG(INFO) << "correct deviation in supernode";
          LOG(INFO) << "#1st_step: " << step;
          LOG(INFO) << "#corr_time: " << corr_time;
          compr_stage = false;
          continue;
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#iter step: " << step;
            LOG(INFO) << "#Batch time: " << exec_time << " sec";
          }
          exec_time = 0;
          corr_time = 0;
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
              double node_type_time = GetCurrentTime();
              vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
              node_type.clear();
              node_type.resize(inner_node_num, std::numeric_limits<char>::max());
              bound_node_values.clear();
              bound_node_values.Init(inner_vertices, app_->default_v());
              parallel_for(vid_t i = inner_vertices.begin().GetValue();
                  i < inner_vertices.end().GetValue(); i++) {
                    vertex_t u(i);
                  if(cpr_->Fc[u] == cpr_->FC_default_value){
                    node_type[i] = 0; // out node
                  }
                  else if(cpr_->Fc[u] >= 0){
                    node_type[i] = 2; // source node
                  }
                  else if(!cpr_->supernode_out_bound[i]){
                    node_type[i] = 4; // inner node
                  }
                  if(cpr_->supernode_out_bound[i]){
                    node_type[i] = 1; // bound node
                    if(cpr_->Fc[u] >= 0){
                      node_type[i] = 3; // source node + bound node
                    }
                  }
              }
              all_nodes.clear();
              all_nodes.resize(5);
              for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
                  all_nodes[node_type[i]].emplace_back(vertex_t(i));
              }
              LOG(INFO) << "node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

              double  transfer_csr_time = GetCurrentTime();

              /* source to in_bound_node */
              // spnode_datas.resize(cpr_->supernodes_num);

              double  init_time_1 = GetCurrentTime();
              spnode_datas.Init(inner_vertices, 0);
              is_e_.clear();
              is_e_offset_.clear();
              ib_e_.clear();
              ib_e_offset_.clear();
              size_t source_e_num = 0;
              size_t bound_e_num = 0;
              std::vector<size_t> is_e_degree(inner_node_num+1, 0);
              std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
              LOG(INFO) << "init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179

              double csr_time_1 = GetCurrentTime();
              // for(auto u : inner_vertices){
              parallel_for(vid_t i = inner_vertices.begin().GetValue();
                  i < inner_vertices.end().GetValue(); i++) {
                vertex_t u(i);
                char type = node_type[i];
                if(type == 2 || type == 3){
                  supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
                  // source_e_num += spnode.bound_delta.size();
                  is_e_degree[i+1] = spnode.bound_delta.size();
                  atomic_add(source_e_num, spnode.bound_delta.size());
                }
                if(type == 1 || type == 3){
                  auto oes = graph_->GetOutgoingAdjList(u);
                  auto it = oes.begin();
                  auto out_degree = oes.Size();
                  vid_t ids_id = cpr_->id2spids[u];
                  size_t temp_cnt = 0;
                  for(vid_t j = 0; j < out_degree; j++){  // 关闭多线程
                    auto& e = *(it + j);
                    if(ids_id != cpr_->id2spids[e.neighbor]){
                      // bound_e_num += 1;
                      temp_cnt += 1;
                    }
                  }
                  ib_e_degree[i+1] += temp_cnt;
                  atomic_add(bound_e_num, temp_cnt);
                }
              }
              LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

              /* get index start */
              double index_time = GetCurrentTime();
              for(vid_t i = 1; i < inner_node_num; i++) {
                ib_e_degree[i] += ib_e_degree[i-1];
                is_e_degree[i] += is_e_degree[i-1];
              }
              LOG(INFO) << "index_time=" << (GetCurrentTime()-index_time); //0.226317

              LOG(INFO) << "inner_node_num=" << inner_node_num;
              LOG(INFO) << "inner_node_num=" << graph_->GetVerticesNum();

              double init_time_2 = GetCurrentTime();
              is_e_.resize(source_e_num);
              ib_e_.resize(bound_e_num);
              is_e_offset_.resize(inner_node_num+1);
              ib_e_offset_.resize(inner_node_num+1);
              LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

              // build edge
              double csr_time_2 = GetCurrentTime();
              // for(auto u : inner_vertices){
              parallel_for(vid_t i = inner_vertices.begin().GetValue();
                  i < inner_vertices.end().GetValue(); i++) {
                vertex_t u(i);
                /* source node */
                vid_t index_s = is_e_degree[i];
                is_e_offset_[u.GetValue()] = &is_e_[index_s];
                char type = node_type[u.GetValue()];
                if(type == 2 || type == 3){
                  supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
                  for(auto& oe : spnode.bound_delta){
                    // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
                    is_e_[index_s].neighbor = oe.first;
                    is_e_[index_s].data = oe.second;
                    index_s++;
                  }
                }
                /* inner_bound node */
                vid_t index_b = ib_e_degree[i];
                ib_e_offset_[u.GetValue()] = &ib_e_[index_b];
                if(type == 1 || type == 3){
                  auto oes = graph_->GetOutgoingAdjList(u);
                  auto it = oes.begin();
                  auto out_degree = oes.Size();
                  vid_t ids_id = cpr_->id2spids[u];
                  for(vid_t j = 0; j < out_degree; j++){
                    auto& e = *(it + j);
                    if(ids_id != cpr_->id2spids[e.neighbor]){
                      ib_e_[index_b] = e;
                      index_b++;
                    }
                  }
                }
              }
              is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
              ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
              LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2);

              LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time);
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
                 VertexArray<value_t, vid_t>& values, bool compr_stage) {
    terminate_checking_time_ -= GetCurrentTime();
    auto vertices = graph_->InnerVertices();
    double diff_sum = 0, global_diff_sum;

    for (auto u : vertices) {
      diff_sum += fabs(last_values[u] - values[u]);
      last_values[u] = values[u];
    }

    communicator_.template Sum(diff_sum, global_diff_sum);

    // if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
    //   LOG(INFO) << "Diff: " << global_diff_sum;
    // }

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
  /* source to in_bound_node */
  Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
  /* in_bound_node to out_bound_node */
  Array<nbr_t, Allocator<nbr_t>> ib_e_;
  Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
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

#endif  // GRAPE_WORKER_SUM_SYNC_ITER_WORKER_H_

