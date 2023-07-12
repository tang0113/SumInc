
#ifndef GRAPE_WORKER_SUM_SYNC_TRAVERSAL_WORKER_H_
#define GRAPE_WORKER_SUM_SYNC_TRAVERSAL_WORKER_H_

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
class SumSyncTraversalWorker : public ParallelEngine {
  static_assert(std::is_base_of<TraversalAppBase<typename APP_T::fragment_t,
                                                 typename APP_T::value_t>,
                                APP_T>::value,
                "SumSyncTraversalWorker should work with App");

 public:
  using fragment_t = typename APP_T::fragment_t;
  using value_t = typename APP_T::value_t;
  using delta_t = typename APP_T::delta_t;
  using vertex_t = typename APP_T::vertex_t;
  using message_manager_t = ParallelMessageManager;
  using oid_t = typename fragment_t::oid_t;
  using vid_t = typename APP_T::vid_t;
  using supernode_t = grape::SuperNodeForTrav<vertex_t, value_t, delta_t, vid_t>;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  using nbr_t = typename fragment_t::nbr_t;
  using nbr_index_t = Nbr<vid_t, delta_t>;
  using adj_list_index_t = AdjList<vid_t, delta_t>;

  SumSyncTraversalWorker(std::shared_ptr<APP_T> app,
                             std::shared_ptr<fragment_t>& graph)
      : app_(app), fragment_(graph) {}

  void Init(const CommSpec& comm_spec,
            const ParallelEngineSpec& pe_spec = DefaultParallelEngineSpec()) {
    fragment_->PrepareToRunApp(APP_T::message_strategy,
                               APP_T::need_split_edges);

    comm_spec_ = comm_spec;

    // 等待所有worker执行完毕
    MPI_Barrier(comm_spec_.comm());

    // 初始化发消息相关的buffer
    messages_.Init(comm_spec_.comm());
    messages_.InitChannels(thread_num());
    communicator_.InitCommunicator(comm_spec.comm());

    InitParallelEngine(pe_spec);
    if (FLAGS_cilk) {
      LOG(INFO) << "cilk Thread num: " << getWorkers();
    }
    LOG(INFO) << "Thread num: " << thread_num();
    
    // allocate dependency arrays
    app_->Init(comm_spec_, fragment_);
    auto deltas = app_->deltas_;
    auto values = app_->values_;

    // init compressor
    if(FLAGS_compress){
      cpr_ = new TravCompressor<APP_T, supernode_t>(app_, fragment_);
      cpr_->init(comm_spec_, communicator_, pe_spec);
      cpr_->run();
      timer_next("init app_");
      app_->reInit(cpr_->all_node_num); // for mirror node
      /* precompute supernode */
      timer_next("pre compute");
      cpr_->precompute_spnode(this->fragment_);
      timer_next("statistic");
      cpr_->statistic();
    }

  }

  void deltaCompute() {
    IncFragmentBuilder<fragment_t> inc_fragment_builder(fragment_,
                                                        FLAGS_directed);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Parsing update file";
    }
    inc_fragment_builder.Init(FLAGS_efile_update);
    auto inner_vertices = fragment_->InnerVertices();
    auto outer_vertices = fragment_->OuterVertices();

    auto deleted_edges = inc_fragment_builder.GetDeletedEdgesGid();

    #if defined(DISTRIBUTED)
      LOG(INFO) << "Distributed vision...";
      std::unordered_set<vid_t> local_gid_set;
      for (auto v : fragment_->Vertices()) {
        local_gid_set.insert(fragment_->Vertex2Gid(v));
      }
    #else
      LOG(INFO) << "Single vision...";
    #endif

    auto vertices = fragment_->Vertices();
    DenseVertexSet<vid_t> curr_modified, next_modified, reset_vertices;

    curr_modified.Init(vertices);
    next_modified.Init(vertices);
    reset_vertices.Init(inner_vertices);  // Only used for counting purpose

    double reset_time = GetCurrentTime();

    size_t pair_num = deleted_edges.size();
    // for (auto& pair : deleted_edges) {
    parallel_for(vid_t i = 0; i < pair_num; i++) {
      auto pair = deleted_edges[i];
      vid_t u_gid = pair.first, v_gid = pair.second;

      #if defined(DISTRIBUTED)
        if (local_gid_set.find(u_gid) != local_gid_set.end() &&
            fragment_->IsInnerGid(v_gid)) {
          vertex_t u, v;
          CHECK(fragment_->Gid2Vertex(u_gid, u));
          CHECK(fragment_->Gid2Vertex(v_gid, v));

          auto parent_gid = app_->DeltaParentGid(v);

          if (parent_gid == u_gid) {
            curr_modified.Insert(v);
          }
        }
      #else
        vertex_t u, v;
        CHECK(fragment_->Gid2Vertex(u_gid, u));
        CHECK(fragment_->Gid2Vertex(v_gid, v));

        auto parent_gid = app_->DeltaParentGid(v);
        if (parent_gid == u_gid) {
          curr_modified.Insert(v);
        }
      #endif
    }

    auto& channels = messages_.Channels();

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Resetting";
    }

    do {
      #if defined(DISTRIBUTED)
      messages_.StartARound();
      messages_.ParallelProcess<fragment_t, grape::EmptyType>(
          thread_num(), *fragment_,
          [&curr_modified](int tid, vertex_t v, const grape::EmptyType& msg) {
            curr_modified.Insert(v);
          });
      #endif

      // ForEachSimple(curr_modified, inner_vertices,
      ForEachCilkOfBitset(curr_modified, inner_vertices,
                    [this, &next_modified, &reset_vertices](int tid, vertex_t u) {
                      auto u_gid = fragment_->Vertex2Gid(u);
                      auto oes = fragment_->GetOutgoingAdjList(u);

                      // for (auto e : oes) {
                      //   auto v = e.neighbor;
                      //   if (app_->DeltaParentGid(v) == u_gid && u != v) { // 注意考虑自环，且是source->source
                      //     next_modified.Insert(v);
                      //   }
                      // }

                      auto out_degree = oes.Size();
                      auto it = oes.begin();
                      granular_for(j, 0, out_degree, (out_degree > 1024), {
                        auto& e = *(it + j);
                        auto v = e.neighbor;
                        if (app_->DeltaParentGid(v) == u_gid && u != v) { // 注意考虑自环，且是source->source
                          next_modified.Insert(v);
                        }
                      })

                      // app_->values_[u] = app_->GetIdentityElement();
                      // app_->deltas_[u].Reset(app_->GetIdentityElement()); 
                      app_->values_[u] = app_->GetInitValue(u);
                      app_->deltas_[u] = app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
                      app_->CombineValueDelta(app_->values_[u], app_->deltas_[u]);

                      reset_vertices.Insert(u); // just count reset node!
                    });

      #if defined(DISTRIBUTED)
      // ForEachSimple(curr_modified, inner_vertices,
      ForEachCilkOfBitset(curr_modified, inner_vertices,
                    [this, &reset_vertices](int tid, vertex_t u) {
                      app_->values_[u] = app_->GetInitValue(u);
                      app_->deltas_[u] = app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
                      app_->CombineValueDelta(app_->values_[u], app_->deltas_[u]);

                      reset_vertices.Insert(u);
                    });
      #endif

      #if defined(DISTRIBUTED)
      // ForEach(next_modified, outer_vertices,
      ForEachCilkOfBitset(next_modified, outer_vertices,
              [&channels, this](int tid, vertex_t v) {
                grape::EmptyType dummy;
                channels[tid].SyncStateOnOuterVertex(*fragment_, v, dummy);
                // app_->deltas_[v].Reset(app_->GetIdentityElement());
                app_->deltas_[v] = app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
              });
      messages_.FinishARound();
      // if (next_modified.ParallelCount(thread_num()) > 0) {
      if (next_modified.PartialEmpty(0, fragment_->GetVerticesNum()) {
        messages_.ForceContinue();
      }
      #endif

      curr_modified.Clear();
      curr_modified.Swap(next_modified);
      // LOG(INFO) << "  next_modified.size=" << next_modified.ParallelCount(8);
      // LOG(INFO) << "  curr_modified.size=" << curr_modified.ParallelCount(8);
    #if defined(DISTRIBUTED)
    } while (!messages_.ToTerminate());
    #else
    } while (curr_modified.ParallelCount(thread_num()) > 0);
    #endif

    LOG(INFO) << "#reset_time: " << (GetCurrentTime() - reset_time);
    print_active_edge("#reset");

    // #if defined(DISTRIBUTED)
    size_t n_reset = 0, local_n_reset = reset_vertices.Count();
    Communicator communicator;
    communicator.InitCommunicator(comm_spec_.comm());
    communicator.template Sum(local_n_reset, n_reset);
    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "# of reset vertices: " << n_reset << " reset percent: "
                << (float) n_reset / fragment_->GetTotalVerticesNum();
      LOG(INFO) << "Start a round from all vertices";
    }
    // #endif

    // We have to use hashmap to keep delta because the outer vertices may
    // change
    VertexArray<value_t, vid_t> values;
    VertexArray<delta_t, vid_t> deltas;

    values.Init(inner_vertices);
    deltas.Init(inner_vertices);

    for (auto v : inner_vertices) {
      values[v] = app_->values_[v];
      deltas[v] = app_->deltas_[v];
    }

    // fragment_ = inc_fragment_builder.Build();

    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    if(FLAGS_compress){
      auto added_edges = inc_fragment_builder.GetAddedEdgesGid();
      cpr_->inc_run(deleted_edges, added_edges, new_graph);
      print_active_edge("#inc_run_cmpIndex");
    }
    fragment_ = new_graph;

    // Important!!! outer vertices may change, we should acquire it after new
    // graph is loaded
    outer_vertices = fragment_->OuterVertices();
    // Reset all states, active vertices will be marked in curr_modified_
    app_->Init(comm_spec_, fragment_);
    if (FLAGS_compress) {
      app_->reInit(cpr_->all_node_num); // for mirror node
    }

    // copy to new graph
    for (auto v : inner_vertices) {
      app_->values_[v] = values[v];
      app_->deltas_[v] = deltas[v];
    }

    // Start a round without any condition
    double resend_time = GetCurrentTime();
    vid_t inner_node_num = inner_vertices.end().GetValue() 
                           - inner_vertices.begin().GetValue();
    parallel_for(vid_t i = 0; i < inner_node_num; i++) {
      vertex_t u(i);
      auto& value = app_->values_[u];
      auto& delta = app_->deltas_[u];

      if (delta.value != app_->GetIdentityElement()) {
        app_->Compute(u, value, delta, next_modified);
      }
    }

    #if defined(DISTRIBUTED)
    messages_.StartARound();
    ForEach(
        next_modified, outer_vertices, [&channels, this](int tid, vertex_t v) {
          auto& delta_to_send = app_->deltas_[v];
          if (delta_to_send.value != app_->GetIdentityElement()) {
            channels[tid].SyncStateOnOuterVertex(*fragment_, v, delta_to_send);
          }
        });
    messages_.FinishARound();
    #endif
    app_->curr_modified_.Swap(next_modified);
    LOG(INFO) << "#resend_time: " << (GetCurrentTime() - resend_time);
    print_active_edge("#resend");

    LOG(INFO) << " app_->curr_modified_.size()=" << app_->curr_modified_.ParallelCount(thread_num());;
  }

    /**
   * Get the threshold by sampling
   * sample_size: Sample size
   * range: the range of the variable population
   * return threshold
   */
   value_t Scheduled(const vid_t sample_size, const VertexRange<vid_t>& range) {
     vid_t begin = range.begin().GetValue();
     vid_t end = range.end().GetValue();
     vid_t all_size = end - begin;
     if (all_size <= sample_size) {
       return app_->GetIdentityElement();
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
         value_t pri = app_->GetPriority(u, app_->values_[u], app_->deltas_[u]);
         sample.emplace_back(fabs(pri));
       }
  
       sort(sample.begin(), sample.end());
       int cut_index = sample_size * (1 - FLAGS_portion);  // Select the threshold position 
       return sample[cut_index];
     }
   }

  void GetInnerIndex(){
    LOG(INFO) << "Build inner index's csr: source to inner_node";
    double start_time = GetCurrentTime();
    vid_t supernode_num = cpr_->supernodes_num;
    std::vector<size_t> degree(supernode_num+1, 0);
    source_nodes.resize(supernode_num);

    parallel_for (vid_t i = 0; i < supernode_num; i++) {
      supernode_t &spnode = cpr_->supernodes[i];
      source_nodes[i] = spnode.id;
      degree[i+1] = spnode.inner_delta.size();
    }

    for(vid_t i = 1; i <= supernode_num; i++) {
      degree[i] += degree[i-1];
    }
    size_t index_num = degree[supernode_num];
    is_iindex_offset_.resize(supernode_num+1);
    is_iindex_.resize(index_num);


    parallel_for(vid_t j = 0; j < supernode_num; j++){
      supernode_t &spnode = cpr_->supernodes[j];
      auto& oes = spnode.inner_delta;
      vid_t index = degree[j];
      is_iindex_offset_[j] = &is_iindex_[index];
      for (auto& oe : oes) {
        is_iindex_[index] = nbr_index_t(oe.first.GetValue(), oe.second);
        index++;
      }
    }
    is_iindex_offset_[supernode_num] = &is_iindex_[index_num];

    parallel_for (vid_t i = 0; i < supernode_num; ++i) {
      std::sort(is_iindex_offset_[i], is_iindex_offset_[i + 1],
              [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
              });
    }

    LOG(INFO) << "Build inner index's csr: cost_time=" << (GetCurrentTime() - start_time);
  }

  void first_step(VertexArray<value_t, vid_t>& values_temp,
                  VertexArray<delta_t, vid_t>& deltas_temp,
                  double& exec_time, bool is_inc = false) {
    double extra_all_time = GetCurrentTime();
    auto inner_vertices = fragment_->InnerVertices();
    vid_t inner_node_num = inner_vertices.end().GetValue() 
                            - inner_vertices.begin().GetValue();
    
    cpr_->get_nodetype(inner_node_num, node_type);

    if (is_inc == true) {
      timer_next("inc pre compute");
      double inc_pre_compute = GetCurrentTime();
      // cpr_->precompute_spnode(fragment_);
      // cpr_->precompute_spnode_all(fragment_); // 应该是被touch到的超点需要
      cpr_->inc_precompute_spnode_mirror(fragment_, node_type); //应该放在更新node_type之后!!!
      inc_pre_compute = GetCurrentTime() - inc_pre_compute;
      LOG(INFO) << "#inc_pre_compute: " << inc_pre_compute;
    }

    //debug
    {
      std::vector<int> cnt;
      cnt.resize(5, 0);
      vertex_t one_inner_node(0);
      for(vid_t i = 0; i < cpr_->supernode_out_mirror.size(); i++) { // can'nt parallel
        for(auto v : cpr_->supernode_out_mirror[i]) {
          cnt[node_type[v.GetValue()]]++;
        }
      }
      for (auto c : cnt) {
        LOG(INFO) << "out_mirror_type_cnt=" << c;
      }
    }

    cpr_->sketch2csr(inner_node_num, node_type, all_nodes, is_e_, is_e_offset_,
                              ib_e_, ib_e_offset_);

    // for(auto t : node_range) {
    //   LOG(INFO) << " node_type_range=" << t;
    // }

    // {
      // check_result("init before");
      exec_time -= GetCurrentTime();
      // app_->next_modified_.Swap(app_->curr_modified_);
      // Update the source id to the new id
      vertex_t source;
      bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
      if (native_source) {
        // vid_t new_source_id = oldId2newId[source.GetValue()];
        // app_->curr_modified_.Insert(vertex_t(new_source_id));
        app_->curr_modified_.Insert(source); // old source node
        // LOG(INFO) << "supernode... newid=" << new_source_id 
                  // << " type4=" << node_range[4];
        LOG(INFO) << "this->Fc[source]=" << cpr_->Fc[source];
      }

    //   LOG(INFO) << "Send one round for supernode...";
    //   LOG(INFO) << "  active_node.size=" 
    //             << app_->curr_modified_.ParallelCount(8);
    //   /* send one round */
    //   ForEachCilkOfBitset(
    //     app_->curr_modified_, fragment_->InnerVertices(), 
    //     [this](int tid, vertex_t u) {
    //       // LOG(INFO) << " ---------u.oid" << cpr_->v2Oid(u);
    //       // u = vertex_t(oldId2newId[u.GetValue()]);
    //       if (node_type[u.GetValue()] < 2) { // 0, 1
    //         auto& delta = app_->deltas_[u];
    //         if (delta.value != app_->GetIdentityElement()) {
    //           auto& value = app_->values_[u];
    //           app_->CombineValueDelta(value, delta);
    //           adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
    //           app_->Compute(u, value, delta, oes, app_->next_modified_);
    //         }
    //       } else if (node_type[u.GetValue()] < 3) { // 2
    //         auto& delta = app_->deltas_[u];
    //         if (delta.value != app_->GetIdentityElement()) {
    //           auto& value = app_->values_[u];
    //           app_->CombineValueDelta(value, delta);
    //           adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
    //           app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
    //         }
    //       } else if (node_type[u.GetValue()] < 4) { // 3
    //         auto& delta = app_->deltas_[u];
    //         if (delta.value != app_->GetIdentityElement()) {
    //           auto& value = app_->values_[u];
    //           app_->CombineValueDelta(value, delta);
    //           /* 1: bound node */
    //           adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
    //           app_->Compute(u, value, delta, oes_b, app_->next_modified_);
    //           /* 2: source node */
    //           adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
    //           app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->next_modified_);
    //         }
    //       }
    //   });
    //   app_->next_modified_.Swap(app_->curr_modified_);
    //   exec_time += GetCurrentTime();
    //   LOG(INFO) << " pre_exec_time=" << exec_time;
    //   LOG(INFO) << " init after bitset.size=" << app_->curr_modified_.ParallelCount(thread_num());
    //   LOG(INFO) << " init after curr_modified_new.size=" 
    //             << app_->curr_modified_.ParallelCount(thread_num());
    // }
    
    // LOG(INFO) << "extra_all_time=" << (GetCurrentTime()- extra_all_time);
    // print_active_edge("#pre_exec");
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());

    // debug;
    if (false) {
      // 统计下最大出度和入度顶点
      {
        LOG(INFO) << "+++++++++++++++++++++++++++++++++++++++++";
        vid_t max_out_id = 0;
        vid_t max_in_id = 0;
        vid_t max_out_degree = 0;
        vid_t max_in_degree = 0;
        for (auto v : this->fragment_->InnerVertices()) {
          const auto& oes = this->fragment_->GetOutgoingAdjList(v);
          if (oes.Size() > max_out_degree) {
            max_out_degree = oes.Size();
            max_out_id = v.GetValue();
          }
          const auto& oes_in = this->fragment_->GetIncomingAdjList(v);
          if (oes_in.Size() > max_in_degree) {
            max_in_degree = oes_in.Size();
            max_in_id = v.GetValue();
          }
        }
        LOG(INFO) << "max_out_degree=" << max_out_degree
                  << " max_out_id=" << this->fragment_->GetId(vertex_t(max_out_id));
        LOG(INFO) << "max_in_degree=" << max_in_degree
                  << " max_in_id=" << this->fragment_->GetId(vertex_t(max_in_id));
        LOG(INFO) << "+++++++++++++++++++++++++++++++++++++++++";
      }

      // expr: motivate mirror-master
      //  统计cluster内部每个边界点(入口点、出口点、入口+出口点)在其指向/被指向其它同一个clust的边数
      if (true) {
        LOG(INFO) << "########################################################";
        LOG(INFO) << " expr: motivate mirror-master";
        std::string digest = FLAGS_efile;
        std::replace(digest.begin(), digest.end(), '/', '_');
        std::string save_path = "./out_edge_mirror_k_" 
                            + std::to_string(FLAGS_mirror_k)
                            + "_cmpthreshold_"
                            + std::to_string(FLAGS_compress_threshold) + "_cid"
                            + digest;
        LOG(INFO) << " save_path=" << save_path;
        std::ofstream fout("./out_edge_mirror_k_" 
                            + std::to_string(FLAGS_mirror_k)
                            + "_cmpthreshold_"
                            + std::to_string(FLAGS_compress_threshold) + "_cid"
                            + digest);
        /*
        // 统计每个cluster中顶点指向每个cluster的边数
        //     不包括指向自己的边；
        //     <vertex, (to_cid, edges), (to_cid, edges) ...>
        //     <degree, (to_cid, edges), (to_cid, edges) ...>
        for (vid_t ids_id = 0; ids_id < cpr_->supernode_ids.size(); ids_id++) {
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[ids_id]; 
          std::unordered_set<vertex_t> &out_mirror = 
                                              cpr_->supernode_out_mirror[ids_id];
          // LOG(INFO) << "---------------------------";
          // LOG(INFO) << " cluster=" << ids_id;
          
          for(auto v : node_set){
            std::unordered_map<vid_t, size_t> edge_count;
            const auto& oes = fragment_->GetOutgoingAdjList(v);
            size_t out_edge_num = 0;
            for(auto& e : oes){
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if(to_ids != ids_id && to_ids != cpr_->ID_default_value
                  && out_mirror.find(e.neighbor) == out_mirror.end()){ // 导致入口Mirror成为内部点
                  std::unordered_set<vertex_t> &in_mirror = 
                                              cpr_->supernode_in_mirror[to_ids];
                if (in_mirror.find(v) == in_mirror.end()) {
                  edge_count[to_ids] += 1;
                  out_edge_num += 1;
                }
              }
              // LOG(INFO) << cpr_->v2Oid(v) << "->" << cpr_->v2Oid(e.neighbor);
            }
            if (edge_count.size() > 0) {
              fout << out_edge_num << ":";
              for(const auto& pair : edge_count) {
                fout << " " << pair.first << "," << pair.second;
                // LOG(INFO) << "   get" << pair.first << ": " << pair.second;
              }
              fout << "\n";
            }
          }
        }*/

        // 统计每个出口/入口点指向/来自同一个cluster的情况：
        //     <v, 出边点指向同一个cluster的边数(不包括内部边)>
        //     <v, 入边点指向同一个cluster的边数(不包括内部边)>
        // 统计时，需要将mirror_k设置为最大，模拟不加Mirror的情况下进行统计
        /*
        for (vid_t ids_id = 0; ids_id < cpr_->supernode_ids.size(); ids_id++) {
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[ids_id]; 
          std::unordered_set<vertex_t> &out_mirror = 
                                              cpr_->supernode_out_mirror[ids_id];
          // LOG(INFO) << "---------------------------";
          // LOG(INFO) << " cluster=" << ids_id;
          for(auto v : node_set){
            std::unordered_map<vid_t, size_t> edge_count;
            const auto& oes = fragment_->GetOutgoingAdjList(v);
            size_t out_edge_num = 0;
            for(auto& e : oes){
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if(to_ids != ids_id && to_ids != cpr_->ID_default_value){
                edge_count[to_ids] += 1;
              }
              // LOG(INFO) << cpr_->v2Oid(v) << "->" << cpr_->v2Oid(e.neighbor);
            }
            if (edge_count.size() > 0) {
              for(const auto& pair : edge_count) {
                fout << pair.second << " "; // 每个点指向同一个cluster的出边数
                // LOG(INFO) << "   get" << pair.first << ": " << pair.second;
              }
              // fout << "\n";
            }
          }
        }
        */

        // 统计每个出口/入口点指向/来自同一个cluster的情况：
        //     <v, 出边点指向同一个cluster的边数(不包括内部边)>
        //     <v, 入边点指向同一个cluster的边数(不包括内部边)>
        // 统计时，需要将mirror_k设置为最大，模拟不加Mirror的情况下进行统计
        for (vid_t ids_id = 0; ids_id < cpr_->supernode_ids.size(); ids_id++) {
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[ids_id]; 
          std::unordered_set<vertex_t> &out_mirror = 
                                              cpr_->supernode_out_mirror[ids_id];
          for(auto v : node_set){
            std::unordered_map<vid_t, size_t> edge_count;
            const auto& oes = fragment_->GetOutgoingAdjList(v);
            size_t out_edge_num = 0;
            for(auto& e : oes){
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if(to_ids != ids_id && to_ids != cpr_->ID_default_value){
                edge_count[to_ids] += 1;
              }
              // LOG(INFO) << cpr_->v2Oid(v) << "->" << cpr_->v2Oid(e.neighbor);
            }
            if (edge_count.size() > 0) {
              for(const auto& pair : edge_count) {
                fout << pair.second << " "; // 每个点指向同一个cluster的出边数
                // LOG(INFO) << "   get" << pair.first << ": " << pair.second;
              }
              // fout << "\n";
            }
          }
        }

        fout.close();
        LOG(INFO) << " finish out_edge_mirror_k_" << FLAGS_mirror_k;
      }
    }

    // // allocate dependency arrays
    // app_->Init(comm_spec_, fragment_);
    int step = 0;
    bool batch_stage = true;
    double exec_time = 0;
    double corr_time = 0;
    bool compr_stage = FLAGS_compress; // true: supernode send
    VertexArray<value_t, vid_t> values_temp;
    VertexArray<delta_t, vid_t> deltas_temp;
    values_temp.Init(fragment_->InnerVertices());
    deltas_temp.Init(fragment_->InnerVertices());
    fid_t fid = fragment_->fid();
    auto vm_ptr = fragment_->vm_ptr();

// debug
// #define DEBUG
    if (compr_stage == true) {
      // app_->next_modified_.Swap(app_->curr_modified_);
      if(!FLAGS_gpu_start){
        first_step(values_temp, deltas_temp, exec_time, false);
      }else{
        double extra_all_time = GetCurrentTime();
        auto inner_vertices = fragment_->InnerVertices();
        vid_t inner_node_num = inner_vertices.end().GetValue() 
                            - inner_vertices.begin().GetValue();
    
        cpr_->get_nodetype(inner_node_num, node_type);
        cpr_->sketch2csr(inner_node_num, node_type, all_nodes, is_e_, is_e_offset_,
                              ib_e_, ib_e_offset_);
        vertex_t source;
        bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
        if (native_source) {
          app_->curr_modified_.Insert(source);
        }
        // first_step(values_temp, deltas_temp, exec_time, false);
      }
    }

    if (compr_stage == false) {
      vertex_t source;
      bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
      if (native_source) {
        app_->curr_modified_.Insert(source);
      }
    }

    LOG(INFO) << "compr_stage=" << compr_stage;

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    
    long long count = 0; // debug
    double for_time = 0.d;
    double clean_bitset_time = 0.d;
    value_t threshold;
    if (FLAGS_portion != 1) {
        LOG(INFO) << "Use priority... portion=" << FLAGS_portion;
    }

    // debug
    {
      LOG(INFO) << "---------------------while before------------------------";
      LOG(INFO) << "step=" << step << " f_send_delta_num=" << app_->f_send_delta_num;
      LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
      app_->f_send_delta_num = 0;
      app_->node_update_num = 0;
      app_->touch_nodes.ParallelClear(8);
      LOG(INFO) << "---------------------end------------------------";
    }

    auto oeoffset = fragment_->getOeoffset();
    vid_t num = fragment_->InnerVertices().size();

    //Ingress,使用gpu,分配内存,传输数据,初始化

    vid_t *size_oe_d, *size_oe_h = (vid_t *)malloc(sizeof(vid_t) * num);//Ingress,用于记录每一个顶点的邻居数
    vid_t *size_ib_d, *size_ib_h = (vid_t *)malloc(sizeof(vid_t) * num);//SumInc,node type:SingleNode
    vid_t *size_is_d, *size_is_h = (vid_t *)malloc(sizeof(vid_t) * num);//SumInc,node type:OnlyInNode

    vid_t *cur_oeoff_d, *cur_oeoff_h = (vid_t *)malloc(sizeof(vid_t) * num);//Ingress,用于记录每一个顶点在邻居大链表中开始的偏移量
    vid_t *cur_iboff_d, *cur_iboff_h = (vid_t *)malloc(sizeof(vid_t) * num);
    vid_t *cur_isoff_d, *cur_isoff_h = (vid_t *)malloc(sizeof(vid_t) * num);

    vid_t *cur_modified_d, *next_modified_d;
    vid_t *cur_modified_size_d, *next_modified_size_d;
    vid_t *cur_modified_size_h = (vid_t *)malloc(sizeof(vid_t) * 1);
    vid_t *is_modified_d;//判断当前顶点是否被修改

    unsigned int oe_offsize = 0;//临时变量
    for(int i = 0;i < num; i++){//Ingress
      cur_oeoff_h[i] = oe_offsize;
      oe_offsize += oeoffset[i+1] - oeoffset[i];
      size_oe_h[i] = oeoffset[i+1] - oeoffset[i];
    }

    unsigned int ib_offsize = 0;
    if(compr_stage){
      for(int i = 0;i < num;i++){//SumInc
        cur_iboff_h[i] = ib_offsize;
        ib_offsize += ib_e_offset_[i+1] - ib_e_offset_[i];
        size_ib_h[i] = ib_e_offset_[i+1] - ib_e_offset_[i];
      }
    }
    
    unsigned int is_offsize = 0;
    if(compr_stage){
      for(int i=0;i < num;i++){
        cur_isoff_h[i] = is_offsize;
        is_offsize += is_e_offset_[i+1] - is_e_offset_[i];
        size_is_h[i] = is_e_offset_[i+1] - is_e_offset_[i];
      }
    }

    auto &values = app_->values_;
    auto &deltas = app_->deltas_;
    value_t *deltas_d, *deltas_h = (value_t *)malloc(sizeof(value_t) * num);
    value_t *values_d;

    vid_t *oeoffset_d, *oeoffset_h = (vid_t *)malloc(sizeof(vid_t) * oe_offsize);//Ingress,记录每个顶点的邻居，形成一条链表
    vid_t *iboffset_d, *iboffset_h = (vid_t *)malloc(sizeof(vid_t) * ib_offsize);//SumInc
    vid_t *isoffset_d, *isoffset_h = (vid_t *)malloc(sizeof(vid_t) * is_offsize);//SumInc

    value_t *oe_edata_d, *oe_edata_h = (value_t *)malloc(sizeof(value_t) * oe_offsize);
    value_t *ib_edata_d, *ib_edata_h = (value_t *)malloc(sizeof(value_t) * ib_offsize);
    value_t *is_edata_d, *is_edata_h = (value_t *)malloc(sizeof(value_t) * is_offsize);
    char *node_type_d, *node_type_h = (char *)malloc(sizeof(char) * num);//SumInc,记录每个顶点的类型



    cudaSetDevice(0);
    //deltas和values
    cudaMalloc(&deltas_d, sizeof(value_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    cudaMalloc(&values_d, sizeof(value_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    check();
    //邻居大列表,所有点的邻接表拼接而成
    cudaMalloc(&oeoffset_d, sizeof(vid_t) * oe_offsize);
    cudaMalloc(&iboffset_d, sizeof(vid_t) * ib_offsize);
    cudaMalloc(&isoffset_d, sizeof(vid_t) * is_offsize);
    LOG(INFO) << "delta size is "<<deltas.size();
    LOG(INFO) << "valuesize is "<<values.size();
    LOG(INFO) << "is size is "<<is_offsize;
    check();
    //边数据
    cudaMalloc(&oe_edata_d, sizeof(value_t) * oe_offsize);
    cudaMalloc(&ib_edata_d, sizeof(value_t) * ib_offsize);
    cudaMalloc(&is_edata_d, sizeof(value_t) * is_offsize);
    check();
    //记录每个点的邻接表在其邻居大列表中的起始位置
    cudaMalloc(&cur_oeoff_d, sizeof(vid_t) * num);
    cudaMalloc(&cur_iboff_d, sizeof(vid_t) * num);
    cudaMalloc(&cur_isoff_d, sizeof(vid_t) * num);
    check();
    //记录每个点的邻居数量
    cudaMalloc(&size_oe_d, sizeof(vid_t) * num);
    cudaMalloc(&size_ib_d, sizeof(vid_t) * num);
    cudaMalloc(&size_is_d, sizeof(vid_t) * num);
    check();
    //顶点类型
    cudaMalloc(&node_type_d, sizeof(char) * num);
    //当前要修改的点,下一个要修改的点
    cudaMalloc(&cur_modified_d, sizeof(vid_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    cudaMalloc(&next_modified_d, sizeof(vid_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    check();
    //当前要修改的点数量
    cudaMalloc(&cur_modified_size_d, sizeof(vid_t) * 1);
    //下一次每个顶点要加入修改的目的顶点数量,设置为num目的是使用GPU时防止多个线程对全局变量同时进行修改
    cudaMalloc(&next_modified_size_d, sizeof(vid_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    cudaMalloc(&is_modified_d, sizeof(vid_t) * (FLAGS_compress ? cpr_->all_node_num : num));
    check();

    unsigned int oe_curIndex = 0, ib_curIndex = 0, is_curIndex = 0;

    //根据压缩或者不压缩进行不同的初始化
    if(FLAGS_compress){
      for(int i = 0,k = 0; i < num; i++,k++){
        if(k < num)//启用压缩时node_type才有效
          node_type_h[k] = node_type[k];
        if(k < num){
          for(int j = 0;j < size_oe_h[k]; j++){
            value_t* temp = reinterpret_cast<value_t*>(&oeoffset[i][j].data);//强制转换,原类型为empty不能直接用
            oe_edata_h[oe_curIndex] = *temp;
            oeoffset_h[oe_curIndex++] = oeoffset[i][j].neighbor.GetValue();
          }
        }
        for(int j = 0;j < size_is_h[i]; j++){
          is_edata_h[is_curIndex] = is_e_offset_[i][j].data.value;
          isoffset_h[is_curIndex++] = is_e_offset_[i][j].neighbor.GetValue();
        }
        for(int j = 0;j < size_ib_h[i];j++){
          value_t* temp = reinterpret_cast<value_t*>(&ib_e_offset_[i][j].data);//强制转换,原类型为empty不能直接用
          ib_edata_h[ib_curIndex] = *temp;
          iboffset_h[ib_curIndex++] = ib_e_offset_[i][j].neighbor.GetValue();
        }
      }
    }
    if(!FLAGS_compress){
      for(int i = 0; i < num; i++){
        for(int j = 0;j < size_oe_h[i]; j++){
            value_t* temp = reinterpret_cast<value_t*>(&oeoffset[i][j].data);//强制转换,原类型为empty不能直接用
            oe_edata_h[oe_curIndex] = *temp;
            oeoffset_h[oe_curIndex++] = oeoffset[i][j].neighbor.GetValue();
          }
      }
    }
    
    
    values.fake2buffer();
    deltas.fake2buffer();
    for(int i = 0;i < num;i++){
      deltas_h[i] = deltas.data_buffer[i].value;
    }
    cudaMemcpy(oeoffset_d, oeoffset_h, sizeof(vid_t) * oe_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(iboffset_d, iboffset_h, sizeof(vid_t) * ib_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(isoffset_d, isoffset_h, sizeof(vid_t) * is_offsize, cudaMemcpyHostToDevice);

    cudaMemcpy(oe_edata_d, oe_edata_h, sizeof(value_t) * oe_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(ib_edata_d, ib_edata_h, sizeof(value_t) * ib_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(is_edata_d, is_edata_h, sizeof(value_t) * is_offsize, cudaMemcpyHostToDevice);

    cudaMemcpy(cur_oeoff_d, cur_oeoff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_iboff_d, cur_iboff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_isoff_d, cur_isoff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);

    cudaMemcpy(deltas_d, deltas_h, sizeof(value_t) * (FLAGS_compress ? cpr_->all_node_num : num), cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * (FLAGS_compress ? cpr_->all_node_num : num), cudaMemcpyHostToDevice);

    cudaMemcpy(size_oe_d, size_oe_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(size_ib_d, size_ib_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(size_is_d, size_is_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    check();
    if(FLAGS_gpu_start && FLAGS_compress){//SumInc

    }
    LOG(INFO) <<"cur size is "<< app_->curr_modified_.Count();
    vertex_t u(0);
    LOG(INFO) << "cur node is" << app_->curr_modified_.Exist(u);
    unsigned int sssp_source = FLAGS_sssp_source;
    if(!app_->curr_modified_.Exist(vertex_t(sssp_source))){
      for(int i = 0;i<num;i++){
        vertex_t u(sssp_source+i);
        vertex_t v(sssp_source-i);
        if(app_->curr_modified_.Exist(v)){
          sssp_source -= i;
          break;
        }
        if(app_->curr_modified_.Exist(u)){
          sssp_source += i;
          break;
        }
      }
    }
    LOG(INFO) << "source is "<<sssp_source;
    tjnsssp::init(oeoffset_d, oe_edata_d, cur_oeoff_d, deltas_d, values_d, size_oe_d, sssp_source, 
                  cur_modified_d, cur_modified_size_d, is_modified_d, (FLAGS_compress ? cpr_->all_node_num : num),
                  iboffset_d, ib_edata_d, cur_iboff_d, size_ib_d, 
                  isoffset_d, is_edata_d, cur_isoff_d, size_is_d, 
                  node_type_d);
                  check();
    check();

    while (true) {
      // LOG(INFO) << "step=" << step << " curr_modified_.size()=" << app_->curr_modified_.ParallelCount(8);
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
      // Traverse outgoing neighbors
      // for_time -= GetCurrentTime();
      if (FLAGS_cilk) {
        if(compr_stage == false){
          // ForEachCilk(
          if(!FLAGS_gpu_start){
            ForEachCilkOfBitset(
              app_->curr_modified_, inner_vertices, [this, &compr_stage, &count, &step](int tid, vertex_t u) {
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];
                // LOG(INFO) << "--- step=" << step << " oid=" << fragment_->GetId(u) << " id=" << u.GetValue() << ": value=" << value << " delta=" << delta.value<<"cur modi"<<u.GetValue();
                if (app_->CombineValueDelta(value, delta)) {
                  app_->Compute(u, last_value, delta, app_->next_modified_);
                  // debug
                  { 
                    #ifdef DEBUG
                    auto oes = fragment_->GetOutgoingAdjList(u);
                    count += oes.Size();
                    #endif
                  }
                }
              });
          }

          if(FLAGS_gpu_start){
            cudaMemcpy(cur_modified_size_h, cur_modified_size_d, sizeof(vid_t) * 1, cudaMemcpyDeviceToHost);
            // check();
            tjnsssp::g_function(cur_modified_size_h, num);
            // check();
            cudaMemcpy(cur_modified_size_h, cur_modified_size_d, sizeof(vid_t) * 1, cudaMemcpyDeviceToHost);
          }
          
        }
        if (compr_stage) {
          if(!FLAGS_gpu_start){
            ForEachCilkOfBitset(
            app_->curr_modified_, fragment_->InnerVertices(), 
            [this](int tid, vertex_t u) {
              if (node_type[u.GetValue()] < 2) { // 0, 1
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (app_->CombineValueDelta(value, delta)) {
                  // auto& value = app_->values_[u];
                  // app_->CombineValueDelta(value, delta);
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, app_->next_modified_);
                }
              } else if (node_type[u.GetValue()] < 3) { // 2
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (app_->CombineValueDelta(value, delta)) {
                  // auto& value = app_->values_[u];
                  // app_->CombineValueDelta(value, delta);
                  adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
                }
              } else if (node_type[u.GetValue()] < 4) { // 3
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (app_->CombineValueDelta(value, delta)) {
                  // auto& value = app_->values_[u];
                  // app_->CombineValueDelta(value, delta);
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, app_->next_modified_);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->next_modified_);
                }
              }
            });
          }
          if(FLAGS_gpu_start){
            cudaMemcpy(cur_modified_size_h, cur_modified_size_d, sizeof(vid_t) * 1, cudaMemcpyDeviceToHost);
            tjnsssp::g_function_compr(cur_modified_size_h, cpr_->all_node_num);
          }
          
        }
      } else {
        if (compr_stage == false) {
          ForEach(
            app_->curr_modified_, inner_vertices, [this, &compr_stage, &count](int tid, vertex_t u) {
              auto& value = app_->values_[u];
              auto last_value = value;
              // We don't cleanup delta with identity element, since we expect
              // the algorithm is monotonic
              auto& delta = app_->deltas_[u];
              if (app_->CombineValueDelta(value, delta)) {
                app_->Compute(u, last_value, delta, app_->next_modified_);
              }
            });
          // 测试单线程
          // for(auto i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue();
          //     i++) {
          //   vertex_t u(i);
          //   if (app_->curr_modified_.Exist(u)) {
          //     auto& value = app_->values_[u];
          //     auto last_value = value;
          //     auto& delta = app_->deltas_[u];
          //     if (app_->CombineValueDelta(value, delta)) {
          //       app_->Compute(u, last_value, delta, app_->next_modified_);
          //     }
          //   }
          // }
        }
        if (compr_stage) {
          ForEach(
              app_->curr_modified_, inner_vertices, [this, &compr_stage, &count, &step](int tid, vertex_t u) {
                char type = node_type[u.GetValue()];
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (type == 0) {
                  /* 0: out node */
                  // auto& value = app_->values_[u];
                  auto last_value = value;
                  // We don't cleanup delta with identity element, since we expect
                  // the algorithm is monotonic
                  // auto& delta = app_->deltas_[u];
                  if (app_->CombineValueDelta(value, delta)) {
                    app_->Compute(u, last_value, delta, app_->next_modified_);
                  }
                }
                else if (type == 1) {
                  /* 1: bound node */
                  // auto& value = app_->values_[u];
                  auto last_value = value;
                  // We don't cleanup delta with identity element, since we expect
                  // the algorithm is monotonic
                  // auto& delta = app_->deltas_[u];

                  if (app_->CombineValueDelta(value, delta)) {
                    adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                    app_->Compute(u, last_value, delta, oes, app_->next_modified_);
                  }
                }
                else if (type == 2) {
                  /* 2: source node */
                  // auto& value = app_->values_[u];
                  auto last_value = value;
                  // We don't cleanup delta with identity element, since we expect
                  // the algorithm is monotonic
                  // auto& delta = app_->deltas_[u];

                  if (app_->CombineValueDelta(value, delta)) {
                    adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                    app_->ComputeByIndexDelta(u, last_value, delta, oes, app_->next_modified_);
                    // app_->CombineValueDelta(spnode.data, delta);
                  }
                }
                else if (type == 3) {
                  // auto& value = app_->values_[u];
                  auto last_value = value;
                  // We don't cleanup delta with identity element, since we expect
                  // the algorithm is monotonic
                  // auto& delta = app_->deltas_[u];

                  if (app_->CombineValueDelta(value, delta)) {
                    /* 1: bound node */
                    adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                    app_->Compute(u, last_value, delta, oes_b, app_->next_modified_);
                    /* 2: source node */
                    adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                    app_->ComputeByIndexDelta(u, last_value, delta, oes_s, app_->next_modified_);
                  }
                }
              });
        }
      }
      // for_time += GetCurrentTime();

      auto& channels = messages_.Channels();
      // send local delta to remote
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

      // if(step % 100 == 0){
      //   LOG(INFO) << "[Worker " << comm_spec_.worker_id()
      //         << "]: Finished IterateKernel - " << step
      //         << " next_modified_.size=" << app_->next_modified_.Count();
      // }

      VLOG(1) << "[Worker " << comm_spec_.worker_id()
              << "]: Finished IterateKernel - " << step;
      messages_.FinishARound();

      // app_->next_modified_.Swap(app_->curr_modified_);

      exec_time += GetCurrentTime();

      bool terminate = messages_.ToTerminate();
      if ( (terminate && !FLAGS_gpu_start) || (!cur_modified_size_h[0] && FLAGS_gpu_start) ) {//if(app_->next_modified_.Count() == 0)
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
          
          // inner nodes receive
          if (FLAGS_cilk) {
            // 方案1.遍历全部的内部点和入口点
            // std::vector<vertex_t>& nodes_2 = all_nodes[2];
            // vid_t size = nodes_2.size();
            // parallel_for (vid_t i = 0; i < size; i++){
            //   vertex_t u(nodes_2[i]);
            //   auto& value = app_->values_[u];
            //   auto& delta = app_->deltas_[u];
            //   app_->CombineValueDelta(value, delta);
            // }
            // std::vector<vertex_t>& nodes_4 = all_nodes[4];
            // vid_t size_4 = nodes_4.size();
            // parallel_for (vid_t i = 0; i < size_4; i++){
            //   vertex_t u(nodes_4[i]);
            //   auto& value = app_->values_[u];
            //   auto& delta = app_->deltas_[u];
            //   app_->CombineValueDelta(value, delta);
            // }

            // 方案2.根据活跃队列进行更新
            // ForEachCilkOfBitset(
            //   app_->next_modified_, inner_vertices, 
            //   [this](int tid, vertex_t u) {
            //     auto& value = app_->values_[u];
            //     auto& delta = app_->deltas_[u];
            //     app_->CombineValueDelta(value, delta);
            //   }
            // );

            // 方案3.合并到入口点发送时一起聚合了
          }

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
    
    free(size_oe_h);
    free(size_ib_h);
    free(size_is_h);

    free(cur_oeoff_h);
    free(cur_iboff_h);
    free(cur_isoff_h);

    free(deltas_h);
    free(oeoffset_h);
    free(iboffset_h);
    free(isoffset_h);

    free(oe_edata_h);
    free(ib_edata_h);
    free(is_edata_h);

    cudaFree(deltas_d);
    cudaFree(values_d);

    cudaFree(oeoffset_d);
    cudaFree(iboffset_d);
    cudaFree(isoffset_d);

    cudaFree(oe_edata_d);
    cudaFree(ib_edata_d);
    cudaFree(is_edata_d);

    cudaFree(cur_oeoff_d);
    cudaFree(cur_iboff_d);
    cudaFree(cur_isoff_d);

    cudaFree(size_oe_d);
    cudaFree(size_ib_d);
    cudaFree(size_is_d);
    // Analysis result
    double d_sum = 0;
    vertex_t source;
    bool native_source =
        fragment_->GetInnerVertex(FLAGS_sssp_source, source);
    size_t visited_num = 0;
    vid_t max_id = native_source ? source.GetValue() : 0;
    for (auto v : fragment_->InnerVertices()) {
      if (app_->values_[v] != app_->GetIdentityElement()) {
        d_sum += app_->values_[v];
        visited_num += 1;
        if (app_->values_[v] > app_->values_[vertex_t(max_id)]) {
          max_id = v.GetValue();
        }
      }
    }
    LOG(INFO) << "max_d[" << fragment_->GetId(vertex_t(max_id)) << "]=" << app_->values_[vertex_t(max_id)];
    LOG(INFO) << "max id in suminc is"<<max_id;
    LOG(INFO) << "d_sum=" << d_sum;
    printf("#d_sum: %.10lf\n", d_sum);
    LOG(INFO) << "count=" << count;
    LOG(INFO) << "#visited_num: " << visited_num;
    LOG(INFO) << "cpr_->old_node_num: " << fragment_->GetVerticesNum() ;
    LOG(INFO) << "#visited_rate: " << (visited_num * 1.0 / fragment_->GetVerticesNum() );
    LOG(INFO) << "exec_time=" << exec_time;
    check_result("check finial realut");

    // debug
    if (FLAGS_compress && false) {
      for (vid_t i = 0; i < cpr_->old_node_num; i++) {
        vertex_t u(i);
        value_t u_v = app_->values_[u];
        value_t u_d = app_->deltas_[u].value;
        if (u_v != u_d) {
          LOG(INFO) << " u_v=" << u_v << " u_d=" << u_d 
                    << " u_oid=" << cpr_->v2Oid(u);
        }
      }
      vid_t oid = 245;
      if (oid < cpr_->old_node_num) {
        vertex_t u;
        bool native_source = fragment_->GetInnerVertex(oid, u);
        LOG(INFO) << " debug-- vid=" << u.GetValue() << " oid=" << oid;
        LOG(INFO) << " type=" << int(node_type[u.GetValue()]);
        LOG(INFO) << " value=" << app_->values_[u] 
                  << " delta=" << app_->deltas_[u].value;

        {
          vid_t spids_id = cpr_->id2spids[u];
          LOG(INFO) << " spids_id=" << spids_id;
          LOG(INFO) << " spid_id=" << cpr_->Fc_map[u];
          if (spids_id != cpr_->ID_default_value) {
            for (auto v : cpr_->supernode_ids[spids_id]) {
              LOG(INFO) << " include oid=" << cpr_->v2Oid(v)
                        << " vid=" << v.GetValue()
                        << " type=" << int(node_type[v.GetValue()]); 
            }
            for (auto v : cpr_->supernode_in_mirror[spids_id]) {
              LOG(INFO) << " include imid=" << cpr_->v2Oid(v); 
            }
            for (auto v : cpr_->supernode_out_mirror[spids_id]) {
              LOG(INFO) << " include omid=" << cpr_->v2Oid(v); 
            }
            for (auto v : cpr_->supernode_source[spids_id]) {
              LOG(INFO) << " include source_oid=" << cpr_->v2Oid(v) 
                        << " vid=" << v.GetValue();
              LOG(INFO) << "    v_value=" << app_->values_[v] 
                        << " delta=" << app_->deltas_[v];
              vid_t u = v.GetValue();
              // adj_list_t oes = adj_list_t(ib_e_offset_[u], ib_e_offset_[u+1]);
              adj_list_index_t oes = adj_list_index_t(is_e_offset_[u], is_e_offset_[u+1]);
              for (auto e : oes) {
                LOG(INFO) << "    " << cpr_->v2Oid(v) << " "
                          << cpr_->v2Oid(e.neighbor) 
                          << "    " << e.data << std::endl;
              }
            }
          }
        }
      }

    //   LOG(INFO) << "supernode_out_bound=" << cpr_->supernode_out_bound[u.GetValue()];
    }

    MPI_Barrier(comm_spec_.comm());
  }

  // 统计点集中点的分布情况：每个点的源顶点和出顶点中指向外部点的情况
  // 每一个点指向同一个cluster的数量
  void print_indegree_outdegree(std::vector<vertex_t> &node_set){
    LOG(INFO) << "--------------------------------------------------";
    LOG(INFO) << " node_set.size=" << node_set.size();
    {
      std::ofstream fout("./out_edge");
      for (auto u : node_set) {
        vid_t i = u.GetValue();
        std::unordered_map<vid_t, size_t> edge_count;
        vid_t spids = cpr_->id2spids[u];
        for(auto e : fragment_->GetOutgoingAdjList(u)) {
          vid_t to_ids = cpr_->id2spids[e.neighbor];
          if(to_ids != spids){
            edge_count[to_ids] += 1;
          }
        }
        fout << i << ":";
        for(const auto& pair : edge_count) {
          fout << " " << pair.second;
        }
        fout << "\n";
      }
      fout.close();
    }
    {
      std::ofstream fout("./in_edge");
      for (auto u : node_set) {
        vid_t i = u.GetValue();
        std::unordered_map<vid_t, size_t> edge_count;
        vid_t spids = cpr_->id2spids[u];
        for(auto e : fragment_->GetIncomingAdjList(u)) {
          vid_t to_ids = cpr_->id2spids[e.neighbor];
          if(to_ids != spids){
            edge_count[to_ids] += 1;
          }
        }
        fout << i << ":";
        for(const auto& pair : edge_count) {
          fout << " " << pair.second;
        }
        fout << "\n";
      }
      fout.close();
    }
    LOG(INFO) << "finish write.2.. ";
  }

  void print_result(std::string position = ""){
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    LOG(INFO) << "-----------result---s------------in " << position;
    for (auto v : inner_vertices) {
      vertex_t p;
      // LOG(INFO) << "oid=" << fragment_->GetId(v) << " id=" << v.GetValue()
      //           << " deltas[v].parent_gid=" << deltas[v].parent_gid;
      // fragment_->Gid2Vertex(deltas[v].parent_gid, p);
      LOG(INFO) << "oid=" << fragment_->GetId(v) << " id=" << v.GetValue() 
                << ": value=" << values[v] << " delta=" << deltas[v].value 
                << " oid=" << fragment_->GetId(p) << std::endl;
    }
    LOG(INFO) << "-----------result---e------------";
  }
  
  void check_result(std::string position = ""){
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    double value_sum = 0;
    double delta_sum = 0;
    LOG(INFO) << "----------check_result in " << position;
    for (auto v : inner_vertices) {
      // LOG(INFO) << " v.oid=" << cpr_->v2Oid(v); 
      if(values[v] != app_->GetIdentityElement())
      value_sum += values[v];
      if(deltas[v].value != app_->GetIdentityElement())
      delta_sum += deltas[v].value;
    }
    printf("---value_sum=%.10lf\n", value_sum);
    printf("---delta_sum=%.10lf\n", delta_sum);
  }

  void Output(std::ostream& os) {
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    // for (auto v : inner_vertices) {
    //   os << fragment_->GetId(v) << " " << values[v] << " " << deltas[v].parent_gid << std::endl;
    //   // os << fragment_->GetId(v) << " " << deltas[v].parent_gid << std::endl;
    // }
    // return ;
    // Write hypergraph to file
    {
      if (FLAGS_compress == false) {
        return ;
      }
      LOG(INFO) << "write supergraph...";
      long long edge_num = 0;
      for (auto u : inner_vertices) {
        char type = node_type[u.GetValue()];
        if (type == 0 || type == 1) {
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          edge_num += oes.Size();
          for (auto e : oes) {
            os << fragment_->GetId(u) << " "
               << fragment_->GetId(e.neighbor) 
               << " " << e.data << std::endl;
          }
        } else if (type == 2) {
          adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          edge_num += oes.Size();
          for (auto e : oes) {
            os << fragment_->GetId(u) << " " 
               << fragment_->GetId(e.neighbor) 
               << " " << e.data.value << std::endl;
          }
        } else if (type == 3) {
          adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          edge_num += oes_b.Size();
          for (auto e : oes_b) {
            os << fragment_->GetId(u) << " " 
               << fragment_->GetId(e.neighbor)
               << " " << e.data << std::endl;
          }
          // os << "----------\n";
          adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          edge_num += oes_s.Size();
          for (auto e : oes_s) {
            os << fragment_->GetId(u) << " " 
               << fragment_->GetId(e.neighbor)
               << " " << e.data.value << std::endl;
          }
        }
        // os << "--------test------\n";
      }
      LOG(INFO) << "edge_num=" << edge_num;
    }
  }

  void Finalize() { messages_.Finalize(); }

  void print_active_edge(std::string position = "") {
    LOG(INFO) << position << "_f_index_count_num: " << app_->f_index_count_num;
    LOG(INFO) << position << "_f_send_delta_num: " << app_->f_send_delta_num;
    app_->f_index_count_num = 0;
    app_->f_send_delta_num = 0;
  }


 private:
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t> fragment_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
  TravCompressor<APP_T, supernode_t>* cpr_;
  /* source to inner_node: index */
  std::vector<vertex_t> source_nodes; // source: type2 + type3
  Array<nbr_index_t, Allocator<nbr_index_t>> is_iindex_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_iindex_offset_;
  /* source to in_bound_node: index */
  Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
  /* in_bound_node to out_bound_node: original edge */
  Array<nbr_t, Allocator<nbr_t>> ib_e_;
  Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
  /* each type of vertices */
  std::vector<std::vector<vertex_t>> all_nodes;
  std::vector<char> node_type; // all node's types, 0:out node, 1:bound node, 2:source node, 3:belong 1 and 2 at the same time, 4:inner node that needn't send message.
  Array<vid_t, Allocator<vid_t>> oldId2newId; // renumber all internal vertices
  Array<vid_t, Allocator<vid_t>> newId2oldId; // renumber all internal vertices
  Array<vid_t, Allocator<vid_t>> oldGid2newGid; // renumber all internal vertices
  Array<vid_t, Allocator<vid_t>> newGid2oldGid; // renumber all internal vertices
  std::vector<vid_t> node_range; // 0-1-2-3-4
  // DenseVertexSet<vid_t> curr_modified_new, next_modified_new;
};

}  // namespace grape

#endif