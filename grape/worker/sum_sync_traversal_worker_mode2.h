
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
    // init compressor
    if(FLAGS_compress){
      cpr_ = new TravCompressor<APP_T, supernode_t>(app_, fragment_);
      cpr_->init(comm_spec_, communicator_, pe_spec);
      cpr_->run();
      cpr_->statistic();
      // // debug
      // {
      //   LOG(INFO) << "测试-----";
      //   exit(1) ;
      // }
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
    std::unordered_set<vid_t> local_gid_set;

    for (auto v : fragment_->Vertices()) {
      local_gid_set.insert(fragment_->Vertex2Gid(v));
    }

    auto vertices = fragment_->Vertices();
    DenseVertexSet<vid_t> curr_modified, next_modified, reset_vertices;

    curr_modified.Init(vertices);
    next_modified.Init(vertices);
    reset_vertices.Init(inner_vertices);  // Only used for counting purpose

    for (auto& pair : deleted_edges) {
      vid_t u_gid = pair.first, v_gid = pair.second;

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
    }

    auto& channels = messages_.Channels();

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Resetting";
    }

    do {
      messages_.StartARound();
      messages_.ParallelProcess<fragment_t, grape::EmptyType>(
          thread_num(), *fragment_,
          [&curr_modified](int tid, vertex_t v, const grape::EmptyType& msg) {
            curr_modified.Insert(v);
          });

      // 可以尝试换成cilk
      ForEachSimple(curr_modified, inner_vertices,
                    [this, &next_modified](int tid, vertex_t u) {
                      auto u_gid = fragment_->Vertex2Gid(u);
                      auto oes = fragment_->GetOutgoingAdjList(u);

                      for (auto e : oes) {
                        auto v = e.neighbor;

                        if (app_->DeltaParentGid(v) == u_gid) {
                          next_modified.Insert(v);
                        }
                      }
                    });

      ForEachSimple(curr_modified, inner_vertices,
                    [this, &reset_vertices](int tid, vertex_t u) {
                      // app_->values_[u] = app_->GetIdentityElement();
                      // app_->deltas_[u].Reset(app_->GetIdentityElement()); 

                      app_->values_[u] = app_->GetInitValue(u);
                      app_->deltas_[u] = app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
                      app_->CombineValueDelta(app_->values_[u], app_->deltas_[u]);

                      reset_vertices.Insert(u);
                    });

      ForEach(next_modified, outer_vertices,
              [&channels, this](int tid, vertex_t v) {
                grape::EmptyType dummy;
                channels[tid].SyncStateOnOuterVertex(*fragment_, v, dummy);
                // app_->deltas_[v].Reset(app_->GetIdentityElement());
                app_->deltas_[v] = app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
              });
      messages_.FinishARound();

      if (next_modified.Count() > 0) {
        messages_.ForceContinue();
      }

      curr_modified.Clear();
      curr_modified.Swap(next_modified);
    } while (!messages_.ToTerminate());

    size_t n_reset = 0, local_n_reset = reset_vertices.Count();

    Communicator communicator;

    communicator.InitCommunicator(comm_spec_.comm());
    communicator.template Sum(local_n_reset, n_reset);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "# of reset vertices: " << n_reset << " reset percent: "
                << (float) n_reset / fragment_->GetTotalVerticesNum();
      LOG(INFO) << "Start a round from all vertices";
    }

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
    {
      LOG(INFO) << "---------------------------reset-------------------------";
      LOG(INFO) << " f_send_delta_num=" << app_->f_send_delta_num;
      LOG(INFO) << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
      app_->f_send_delta_num = 0;
      app_->node_update_num = 0;
      app_->touch_nodes.ParallelClear(8);
      LOG(INFO) << "---------------------------end-------------------------";
    }

    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    if(FLAGS_compress){
      auto added_edges = inc_fragment_builder.GetAddedEdgesGid();
      cpr_->inc_run(deleted_edges, added_edges, new_graph);
      {
        LOG(INFO) << "---------------------------inc_run-------------------------";
        LOG(INFO) << " f_send_delta_num=" << app_->f_send_delta_num;
        LOG(INFO) << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
        app_->f_send_delta_num = 0;
        app_->node_update_num = 0;
        app_->touch_nodes.ParallelClear(8);
        LOG(INFO) << "---------------------------end-------------------------";
      }
    }
    fragment_ = new_graph;

    // Important!!! outer vertices may change, we should acquire it after new
    // graph is loaded
    outer_vertices = fragment_->OuterVertices();
    // Reset all states, active vertices will be marked in curr_modified_
    app_->Init(comm_spec_, fragment_);

    // copy to new graph
    for (auto v : inner_vertices) {
      app_->values_[v] = values[v];
      app_->deltas_[v] = deltas[v];
    }

    // Start a round without any condition
    messages_.StartARound();
    for (auto u : inner_vertices) {
      auto& value = app_->values_[u];
      auto& delta = app_->deltas_[u];

      if (delta.value != app_->GetIdentityElement()) {
        app_->Compute(u, value, delta, next_modified);
      }
    }

    ForEach(
        next_modified, outer_vertices, [&channels, this](int tid, vertex_t v) {
          auto& delta_to_send = app_->deltas_[v];
          if (delta_to_send.value != app_->GetIdentityElement()) {
            channels[tid].SyncStateOnOuterVertex(*fragment_, v, delta_to_send);
          }
        });
    messages_.FinishARound();
    app_->next_modified_.Swap(next_modified);

    {
      LOG(INFO) << "---------------------------reissue------------------------";
      LOG(INFO) << " f_send_delta_num=" << app_->f_send_delta_num;
      LOG(INFO) << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
      app_->f_send_delta_num = 0;
      app_->node_update_num = 0;
      app_->touch_nodes.ParallelClear(8);
      LOG(INFO) << "---------------------------end------------------------";
    }

    if(FLAGS_compress){
      timer_next("inc pre compute");
      double inc_pre_compute = GetCurrentTime();
      cpr_->precompute_spnode(fragment_);
      cpr_->precompute_spnode_all(fragment_); // 应该是被touch到的超点需要
      inc_pre_compute = GetCurrentTime() - inc_pre_compute;
      LOG(INFO) << "#inc_pre_compute: " << inc_pre_compute;
    }
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

  void Query() {
    MPI_Barrier(comm_spec_.comm());

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
    if(compr_stage){
      double extra_all_time = GetCurrentTime();
      auto inner_vertices = fragment_->InnerVertices();
      double node_type_time = GetCurrentTime();
      vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
      node_type.clear();
      node_type.resize(inner_node_num, std::numeric_limits<char>::max());
      // bound_node_values.clear();
      // bound_node_values.Init(inner_vertices, app_->default_v());
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
      LOG(INFO) << " node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

      /* renumber internal vertices */
      oldId2newId.clear();
      oldId2newId.resize(inner_node_num);
      newId2oldId.clear();
      newId2oldId.resize(inner_node_num);
      oldGid2newGid.clear();
      oldGid2newGid.resize(inner_node_num); // 分布式情况下，可能会越界，换成unordered_map？？？
      newGid2oldGid.clear();
      newGid2oldGid.resize(inner_node_num);
      node_range.clear();
      node_range.resize(6);
      vid_t index_id = 0;
      // for (int i = 0; i < 5; i++) {
      //   index_id[i+1] = index_id[i] + all_nodes[i].size();
      // }
      for (vid_t i = 0; i < 5; i++) {
        const std::vector<vertex_t>& nodes = all_nodes[i];
        size_t size = nodes.size();
        node_range[i] = index_id;
        parallel_for (vid_t j = 0; j < size; j++) {
          oldId2newId[nodes[j].GetValue()] = index_id + j;
          newId2oldId[index_id + j] = nodes[j].GetValue();
          // 需要加个判断，只转化本地gid
          vid_t old_gid = fragment_->Vertex2Gid(nodes[j]);
          vid_t new_gid = fragment_->Vertex2Gid(vertex_t(index_id + j));
          oldGid2newGid[old_gid] = new_gid;
          newGid2oldGid[new_gid] = old_gid;
        }
        index_id += size;
      }
      node_range[5] = index_id;

      // debug
      { 
#ifdef DEBUG
        for (int i = 0; i < 5; i++) {
          LOG(INFO) << "node_" << i << "=" << all_nodes[i].size();
          LOG(INFO) << "node_range[" << i << "]=" << node_range[i];
        }

        LOG(INFO) << "node_4(inner)/all_node=" << (all_nodes[4].size() * 1.0 / inner_node_num);

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
            auto oes = fragment_->GetOutgoingAdjList(u);
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
            auto oes = fragment_->GetOutgoingAdjList(u);
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
#endif
      }

      double  transfer_csr_time = GetCurrentTime();

      /* source to in_bound_node */

      double  init_time_1 = GetCurrentTime();
      // spnode_datas.Init(inner_vertices, 0);
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << " init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == 2 || type == 3){ // index
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          // is_e_degree[i+1] = spnode.bound_delta.size();
          is_e_degree[oldId2newId[i]+1] = spnode.bound_delta.size();
        }
        if(type == 1 || type == 3){ // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              // bound_e_num += 1;
              temp_cnt += 1;
            }
          }
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] = temp_cnt;
        } else if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] = temp_cnt;
        }
      }
      LOG(INFO) << " csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      LOG(INFO) << " index_time=" << (GetCurrentTime()-index_time); //0.226317

#ifdef DEBUG
      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << fragment_->GetVerticesNum();
#endif

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << " init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      /* build index/edge */
      double csr_time_2 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        // vid_t index_s = is_e_degree[i];
        // is_e_offset_[u.GetValue()] = &is_e_[index_s];
        vid_t new_id = oldId2newId[i];
        vid_t index_s = is_e_degree[new_id];
        is_e_offset_[new_id] = &is_e_[index_s];
        char type = node_type[i];
        if(type == 2 || type == 3){ // index
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          for(auto& oe : spnode.bound_delta){
            // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
            // is_e_[index_s].neighbor = oe.first;
            if (oe.first.GetValue() < inner_node_num) {
              is_e_[index_s].neighbor = oldId2newId[oe.first.GetValue()];
            } else {
              is_e_[index_s].neighbor = oe.first;
            }
            // The dependent parent id also donot needs to be updated, 
            // because it is gid.
            is_e_[index_s].data = oe.second;
            if (oe.second.parent_gid < inner_node_num) {
              is_e_[index_s].data.parent_gid = oldId2newId[oe.second.parent_gid];
            }
            index_s++;
          }
        }
        /* inner_bound node */
        // vid_t index_b = ib_e_degree[i];
        vid_t index_b = ib_e_degree[new_id];
        ib_e_offset_[new_id] = &ib_e_[index_b];
        if(type == 1 || type == 3){ // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              ib_e_[index_b] = e;
              auto nbr = ib_e_[index_b].neighbor;
              if (nbr.GetValue() < inner_node_num) {
                ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
              }
              index_b++;
            }
          }
        }
        if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            auto nbr = ib_e_[index_b].neighbor;
            if (nbr.GetValue() < inner_node_num) {
              ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
            }
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << " csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      // copy from graph SumInc/grape/fragment/immutable_edgecut_fragment.h
      // test cache
      {
        // ib_e_offset_ should have been sorted.
        // for (vid_t i = 0; i < inner_node_num; ++i) {
        //   std::sort(ib_e_offset_[i], ib_e_offset_[i + 1],
        //           [](const nbr_t& lhs, const nbr_t& rhs) {
        //             return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
        //           });
        // }
        parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }

      // GetInnerIndex(); // build inner index's csr

      /* exchage value and delta of old_id and new_id */
      double exchage_time = GetCurrentTime();
      parallel_for (vid_t i = 0; i < inner_node_num; i++) {
        vertex_t v(i);
        values_temp[v] = app_->values_[v];
        deltas_temp[v] = app_->deltas_[v];
        // vertex_t p;                            
        // fragment_->Gid2Vertex(deltas_temp[v].parent_gid, p);             
        // if (p.GetValue() < inner_node_num) {               
        //   deltas_temp[v].parent_gid = fragment_->Vertex2Gid(vertex_t(newId2oldId[p.GetValue()]));             
        // }
        deltas_temp[v].parent_gid = oldGid2newGid[deltas_temp[v].parent_gid];
      }
      parallel_for (vid_t i = 0; i < inner_node_num; i++) {
        vertex_t v(i);
        app_->values_[vertex_t(oldId2newId[i])] = values_temp[v];
        app_->deltas_[vertex_t(oldId2newId[i])] = deltas_temp[v];
      }
      LOG(INFO) << "#exchage_time: " << (GetCurrentTime()- exchage_time); //3.88149

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149

      {
        // check_result("init before");
        exec_time -= GetCurrentTime();
        // Update the source id to the new id
        vertex_t source;
        bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
        if (native_source) {
          vid_t new_source_id = oldId2newId[source.GetValue()];
          app_->curr_modified_.Insert(vertex_t(new_source_id));
          LOG(INFO) << "supernode... newid=" << new_source_id << " type4=" << node_range[4];
          LOG(INFO) << "this->Fc[source]=" << cpr_->Fc[source];
          // if (new_source_id >= node_range[4] || (new_source_id >= node_range[1] && new_source_id < node_range[2])) {
          if (new_source_id > node_range[0]) {
            LOG(INFO) << "Send one round for supernode...";
            /* send one round */
            if (FLAGS_cilk) {
              /* type=1: bound node */
              parallel_for(vid_t i = node_range[1]; i < node_range[2]; i++){
                vertex_t u(i);
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, app_->curr_modified_);
                }
              }
              /* type=3: bound + soure node */
              parallel_for(vid_t i = node_range[3]; i < node_range[4]; i++){
                vertex_t u(i);
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, app_->curr_modified_);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->curr_modified_);
                }
              }
            } else {
              ForEach(VertexRange<vid_t>(node_range[1], node_range[2]), [this](int tid, vertex_t u) {
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, app_->curr_modified_);
                }
              });
              ForEach(VertexRange<vid_t>(node_range[3], node_range[4]), [this](int tid, vertex_t u) {
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, app_->curr_modified_);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->curr_modified_);
                }
              });
            }
          }
        }
        exec_time += GetCurrentTime();
        LOG(INFO) << " pre_exec_time=" << exec_time;
        // check_result("init after");
        LOG(INFO) << "init after bitset.size=" << app_->curr_modified_.ParallelCount(thread_num());
      }
      
      LOG(INFO) << "extra_all_time=" << (GetCurrentTime()- extra_all_time);

      //debug
      {
#ifdef DEBUG
        const std::vector<vertex_t>& nodes_0 = all_nodes[0];
        vid_t node_0_size = nodes_0.size();
        size_t max_edge_0_num = 0;
        size_t edge_0_num = 0;
        for(vid_t i = 0; i < node_0_size; i++){
          vertex_t u(nodes_0[i]);
          auto oes = fragment_->GetOutgoingAdjList(u);
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
#endif
      }

      // debug: 统计每个类型点的最大度数
      if (false) {
        std::vector<size_t> max_indegree;
        max_indegree.clear();
        max_indegree.resize(5);
        std::vector<size_t> max_outdegree;
        max_outdegree.clear();
        max_outdegree.resize(5);
        for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
          vertex_t u(i);
          max_outdegree[node_type[i]] = std::max(max_outdegree[node_type[i]], fragment_->GetOutgoingAdjList(u).Size());
          max_indegree[node_type[i]] = std::max(max_indegree[node_type[i]], fragment_->GetIncomingAdjList(u).Size());
        }
        for (int i = 0; i < 5; i++) {
          LOG(INFO) << "max_indegree typeid=" << i << " : " << max_indegree[i];
        }
        for (int i = 0; i < 5; i++) {
          LOG(INFO) << "max_outdegree typeid=" << i << " : " << max_outdegree[i];
        }

        // std::vector<size_t> indegree;
        // indegree.clear();
        // indegree.resize(5);

        for (int t = 0; t < 5; t++) {
          std::ofstream fout("./indegree_" + std::to_string(t));
          for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            if (node_type[i] == t) {
              fout << fragment_->GetIncomingAdjList(u).Size() << " ";
            }
          }
          fout.close();
        }

        for (int t = 0; t < 5; t++) {
          std::ofstream fout("./outdegree_" + std::to_string(t));
          for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            if (node_type[i] == t) {
              fout << fragment_->GetOutgoingAdjList(u).Size() << " ";
            }
          }
          fout.close();
        }
        LOG(INFO) << "finish write.1.. ";
      }
      // debug: 统计cluster内部每个边界点(入口点、出口点、入口+出口点)在其指向/被指向其它同一个clust的边数
      if (false) {
        for (int t = 1; t < 4; t++) {
          std::ofstream fout("./out_edge_" + std::to_string(t));
          for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            if (node_type[i] == t) {
              // fout << fragment_->GetOutgoingAdjList(u).Size() << " ";
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
          }
          fout.close();
        }
        for (int t = 1; t < 4; t++) {
          std::ofstream fout("./in_edge_" + std::to_string(t));
          for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            if (node_type[i] == t) {
              // fout << fragment_->GetOutgoingAdjList(u).Size() << " ";
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
          }
          fout.close();
        }
        LOG(INFO) << "finish write.2.. ";
      }
      // debug: (入口点)统计如果采用Mirror-Master能对边的减少率能提高多少
      if (true) {
        LOG(INFO) << "-------------Mirror-Master---------------";
        size_t k = FLAGS_mirror_k; // 阈值
        size_t mirror_num = 0;
        size_t reduce_edge_num = 0;
        size_t new_index_num = 0;
        size_t old_index_num = 0;
        const vid_t spn_ids_num = cpr_->supernode_ids.size(); 
        for (vid_t j = 0; j < spn_ids_num; j++){
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[j];
          // 统计所有入口点的源顶点
          std::unordered_map<vid_t, size_t> frequent;
          size_t exit_node_num = 0;
          for (auto u : node_set) {
            for (auto e : fragment_->GetIncomingAdjList(u)) {
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if (to_ids != j) { // 外部点
                frequent[e.neighbor.GetValue()] += 1;
              }
            }
            if (cpr_->supernode_out_bound[u.GetValue()]) {
              exit_node_num++;
            }
          }
          // 分析出现频率
          size_t max_fre = 0;
          size_t out_edge_num = 0;
          size_t source_node_num = 0;
          size_t entry_node_num = cpr_->supernode_source[j].size();
          for (const auto& fre : frequent) {
            if (fre.second > max_fre) {
              max_fre = fre.second;
            }
            out_edge_num += fre.second;
            source_node_num += 1;
            // LOG(INFO) << fre.first << ": " << fre.second << std::endl;
          }
          bool f2 = (out_edge_num + entry_node_num * exit_node_num > 
            source_node_num + source_node_num * exit_node_num); // 考虑了用mirror减少的边
          // if (max_fre > k && source_node_num < entry_node_num) { // f1 + f2
          // if (max_fre > k) { // f1
          if (max_fre > k && f2) { // f1 + f2: 考虑了mirror减少的边
            mirror_num += source_node_num;
            reduce_edge_num += out_edge_num;
            new_index_num += source_node_num * exit_node_num;
          } else {
            new_index_num += entry_node_num * exit_node_num;
          }
          old_index_num += entry_node_num * exit_node_num;
          // LOG(INFO) << "max_fre=" << max_fre;
          // LOG(INFO) << "------- entry_node_num=" << entry_node_num 
          //   << " exit_node_num=" << exit_node_num 
          //   << " out_edge_num=" << out_edge_num;
        }
        LOG(INFO) << "k=" << k << " mirror_num= " << mirror_num 
          << " reduce_edge_num=" << reduce_edge_num 
          << " new_index_num=" << new_index_num
          << " old_index_num=" << old_index_num;
      }
      // debug  (入口点)统计如果采用Mirror-Master能对边的减少率能提高多少, 前期没有入口*出口过滤，在此处过滤
      if (true) {
        LOG(INFO) << "------Mirror-Master---entry_node: no-filter on early stage------";
        typedef long long count_t;
        count_t k = FLAGS_mirror_k; // 阈值
        count_t mirror_num = 0;
        count_t reduce_edge_num = 0;
        count_t new_index_num = 0;
        count_t old_index_num = 0;
        count_t old_inner_edge = 0;
        count_t new_inner_edge = 0;
        count_t spnids_num = 0;
        count_t add_spnids_num = 0;
        count_t all_old_exit_node_num = 0;
        count_t all_new_exit_node_num = 0;
        count_t all_old_entry_node_num = 0;
        count_t all_new_entry_node_num = 0;
        count_t abandon_node_num = 0; // 不满足入口*出口舍弃的点
        count_t abandon_edge_num = 0;
        const vid_t spn_ids_num = cpr_->supernode_ids.size(); 
        for (vid_t j = 0; j < spn_ids_num; j++){
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[j];
          // 统计所有入口点的源顶点
          std::unordered_map<vid_t, size_t> frequent;
          size_t old_exit_node_num = 0;
          size_t temp_old_inner_edge = 0;
          for (auto u : node_set) {
            for (auto e : fragment_->GetIncomingAdjList(u)) {
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if (to_ids != j) { // 外部点
                frequent[e.neighbor.GetValue()] += 1;
              } else {
                temp_old_inner_edge++;
              }
            }
            if (cpr_->supernode_out_bound[u.GetValue()]) {
              old_exit_node_num++;
            }
          }
          // 分析出现频率
          size_t out_edge_num = 0;
          size_t mirror_node_num = 0;
          size_t old_entry_node_num = cpr_->supernode_source[j].size();
          std::set<vertex_t> old_P;
          old_P.insert(node_set.begin(), node_set.end());
          std::set<vertex_t> P;
          P.insert(node_set.begin(), node_set.end());
          
          for (const auto& fre : frequent) {
            if (fre.second > k) {
              out_edge_num += fre.second;
              mirror_node_num += 1;
              P.insert(vertex_t(fre.first));  // 高频入口点的源顶点作为Mirror点
            }
            // LOG(INFO) << fre.first << ": " << fre.second << std::endl;
          }
          // 统计老的出口点
          // std::set<vertex_t> old_B; // belong to P, bound vertices
          // for(auto v : old_P){ // 遍历源来的点集
          //   const auto& oes = fragment_->GetOutgoingAdjList(v);
          //   for(auto& e : oes){
          //       if(old_P.find(e.neighbor) == old_P.end()){ // 包含Mirror
          //           old_B.insert(v);
          //           break;
          //       }
          //   }
          // }

          // 统计新的出口点
          std::set<vertex_t> B; // belong to P, bound vertices
          for(auto v : node_set){ // 遍历源来的点集
            const auto& oes = fragment_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(old_P.find(e.neighbor) == old_P.end()){ // 不包含Mirror
                    B.insert(v);
                    break;
                }
            }
          }
          // 统计新的入口点
          std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
          for(auto v : node_set){ // 遍历源来的点集
              const auto& oes = fragment_->GetIncomingAdjList(v);
              for(auto& e : oes){
                  if(P.find(e.neighbor) == P.end()){ // 包含Mirror
                      S.insert(v);
                      break;
                  }
              }
          }
          // 统计最终结果
          size_t new_exit_node_num = B.size();
          size_t new_entry_node_num = S.size() + mirror_node_num;
          size_t temp_old_index_num = old_exit_node_num * old_entry_node_num;
          size_t temp_new_index_num = new_exit_node_num * new_entry_node_num;

          const bool original_compress_condition = 
            (temp_old_index_num < temp_old_inner_edge);
          const bool mirror_compress_condition = 
            (temp_new_index_num + mirror_node_num 
              < temp_old_inner_edge + out_edge_num); // 加Mirror后是否应该压缩

          // 统计未能压缩的点和边，即放弃的cluster
          if ( original_compress_condition == false 
            && mirror_compress_condition == false) {
              abandon_edge_num += temp_old_inner_edge;
              abandon_node_num += old_P.size();
          }

          // 不加mirror点的情况
          if (original_compress_condition == true) {
            spnids_num++;
            old_inner_edge += temp_old_inner_edge;
            old_index_num += temp_old_index_num;
            all_old_entry_node_num += old_entry_node_num;
            all_old_exit_node_num += old_exit_node_num;
            new_inner_edge += temp_old_inner_edge;
            new_index_num += temp_old_index_num;
            all_new_entry_node_num += new_entry_node_num;
            all_new_exit_node_num += new_exit_node_num;
          }
          // 加mirror点的情况
          if (mirror_compress_condition == true) { // 加Mirror后是否应该压缩
            if (original_compress_condition == false) { // 未加Mirror时是否压缩了
              add_spnids_num++; // 仅仅因为加mirror才成为cluster
              new_inner_edge += temp_old_inner_edge;
              all_new_entry_node_num += new_entry_node_num;
              all_new_exit_node_num += new_exit_node_num;
            } else {
              new_index_num -= temp_old_index_num;
            }
            const bool f2 = (out_edge_num + temp_old_index_num 
              > mirror_node_num + temp_new_index_num); // 相对于普通压缩而言，加了Mirror后是否有收益
            if (f2) {
              mirror_num += mirror_node_num;
              reduce_edge_num += out_edge_num;
              new_index_num += temp_new_index_num; // 使用新方案
            } else {
              new_index_num += temp_old_index_num; // 依然老方案
            }
          }

          // LOG(INFO) << "test---" << "temp_old_inner_edge=" << temp_old_inner_edge
            // << " new_inner_edge=" << new_inner_edge;
          // LOG(INFO) << "mirror_node_num=" << mirror_node_num;
          // LOG(INFO) << "------- out_edge_num=" << out_edge_num 
          //   << " new_exit_node_num=" << new_exit_node_num 
          //   << " new_entry_node_num=" << new_entry_node_num 
          //   << " old_exit_node_num=" << old_exit_node_num 
          //   << " old_entry_node_num=" << old_entry_node_num;
        }
        LOG(INFO) << "k=" << k << " mirror_num= " << mirror_num 
          << " reduce_edge_num=" << reduce_edge_num 
          << " new_index_num=" << new_index_num
          << " old_index_num=" << old_index_num;
        LOG(INFO) << "spnids_num=" << spnids_num 
          << " add_spnids_num=" << add_spnids_num
          << " new_inner_edge=" << new_inner_edge
          << " old_inner_edge=" << old_inner_edge;
        LOG(INFO) << "all_old_entry_node_num=" << all_old_entry_node_num
          << " all_old_exit_node_num=" << all_old_exit_node_num
          << " all_new_entry_node_num=" << all_new_entry_node_num
          << " all_new_exit_node_num=" << all_new_exit_node_num;
        LOG(INFO) << "abandon_edge_num=" << abandon_edge_num 
          << " abandon_node_num=" << abandon_node_num;
      }
      // debug: (出口点)统计如果采用Mirror-Master能对边的减少率能提高多少
      if (true) {
        LOG(INFO) << "-------------Mirror-Master---exit_node------------";
        size_t k = FLAGS_mirror_k; // 阈值
        size_t mirror_num = 0;
        size_t reduce_edge_num = 0;
        size_t new_index_num = 0;
        size_t old_index_num = 0;
        const vid_t spn_ids_num = cpr_->supernode_ids.size(); 
        for (vid_t j = 0; j < spn_ids_num; j++){
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[j];
          // 统计所有入口点的源顶点
          std::unordered_map<vid_t, size_t> frequent;
          size_t old_exit_node_num = 0;
          for (auto u : node_set) {
            for (auto e : fragment_->GetOutgoingAdjList(u)) {
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if (to_ids != j) { // 外部点
                frequent[e.neighbor.GetValue()] += 1;
              }
            }
            if (cpr_->supernode_out_bound[u.GetValue()]) {
              old_exit_node_num++;
            }
          }
          // 分析出现频率
          size_t out_edge_num = 0;
          size_t mirror_node_num = 0;
          size_t old_entry_node_num = cpr_->supernode_source[j].size();
          std::set<vertex_t> old_P;
          old_P.insert(node_set.begin(), node_set.end());
          std::set<vertex_t> P;
          P.insert(node_set.begin(), node_set.end());
          
          for (const auto& fre : frequent) {
            if (fre.second > k) {
              out_edge_num += fre.second;
              mirror_node_num += 1;
              P.insert(vertex_t(fre.first));  // 高频出口点的目的点作为Mirror点
            }
            // LOG(INFO) << fre.first << ": " << fre.second << std::endl;
          }
          // 统计新的出口点
          std::set<vertex_t> B; // belong to P, bound vertices
          for(auto v : node_set){ // 遍历源来的点集
            const auto& oes = fragment_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) == P.end()){ // 包含Mirror
                    B.insert(v);
                    break;
                }
            }
          }
          std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
          for(auto v : node_set){ // 遍历源来的点集
              const auto& oes = fragment_->GetIncomingAdjList(v);
              for(auto& e : oes){
                  if(old_P.find(e.neighbor) == old_P.end()){ // 不包含Mirror
                      S.insert(v);
                      break;
                  }
              }
          }
          // 统计最终结果
          size_t new_exit_node_num = B.size() + mirror_node_num;
          size_t new_entry_node_num = S.size();

          bool f2 = (out_edge_num + old_exit_node_num * old_entry_node_num 
            > mirror_node_num + new_exit_node_num * new_entry_node_num); // 加了Mirror后是否有收益
          if (f2) {
            mirror_num += mirror_node_num;
            reduce_edge_num += out_edge_num;
            new_index_num += new_exit_node_num * new_entry_node_num; // 使用新方案
          } else {
            new_index_num += old_exit_node_num * old_entry_node_num; // 依然老方案
          }
          old_index_num += old_exit_node_num * old_entry_node_num;

          // LOG(INFO) << "mirror_node_num=" << mirror_node_num;
          // LOG(INFO) << "------- out_edge_num=" << out_edge_num 
          //   << " new_exit_node_num=" << new_exit_node_num 
          //   << " new_entry_node_num=" << new_entry_node_num 
          //   << " old_exit_node_num=" << old_exit_node_num 
          //   << " old_entry_node_num=" << old_entry_node_num;
        }
        LOG(INFO) << "k=" << k << " mirror_num= " << mirror_num 
          << " reduce_edge_num=" << reduce_edge_num 
          << " new_index_num=" << new_index_num
          << " old_index_num=" << old_index_num;
      }
      // debug: (出口点)统计如果采用Mirror-Master能对边的减少率能提高多少, 前期没有入口*出口过滤，在此处过滤
      if (true) {
        LOG(INFO) << "------Mirror-Master---exit_node: no-filter on early stage------";
        typedef long long count_t;
        size_t k = FLAGS_mirror_k; // 阈值
        count_t mirror_num = 0;
        count_t reduce_edge_num = 0;
        count_t new_index_num = 0;
        count_t old_index_num = 0;
        count_t old_inner_edge = 0;
        count_t new_inner_edge = 0;
        count_t spnids_num = 0;
        count_t add_spnids_num = 0;
        count_t all_old_exit_node_num = 0;
        count_t all_new_exit_node_num = 0;
        count_t all_old_entry_node_num = 0;
        count_t all_new_entry_node_num = 0;
        count_t abandon_node_num = 0; // 不满足入口*出口舍弃的点
        count_t abandon_edge_num = 0;
        const vid_t spn_ids_num = cpr_->supernode_ids.size(); 
        for (vid_t j = 0; j < spn_ids_num; j++){
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[j];
          // 统计所有入口点的源顶点
          std::unordered_map<vid_t, count_t> frequent;
          count_t old_exit_node_num = 0;
          count_t temp_old_inner_edge = 0;
          for (auto u : node_set) {
            for (auto e : fragment_->GetOutgoingAdjList(u)) {
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if (to_ids != j) { // 外部点
                frequent[e.neighbor.GetValue()] += 1;
              } else {
                temp_old_inner_edge++;
              }
            }
            if (cpr_->supernode_out_bound[u.GetValue()]) {
              old_exit_node_num++;
            }
          }
          // 分析出现频率
          count_t out_edge_num = 0;
          count_t mirror_node_num = 0;
          count_t old_entry_node_num = cpr_->supernode_source[j].size();
          std::set<vertex_t> old_P;
          old_P.insert(node_set.begin(), node_set.end());
          std::set<vertex_t> P;
          P.insert(node_set.begin(), node_set.end());
          
          for (const auto& fre : frequent) {
            if (fre.second > k) {
              out_edge_num += fre.second;
              mirror_node_num += 1;
              P.insert(vertex_t(fre.first));  // 高频出口点的目的点作为Mirror点
            }
            // LOG(INFO) << fre.first << ": " << fre.second << std::endl;
          }
          // 统计老的出口点
          // std::set<vertex_t> old_B; // belong to P, bound vertices
          // for(auto v : old_P){ // 遍历源来的点集
          //   const auto& oes = fragment_->GetOutgoingAdjList(v);
          //   for(auto& e : oes){
          //       if(old_P.find(e.neighbor) == old_P.end()){ // 包含Mirror
          //           old_B.insert(v);
          //           break;
          //       }
          //   }
          // }
          // 统计新的出口点
          std::set<vertex_t> B; // belong to P, bound vertices
          for(auto v : node_set){ // 遍历源来的点集
            const auto& oes = fragment_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(P.find(e.neighbor) == P.end()){ // 包含Mirror
                    B.insert(v);
                    break;
                }
            }
          }
          std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
          for(auto v : node_set){ // 遍历源来的点集
              const auto& oes = fragment_->GetIncomingAdjList(v);
              for(auto& e : oes){
                  if(old_P.find(e.neighbor) == old_P.end()){ // 不包含Mirror
                      S.insert(v);
                      break;
                  }
              }
          }
          // 统计最终结果
          count_t new_exit_node_num = B.size() + mirror_node_num;
          count_t new_entry_node_num = S.size();
          count_t temp_old_index_num = old_exit_node_num * old_entry_node_num;
          count_t temp_new_index_num = new_exit_node_num * new_entry_node_num;

          const bool original_compress_condition = 
            (temp_old_index_num < temp_old_inner_edge);
          const bool mirror_compress_condition = 
            (temp_new_index_num + mirror_node_num 
              < temp_old_inner_edge + out_edge_num); // 加Mirror后是否应该压缩

          // 统计未能压缩的点和边，即放弃的cluster
          if ( original_compress_condition == false 
            && mirror_compress_condition == false) {
              abandon_edge_num += temp_old_inner_edge;
              abandon_node_num += old_P.size();
          }

          // 不加mirror点的情况
          if (original_compress_condition == true) {
            spnids_num++;
            old_inner_edge += temp_old_inner_edge;
            old_index_num += temp_old_index_num;
            all_old_entry_node_num += old_entry_node_num;
            all_old_exit_node_num += old_exit_node_num;
            new_inner_edge += temp_old_inner_edge;
            new_index_num += temp_old_index_num;
            all_new_entry_node_num += new_entry_node_num;
            all_new_exit_node_num += new_exit_node_num;
          }
          // 加mirror点的情况
          if (mirror_compress_condition == true) { // 加Mirror后是否应该压缩
            if (original_compress_condition == false) { // 未加Mirror时是否压缩了
              add_spnids_num++; // 仅仅因为加mirror才成为cluster
              new_inner_edge += temp_old_inner_edge;
              all_new_entry_node_num += new_entry_node_num;
              all_new_exit_node_num += new_exit_node_num;
            } else {
              new_index_num -= temp_old_index_num;
            }
            const bool f2 = (out_edge_num + temp_old_index_num 
              > mirror_node_num + temp_new_index_num); // 相对于普通压缩而言，加了Mirror后是否有收益
            if (f2) {
              mirror_num += mirror_node_num;
              reduce_edge_num += out_edge_num;
              new_index_num += temp_new_index_num; // 使用新方案
            } else {
              new_index_num += temp_old_index_num; // 依然老方案
            }
          } else {

          }

          // LOG(INFO) << "test---" << "temp_old_inner_edge=" << temp_old_inner_edge
            // << " new_inner_edge=" << new_inner_edge;
          // LOG(INFO) << "mirror_node_num=" << mirror_node_num;
          // LOG(INFO) << "------- out_edge_num=" << out_edge_num 
          //   << " new_exit_node_num=" << new_exit_node_num 
          //   << " new_entry_node_num=" << new_entry_node_num 
          //   << " old_exit_node_num=" << old_exit_node_num 
          //   << " old_entry_node_num=" << old_entry_node_num;
        }
        LOG(INFO) << "k=" << k << " mirror_num= " << mirror_num 
          << " reduce_edge_num=" << reduce_edge_num 
          << " new_index_num=" << new_index_num
          << " old_index_num=" << old_index_num;
        LOG(INFO) << "spnids_num=" << spnids_num 
          << " add_spnids_num=" << add_spnids_num
          << " new_inner_edge=" << new_inner_edge
          << " old_inner_edge=" << old_inner_edge;
        LOG(INFO) << "all_old_entry_node_num=" << all_old_entry_node_num
          << " all_old_exit_node_num=" << all_old_exit_node_num
          << " all_new_entry_node_num=" << all_new_entry_node_num
          << " all_new_exit_node_num=" << all_new_exit_node_num;
        LOG(INFO) << "abandon_edge_num=" << abandon_edge_num 
          << " abandon_node_num=" << abandon_node_num;
      }
      // debug  (入口点+出口点)统计如果采用Mirror-Master能对边的减少率能提高多少, 前期没有入口*出口过滤，在此处过滤
      if (true) {
        LOG(INFO) << "------Mirror-Master---entry+exit_node: no-filter on early stage------";
        typedef long long count_t;
        count_t k = FLAGS_mirror_k; // 阈值
        count_t mirror_num = 0;
        count_t reduce_edge_num = 0;
        count_t new_index_num = 0;
        count_t old_index_num = 0;
        count_t old_inner_edge = 0;
        count_t new_inner_edge = 0;
        count_t spnids_num = 0;
        count_t add_spnids_num = 0;
        count_t all_old_exit_node_num = 0;
        count_t all_new_exit_node_num = 0;
        count_t all_old_entry_node_num = 0;
        count_t all_new_entry_node_num = 0;
        count_t abandon_node_num = 0; // 不满足入口*出口舍弃的点
        count_t abandon_edge_num = 0;
        const vid_t spn_ids_num = cpr_->supernode_ids.size(); 
        // std::vector<vertex_t> abandon_node_set; // 被解压的顶点
        // std::vector<count_t> abandon_node_set_size; // 记录内解压顶点的大小 
        for (vid_t j = 0; j < spn_ids_num; j++){
          std::vector<vertex_t> &node_set = cpr_->supernode_ids[j];
          // 统计所有入口点/出口点的源顶点
          std::unordered_map<vid_t, size_t> in_frequent;
          std::unordered_map<vid_t, size_t> out_frequent;
          count_t old_exit_node_num = 0;
          count_t temp_old_inner_edge = 0;
          for (auto u : node_set) {
            for (auto e : fragment_->GetIncomingAdjList(u)) {
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if (to_ids != j) { // 外部点
                in_frequent[e.neighbor.GetValue()] += 1;
              } else {
                temp_old_inner_edge++;
              }
            }
            for (auto e : fragment_->GetOutgoingAdjList(u)) {
              vid_t to_ids = cpr_->id2spids[e.neighbor];
              if (to_ids != j) { // 外部点
                out_frequent[e.neighbor.GetValue()] += 1;
              }
            }
            if (cpr_->supernode_out_bound[u.GetValue()]) {
              old_exit_node_num++;
            }
          }
          // 分析出现频率
          count_t in_edge_num = 0;
          count_t out_edge_num = 0;
          count_t in_mirror_node_num = 0;
          count_t out_mirror_node_num = 0;
          count_t old_entry_node_num = cpr_->supernode_source[j].size();
          std::set<vertex_t> old_P;
          old_P.insert(node_set.begin(), node_set.end());
          std::set<vertex_t> in_P;
          in_P.insert(node_set.begin(), node_set.end());
          std::set<vertex_t> out_P;
          out_P.insert(node_set.begin(), node_set.end());
          
          for (const auto& fre : in_frequent) {
            if (fre.second > k) {
              in_edge_num += fre.second;
              in_mirror_node_num += 1;
              in_P.insert(vertex_t(fre.first));  // 高频入口点的源顶点作为Mirror点
            }
            // LOG(INFO) << "in: " << fre.first << ": " << fre.second << std::endl;
          }
          for (const auto& fre : out_frequent) {
            if (fre.second > k) {
              out_edge_num += fre.second;
              out_mirror_node_num += 1;
              out_P.insert(vertex_t(fre.first));  // 高频出口点的源顶点作为Mirror点
            }
            // LOG(INFO) << "out: " << fre.first << ": " << fre.second << std::endl;
          }
          // 统计老的出口点
          // std::set<vertex_t> old_B; // belong to P, bound vertices
          // for(auto v : old_P){ // 遍历源来的点集
          //   const auto& oes = fragment_->GetOutgoingAdjList(v);
          //   for(auto& e : oes){
          //       if(old_P.find(e.neighbor) == old_P.end()){ // 包含Mirror
          //         old_B.insert(v);
          //         reak;
          //       }
          //   }
          // }

          // 统计新的出口点
          std::set<vertex_t> B; // belong to P, bound vertices
          for(auto v : node_set){ // 遍历源来的点集
            const auto& oes = fragment_->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(out_P.find(e.neighbor) == out_P.end()){ // 不包含Mirror
                  B.insert(v);
                  break;
                }
            }
          }
          // 统计新的入口点
          std::set<vertex_t> S; // belong to P, with vertices of incoming edges from the outside
          for(auto v : node_set){ // 遍历源来的点集
              const auto& oes = fragment_->GetIncomingAdjList(v);
              for(auto& e : oes){
                  if(in_P.find(e.neighbor) == in_P.end()){ // 包含Mirror
                    S.insert(v);
                    break;
                  }
              }
          }
          // 统计最终结果
          count_t new_exit_node_num = B.size() + out_mirror_node_num;
          count_t new_entry_node_num = S.size() + in_mirror_node_num;
          count_t temp_old_index_num = old_exit_node_num * old_entry_node_num;
          count_t temp_entry_index_num = old_exit_node_num * new_entry_node_num;
          count_t temp_exit_index_num = new_exit_node_num * old_entry_node_num;
          count_t temp_new_index_num = new_exit_node_num * new_entry_node_num;

          // LOG(INFO) << "||||||||| temp_old_index_num=" << temp_old_index_num;
          // LOG(INFO) << "||||||||| temp_entry_index_num=" << temp_entry_index_num;
          // LOG(INFO) << "||||||||| temp_exit_index_num=" << temp_exit_index_num;
          // LOG(INFO) << "||||||||| temp_new_index_num=" << temp_new_index_num;

          const bool original_compress_condition = 
            (temp_old_index_num < temp_old_inner_edge);
          // const bool mirror_compress_condition = 
          //   (temp_new_index_num + in_mirror_node_num + out_mirror_node_num
          //     < temp_old_inner_edge + in_edge_num + out_edge_num); // 加Mirror后是否应该压缩

          std::vector<count_t> benefit;
          benefit.resize(4);
          benefit[0] = temp_old_inner_edge - temp_old_index_num; // 不加mirror
          benefit[1] = temp_old_inner_edge + in_edge_num 
            - (temp_entry_index_num + in_mirror_node_num); // 加入口点mirror
          benefit[2] = temp_old_inner_edge + out_edge_num 
            - (temp_exit_index_num + out_mirror_node_num); // 加出口点mirror
          benefit[3] = temp_old_inner_edge + in_edge_num + out_edge_num - 
            (temp_new_index_num + in_mirror_node_num + out_mirror_node_num); // 入+出miiror

          int max_i = 0;
          for (int i = 0; i < benefit.size(); i++) {
            // LOG(INFO) << "benefit[" << i << "]=" << benefit[i];
            if (benefit[max_i] < benefit[i]) {
              max_i = i;
            }
          }
          count_t max_benefit = benefit[max_i];
          // LOG(INFO) << "== max_i=" << max_i << " max_benefit=" << max_benefit;

          // 统计未能压缩的点和边，即放弃的cluster
          if (max_benefit <= 0) {
              abandon_edge_num += temp_old_inner_edge;
              abandon_node_num += old_P.size();
              // abandon_node_set.insert(abandon_node_set.end(), node_set.begin(), node_set.end());
              // abandon_node_set_size.emplace_back(abandon_node_set.size());
              // std::cout << "--------------\n node_set.size=" << old_P.size() << std::endl;
              // std::cout << "temp_old_inner_edge=" << temp_old_inner_edge << std::endl;
              // std::cout << "temp_old_index_num=" << temp_old_index_num << std::endl;
              // std::cout << " old_entry_node_num=" << old_entry_node_num
              //   << " old_exit_node_num=" << old_exit_node_num << std::endl;
              // std::cout << "temp_new_index_num=" << temp_new_index_num << std::endl;
              // std::cout << " new_entry_node_num=" << new_entry_node_num
              //   << " new_exit_node_num=" << new_exit_node_num << std::endl;
              // std::cout << " in_mirror_node_num=" << in_mirror_node_num
              //   << " out_mirror_node_num=" << out_mirror_node_num << std::endl;
          }

          // 不加mirror点的情况
          if (original_compress_condition == true) {
            spnids_num++;
            old_inner_edge += temp_old_inner_edge;
            old_index_num += temp_old_index_num;
            all_old_entry_node_num += old_entry_node_num;
            all_old_exit_node_num += old_exit_node_num;
          }
          // 四种方案中选择一种
          if (max_benefit > 0) {
            if (original_compress_condition == false) { // 未加Mirror时未压缩
              add_spnids_num++; // 仅仅因为加mirror才成为cluster
            } 
            new_inner_edge += temp_old_inner_edge;
            if (max_i == 0) { // 不加mirror
              new_index_num += temp_old_index_num;
              all_new_entry_node_num += old_entry_node_num;
              all_new_exit_node_num += old_exit_node_num;
            } else if (max_i == 1) { // 加入口点mirror
              mirror_num += in_mirror_node_num;
              reduce_edge_num += in_edge_num;
              new_index_num += temp_entry_index_num;
              all_new_entry_node_num += new_entry_node_num;
              all_new_exit_node_num += old_exit_node_num;
            } else if (max_i == 2) { // 加出口点mirror
              mirror_num += out_mirror_node_num;
              reduce_edge_num += out_edge_num;
              new_index_num += temp_exit_index_num;
              all_new_entry_node_num += old_entry_node_num;
              all_new_exit_node_num += new_exit_node_num;
            } else if (max_i == 3) {
              mirror_num += in_mirror_node_num;
              mirror_num += out_mirror_node_num;
              reduce_edge_num += in_edge_num;
              reduce_edge_num += out_edge_num;
              new_index_num += temp_new_index_num;
              all_new_entry_node_num += new_entry_node_num;
              all_new_exit_node_num += new_exit_node_num;
            } else {
              LOG(INFO) << "no this type. max_i=" << max_i;
              exit(0);
            }
          }

          // LOG(INFO) << "test---" << "temp_old_inner_edge=" << temp_old_inner_edge
          //   << " new_inner_edge=" << new_inner_edge;
          // LOG(INFO) << "in_mirror_node_num=" << in_mirror_node_num;
          // LOG(INFO) << "out_mirror_node_num=" << out_mirror_node_num;
          // LOG(INFO) << "------- out_edge_num=" << out_edge_num 
          //   << " new_exit_node_num=" << new_exit_node_num 
          //   << " new_entry_node_num=" << new_entry_node_num 
          //   << " old_exit_node_num=" << old_exit_node_num 
          //   << " old_entry_node_num=" << old_entry_node_num;
        }
        // print_indegree_outdegree(abandon_node_set);

        LOG(INFO) << "k=" << k << " mirror_num= " << mirror_num 
          << " reduce_edge_num=" << reduce_edge_num 
          << " new_index_num=" << new_index_num
          << " old_index_num=" << old_index_num;
        LOG(INFO) << "spnids_num=" << spnids_num 
          << " add_spnids_num=" << add_spnids_num
          << " new_inner_edge=" << new_inner_edge
          << " old_inner_edge=" << old_inner_edge;
        LOG(INFO) << "all_old_entry_node_num=" << all_old_entry_node_num
          << " all_old_exit_node_num=" << all_old_exit_node_num
          << " all_new_entry_node_num=" << all_new_entry_node_num
          << " all_new_exit_node_num=" << all_new_exit_node_num;
        LOG(INFO) << "abandon_edge_num=" << abandon_edge_num 
          << " abandon_node_num=" << abandon_node_num;
      }
 


      // debug
      {
        vertex_t u(6);
        LOG(INFO) << "id=6, type=" << int(node_type[6]);
        vertex_t source;
        bool native_source = 
                fragment_->GetInnerVertex(FLAGS_sssp_source, source);
        vid_t u_spid = cpr_->id2spids[u];
        vid_t s_spid = cpr_->id2spids[source];
        LOG(INFO) << "u_spid=" << u_spid << " s_spid=" << s_spid;
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

    while (true) {
      // LOG(INFO) << "step=" << step << " curr_modified_.size()=" << app_->curr_modified_.ParallelCount(8);
      exec_time -= GetCurrentTime();
      ++step;

      // if (FLAGS_portion != 1) {
      //   // threshold = Scheduled(100000, 
      //   //                       VertexRange<vid_t>(node_range[0], node_range[4]));
      //   threshold = step + 10;
      //   LOG(INFO) << "step=" << step << " set.size()=" 
      //             << app_->curr_modified_.ParallelCount(8) 
      //             << " threshold=" << threshold;
      // }
      
      auto inner_vertices = fragment_->InnerVertices();
      auto outer_vertices = fragment_->OuterVertices();

      messages_.StartARound();
      app_->next_modified_.ParallelClear(thread_num());

      {
        messages_.ParallelProcess<fragment_t, DependencyData<vid_t, value_t>>(
            thread_num(), *fragment_,
            [this](int tid, vertex_t v,
                   const DependencyData<vid_t, value_t>& msg) {
              if (app_->AccumulateToAtomic(v, msg)) {
                app_->curr_modified_.Insert(v);
              }
            });
      }

      // Traverse outgoing neighbors
      for_time -= GetCurrentTime();
      if (FLAGS_cilk) {
        if(compr_stage == false){
          // ForEachCilk(
          ForEachCilkOfBitset(
              app_->curr_modified_, inner_vertices, [this, &compr_stage, &count, &step](int tid, vertex_t u) {
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];
                // LOG(INFO) << "--- step=" << step << " oid=" << fragment_->GetId(u) << " id=" << u.GetValue() << ": value=" << value << " delta=" << delta.value;

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
        if (compr_stage) {
          // type0 and type1: out, bound
          ForEachCilkOfBitset(
            app_->curr_modified_, 
            VertexRange<vid_t>(node_range[0], node_range[2]), 
            [this, &compr_stage, &count, &step, &threshold](int tid, vertex_t u) {
              // We don't cleanup delta with identity element, since we expect
              // the algorithm is monotonic
              auto& delta = app_->deltas_[u];
              auto& value = app_->values_[u];
              // if (delta.value <= threshold) {
                if (app_->CombineValueDelta(value, delta)) { // 这些判断是否有必要!
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, app_->next_modified_);
                  // debug
                  { 
                    #ifdef DEBUG
                    count += oes.Size();
                    #endif
                  }
                }
              // } else {
              //   app_->next_modified_.Insert(u);
              // }
          });
          // type2: source
          ForEachCilkOfBitset(
            app_->curr_modified_, 
            VertexRange<vid_t>(node_range[2], node_range[3]), 
            [this, &compr_stage, &count, &step, &threshold](int tid, vertex_t u) {
              auto& delta = app_->deltas_[u];
              auto& value = app_->values_[u];
              // if (delta.value <= threshold) {
                if (app_->CombineValueDelta(value, delta)) {
                  adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
                  // app_->CombineValueDelta(spnode.data, delta);
                  // debug
                  { 
                    #ifdef DEBUG
                    count += oes.Size();
                    #endif
                  }
                }
              // } else {
              //   app_->next_modified_.Insert(u);
              // }
          });
          // type3: bound + source
          ForEachCilkOfBitset(
            app_->curr_modified_, 
            VertexRange<vid_t>(node_range[3], node_range[4]), 
            [this, &compr_stage, &count, &step, &threshold](int tid, vertex_t u) {
              auto& delta = app_->deltas_[u];
              auto& value = app_->values_[u];
              // if (delta.value <= threshold) {
                if (app_->CombineValueDelta(value, delta)) {
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, app_->next_modified_);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->next_modified_);
                  // debug
                  { 
                    #ifdef DEBUG
                    count += oes_b.Size();
                    count += oes_s.Size();
                    #endif
                  }
                }
              // } else {
              //   app_->next_modified_.Insert(u);
              // }
          });
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
                // debug
                { 
                  #ifdef DEBUG
                  auto oes = fragment_->GetOutgoingAdjList(u);
                  count += oes.Size();
                  #endif
                }
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
                //debug
                  auto& delta = app_->deltas_[u];
                  auto& value = app_->values_[u];
                //   if (delta.value < value && delta.value > step) {
                //     app_->next_modified_.Insert(u);
                //   }
                // else  // ----> debug
                if (type == 0) {
                  /* 0: out node */
                  // auto& value = app_->values_[u];
                  auto last_value = value;
                  // We don't cleanup delta with identity element, since we expect
                  // the algorithm is monotonic
                  // auto& delta = app_->deltas_[u];

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
                    // debug
                    { 
                      #ifdef DEBUG
                      count += oes.Size();
                      #endif
                    }
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
                    // debug
                    { 
                      #ifdef DEBUG
                      count += oes.Size();
                      #endif
                    }
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
                    // debug
                    { 
                      #ifdef DEBUG
                      count += oes_b.Size();
                      count += oes_s.Size();
                      #endif
                    }
                  }
                }
              });
          
        }
      }
      for_time += GetCurrentTime();

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

      if (terminate) {
        if(compr_stage){
          {
            LOG(INFO) << "---------------------compute------------------------";
            LOG(INFO) << "step=" << step << " f_send_delta_num=" << app_->f_send_delta_num;
            LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
            app_->f_send_delta_num = 0;
            app_->node_update_num = 0;
            app_->touch_nodes.ParallelClear(8);
            LOG(INFO) << "---------------------compute------------------------";
          }
          LOG(INFO) << "start correct...";
          // check_result("correct before");
          timer_next("correct deviation");
          compr_stage = false;
          corr_time -= GetCurrentTime();
          /* exchage value and delta of old_id and new_id */
          double exchage_time = GetCurrentTime();
          vid_t inner_node_num = node_range[5];
          parallel_for (vid_t i = node_range[0]; i < node_range[5]; i++){
            vertex_t v(i);
            values_temp[v] = app_->values_[v];
            deltas_temp[v] = app_->deltas_[v];
            // vertex_t p;               
            // fragment_->Gid2Vertex(deltas_temp[v].parent_gid, p);
            // if (p.GetValue() < inner_node_num) {
            //   deltas_temp[v].parent_gid = fragment_->Vertex2Gid(vertex_t(newId2oldId[p.GetValue()]));
            // }
            vid_t& v_parent_gid = deltas_temp[v].parent_gid;
            fid_t v_fid = vm_ptr->GetFidFromGid(v_parent_gid);
            if (v_fid == fid) {
              v_parent_gid = newGid2oldGid[v_parent_gid];
            }
          }
          // Note that it should be reverse mapped to the original value
          parallel_for (vid_t i = node_range[0]; i < node_range[5]; i++){
            vertex_t v(i);
            app_->values_[v] = values_temp[vertex_t(oldId2newId[v.GetValue()])];
            app_->deltas_[v] = deltas_temp[vertex_t(oldId2newId[v.GetValue()])];
          }
          LOG(INFO) << "exchage_time: " << (GetCurrentTime() - exchage_time);
          // check_result("exchage data");

          // supernode send by inner_delta
          LOG(INFO) << "cpr_->supernodes_num=" << cpr_->supernodes_num;
          parallel_for(vid_t j = 0; j < cpr_->supernodes_num; j++){
            supernode_t &spnode = cpr_->supernodes[j];
            auto u = spnode.id;
            auto& value = app_->values_[u];
            if (value != app_->GetIdentityElement()) { // right
              auto& delta = app_->deltas_[u];
              vid_t spid = cpr_->id2spids[u];
              vertex_t p;
              fragment_->Gid2Vertex(delta.parent_gid, p);
              vertex_t source;
              bool native_source = 
                  fragment_->GetInnerVertex(FLAGS_sssp_source, source);
              if (spid != cpr_->id2spids[p] || (native_source && source == p)) { // Only nodes that depend on external nodes need to send
                auto& oes = spnode.inner_delta;
              // if(value != app_->GetIdentityElement() && (cpr_->if_touch[j] || spnode.data != value)){ // error
                app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
              }
            }
          }
          // parallel_for(vid_t j = 0; j < cpr_->supernodes_num; j++){
          //   vertex_t u = source_nodes[j];
          //   auto& value = app_->values_[u];
          //   if (value != app_->GetIdentityElement()) { // right
          //     auto& delta = app_->deltas_[u];
          //     adj_list_index_t oes = adj_list_index_t(is_iindex_offset_[j], is_iindex_offset_[j+1]);
          //     app_->FinalComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
          //   }
          // }
          
          // inner nodes receive
          if (FLAGS_cilk) {
            // inner node & source node
            // parallel_for(vid_t i = inner_vertices.begin().GetValue();
            //              i < inner_vertices.end().GetValue(); i++) {
            //   if (node_type[i] == 2 || node_type[i] == 4) {
            //     vertex_t u(i);
            //     auto& value = app_->values_[u];
            //     auto& delta = app_->deltas_[u];
            //     app_->CombineValueDelta(value, delta);
            //   }
            // }
            std::vector<vertex_t>& nodes_2 = all_nodes[2];
            vid_t size = nodes_2.size();
            parallel_for (vid_t i = 0; i < size; i++){
              vertex_t u(nodes_2[i]);
              auto& value = app_->values_[u];
              auto& delta = app_->deltas_[u];
              app_->CombineValueDelta(value, delta);
            }
            std::vector<vertex_t>& nodes_4 = all_nodes[4];
            vid_t size_4 = nodes_4.size();
            parallel_for (vid_t i = 0; i < size_4; i++){
              vertex_t u(nodes_4[i]);
              auto& value = app_->values_[u];
              auto& delta = app_->deltas_[u];
              app_->CombineValueDelta(value, delta);
            }
          } else {
            // inner node
            ForEach(VertexRange<vid_t>(node_range[4], node_range[5]), [this](int tid, vertex_t u) {
                auto& value = app_->values_[u];
                auto& delta = app_->deltas_[u];
                app_->CombineValueDelta(value, delta);
              }
            );
            // source node
            ForEach(VertexRange<vid_t>(node_range[2], node_range[3]), [this](int tid, vertex_t u) {
                auto& value = app_->values_[u];
                auto& delta = app_->deltas_[u];
                app_->CombineValueDelta(value, delta);
              }
            );
          }

          // check_result("corr before");
          corr_time += GetCurrentTime();
          LOG(INFO) << "#corr_time: " << corr_time;
          LOG(INFO) << "#1st_step: " << step;
          // print_result();
          app_->next_modified_.Swap(app_->curr_modified_);
          // continue;  // Unnecessary!!!
          // break;
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#iter step: " << step;
            LOG(INFO) << "#Batch time: " << exec_time;
            LOG(INFO) << "#for_time: " << for_time;
            for_time = 0;

            LOG(INFO) << "step=" << step << " f_send_delta_num=" << app_->f_send_delta_num;
            LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
            app_->f_send_delta_num = 0;
            app_->node_update_num = 0;
            app_->touch_nodes.ParallelClear(8);
          }
          exec_time = 0;
          step = 1;

          if (!FLAGS_efile_update.empty()) {
            LOG(INFO) << "-------------------------------------------------------------------";
            LOG(INFO) << "-------------------------------------------------------------------";
            LOG(INFO) << "-------------------------------------------------------------------";
            // FLAGS_compress = false; // 测试
            compr_stage = FLAGS_compress; // use supernode
            timer_next("reloadGraph");
            deltaCompute();  // reload graph
            // compr_stage = false; // 测试
            LOG(INFO) << "\n-----load graph finish, app_->next_modified_.size=" << app_->next_modified_.ParallelCount(8);
            timer_next("inc algorithm");
            // break; // 测试
            {
              LOG(INFO) << "---------------------------reloadGraph-------------------------";
              LOG(INFO) << "step=" << step << " f_send_delta_num=" << app_->f_send_delta_num;
              LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
              app_->f_send_delta_num = 0;
              app_->node_update_num = 0;
              app_->touch_nodes.ParallelClear(8);
              LOG(INFO) << "---------------------------reloadGraph-------------------------";
            }

            // 下面是老版本的增量后的统计
            if(compr_stage && 0){
              auto inner_vertices = fragment_->InnerVertices();
              double node_type_time = GetCurrentTime();
              vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
              node_type.clear();
              node_type.resize(inner_node_num, std::numeric_limits<char>::max());
              // bound_node_values.clear();
              // bound_node_values.Init(inner_vertices, app_->default_v());
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
                    auto oes = fragment_->GetOutgoingAdjList(u);
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
                    auto oes = fragment_->GetOutgoingAdjList(u);
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

              double  init_time_1 = GetCurrentTime();
              // spnode_datas.Init(inner_vertices, 0);
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
                  is_e_degree[i+1] = spnode.bound_delta.size();
                  atomic_add(source_e_num, spnode.bound_delta.size());
                }
                if(type == 1 || type == 3){
                  auto oes = fragment_->GetOutgoingAdjList(u);
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
              LOG(INFO) << "inner_node_num=" << fragment_->GetVerticesNum();

              double init_time_2 = GetCurrentTime();
              is_e_.resize(source_e_num);
              ib_e_.resize(bound_e_num);
              is_e_offset_.resize(inner_node_num+1);
              ib_e_offset_.resize(inner_node_num+1);
              LOG(INFO) << "init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

              /* build index/edge */
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
                  auto oes = fragment_->GetOutgoingAdjList(u);
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

              // re-init curr_modified
              app_->curr_modified_.Init(fragment_->Vertices());
              for (auto u : inner_vertices) {  // 可与尝试并行！！！
                if (node_type[u.GetValue()] != 4) {
                  app_->curr_modified_.Insert(u);
                }
              }

              //debug
              {
                const std::vector<vertex_t>& nodes_0 = all_nodes[0];
                vid_t node_0_size = nodes_0.size();
                size_t max_edge_0_num = 0;
                size_t edge_0_num = 0;
                for(vid_t i = 0; i < node_0_size; i++){
                  vertex_t u(nodes_0[i]);
                  auto oes = fragment_->GetOutgoingAdjList(u);
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

              LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time);

              exec_time -= GetCurrentTime();
              /* The supernode entry vertex sends one round unconditionally. */
              ForEach(inner_vertices, [this](int tid, vertex_t u) {
                auto& delta = app_->deltas_[u];
                // 需要将超点内部消息传出去！！！
                if ((node_type[u.GetValue()] == 1 || node_type[u.GetValue()] == 3) && delta.value != app_->GetIdentityElement()) {
                    auto& value = app_->values_[u];
                    // We don't cleanup delta with identity element, since we expect
                    // the algorithm is monotonic
                    adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                    app_->Compute(u, value, delta, oes, app_->next_modified_);
                  }
                }
              );
              exec_time += GetCurrentTime();
            }

            // 新版本重排序
            if(compr_stage){
      double extra_all_time = GetCurrentTime();
      auto inner_vertices = fragment_->InnerVertices();
      double node_type_time = GetCurrentTime();
      vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
      node_type.clear();
      node_type.resize(inner_node_num, std::numeric_limits<char>::max());
      // bound_node_values.clear();
      // bound_node_values.Init(inner_vertices, app_->default_v());
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
      LOG(INFO) << " node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

      /* renumber internal vertices */
      oldId2newId.clear();
      oldId2newId.resize(inner_node_num);
      newId2oldId.clear();
      newId2oldId.resize(inner_node_num);
      oldGid2newGid.clear();
      oldGid2newGid.resize(inner_node_num); // 分布式情况下，可能会越界，换成unordered_map？？？
      newGid2oldGid.clear();
      newGid2oldGid.resize(inner_node_num);
      node_range.clear();
      node_range.resize(6);
      vid_t index_id = 0;
      // for (int i = 0; i < 5; i++) {
      //   index_id[i+1] = index_id[i] + all_nodes[i].size();
      // }
      for (vid_t i = 0; i < 5; i++) {
        const std::vector<vertex_t>& nodes = all_nodes[i];
        size_t size = nodes.size();
        node_range[i] = index_id;
        parallel_for (vid_t j = 0; j < size; j++) {
          oldId2newId[nodes[j].GetValue()] = index_id + j;
          newId2oldId[index_id + j] = nodes[j].GetValue();
          // 需要加个判断，只转化本地gid
          vid_t old_gid = fragment_->Vertex2Gid(nodes[j]);
          vid_t new_gid = fragment_->Vertex2Gid(vertex_t(index_id + j));
          oldGid2newGid[old_gid] = new_gid;
          newGid2oldGid[new_gid] = old_gid;
        }
        index_id += size;
      }
      node_range[5] = index_id;

      // debug
      { 
#ifdef DEBUG
        for (int i = 0; i < 5; i++) {
          LOG(INFO) << "node_" << i << "=" << all_nodes[i].size();
          LOG(INFO) << "node_range[" << i << "]=" << node_range[i];
        }

        LOG(INFO) << "node_4(inner)/all_node=" << (all_nodes[4].size() * 1.0 / inner_node_num);

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
            auto oes = fragment_->GetOutgoingAdjList(u);
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
            auto oes = fragment_->GetOutgoingAdjList(u);
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
#endif
      }

      double  transfer_csr_time = GetCurrentTime();

      /* source to in_bound_node */

      double  init_time_1 = GetCurrentTime();
      // spnode_datas.Init(inner_vertices, 0);
      is_e_.clear();
      is_e_offset_.clear();
      ib_e_.clear();
      ib_e_offset_.clear();
      size_t source_e_num = 0;
      size_t bound_e_num = 0;
      std::vector<size_t> is_e_degree(inner_node_num+1, 0);
      std::vector<size_t> ib_e_degree(inner_node_num+1, 0);
      LOG(INFO) << " init_time_1=" << (GetCurrentTime()-init_time_1); //1.05179


      double csr_time_1 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        char type = node_type[i];
        if(type == 2 || type == 3){ // index
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          // is_e_degree[i+1] = spnode.bound_delta.size();
          is_e_degree[oldId2newId[i]+1] = spnode.bound_delta.size();
        }
        if(type == 1 || type == 3){ // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          size_t temp_cnt = 0;
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              // bound_e_num += 1;
              temp_cnt += 1;
            }
          }
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] = temp_cnt;
        } else if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          // ib_e_degree[i+1] += temp_cnt;
          ib_e_degree[oldId2newId[i]+1] = temp_cnt;
        }
      }
      LOG(INFO) << " csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
      LOG(INFO) << " index_time=" << (GetCurrentTime()-index_time); //0.226317

#ifdef DEBUG
      LOG(INFO) << "inner_node_num=" << inner_node_num;
      LOG(INFO) << "inner_node_num=" << fragment_->GetVerticesNum();
#endif

      double init_time_2 = GetCurrentTime();
      is_e_.resize(source_e_num);
      ib_e_.resize(bound_e_num);
      is_e_offset_.resize(inner_node_num+1);
      ib_e_offset_.resize(inner_node_num+1);
      LOG(INFO) << " init_time_2=" << (GetCurrentTime()-init_time_2); //1.13601

      /* build index/edge */
      double csr_time_2 = GetCurrentTime();
      // for(auto u : inner_vertices){
      parallel_for(vid_t i = inner_vertices.begin().GetValue();
          i < inner_vertices.end().GetValue(); i++) {
        vertex_t u(i);
        /* source node */
        // vid_t index_s = is_e_degree[i];
        // is_e_offset_[u.GetValue()] = &is_e_[index_s];
        vid_t new_id = oldId2newId[i];
        vid_t index_s = is_e_degree[new_id];
        is_e_offset_[new_id] = &is_e_[index_s];
        char type = node_type[i];
        if(type == 2 || type == 3){ // index
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          for(auto& oe : spnode.bound_delta){
            // is_e_[index_s] = nbr_index_t(oe.first.GetValue(), oe.second);
            // is_e_[index_s].neighbor = oe.first;
            if (oe.first.GetValue() < inner_node_num) {
              is_e_[index_s].neighbor = oldId2newId[oe.first.GetValue()];
            } else {
              is_e_[index_s].neighbor = oe.first;
            }
            // The dependent parent id also donot needs to be updated, 
            // because it is gid.
            is_e_[index_s].data = oe.second;
            if (oe.second.parent_gid < inner_node_num) {
              is_e_[index_s].data.parent_gid = oldId2newId[oe.second.parent_gid];
            }
            index_s++;
          }
        }
        /* inner_bound node */
        // vid_t index_b = ib_e_degree[i];
        vid_t index_b = ib_e_degree[new_id];
        ib_e_offset_[new_id] = &ib_e_[index_b];
        if(type == 1 || type == 3){ // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          vid_t ids_id = cpr_->id2spids[u];
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            if(ids_id != cpr_->id2spids[e.neighbor]){
              ib_e_[index_b] = e;
              auto nbr = ib_e_[index_b].neighbor;
              if (nbr.GetValue() < inner_node_num) {
                ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
              }
              index_b++;
            }
          }
        }
        if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            auto nbr = ib_e_[index_b].neighbor;
            if (nbr.GetValue() < inner_node_num) {
              ib_e_[index_b].neighbor = oldId2newId[nbr.GetValue()];
            }
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << " csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

      // copy from graph SumInc/grape/fragment/immutable_edgecut_fragment.h
      // test cache
      {
        // ib_e_offset_ should have been sorted.
        // for (vid_t i = 0; i < inner_node_num; ++i) {
        //   std::sort(ib_e_offset_[i], ib_e_offset_[i + 1],
        //           [](const nbr_t& lhs, const nbr_t& rhs) {
        //             return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
        //           });
        // }
        parallel_for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }

      // GetInnerIndex(); // build inner index's csr

      /* exchage value and delta of old_id and new_id */
      double exchage_time = GetCurrentTime();
      parallel_for (vid_t i = 0; i < inner_node_num; i++) {
        vertex_t v(i);
        values_temp[v] = app_->values_[v];
        deltas_temp[v] = app_->deltas_[v];
        // vertex_t p;                            
        // fragment_->Gid2Vertex(deltas_temp[v].parent_gid, p);             
        // if (p.GetValue() < inner_node_num) {               
        //   deltas_temp[v].parent_gid = fragment_->Vertex2Gid(vertex_t(newId2oldId[p.GetValue()]));             
        // }
        deltas_temp[v].parent_gid = oldGid2newGid[deltas_temp[v].parent_gid];
      }
      parallel_for (vid_t i = 0; i < inner_node_num; i++) {
        vertex_t v(i);
        app_->values_[vertex_t(oldId2newId[i])] = values_temp[v];
        app_->deltas_[vertex_t(oldId2newId[i])] = deltas_temp[v];
      }
      LOG(INFO) << "#exchage_time: " << (GetCurrentTime()- exchage_time); //3.88149

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149

      {
        // check_result("init before");
        exec_time -= GetCurrentTime();
        // Update the source id to the new id
        vertex_t source;
        bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
        if (native_source) {
          vid_t new_source_id = oldId2newId[source.GetValue()];
          app_->curr_modified_.Insert(vertex_t(new_source_id));
          LOG(INFO) << "supernode... newid=" << new_source_id << " type4=" << node_range[4];
          LOG(INFO) << "this->Fc[source]=" << cpr_->Fc[source];
          // if (new_source_id >= node_range[4] || (new_source_id >= node_range[1] && new_source_id < node_range[2])) {
          if (new_source_id > node_range[0]) {
            LOG(INFO) << "Send one round for supernode...";
            /* send one round */
            if (FLAGS_cilk) {
              /* type=1: bound node */
              parallel_for(vid_t i = node_range[1]; i < node_range[2]; i++){
                vertex_t u(i);
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, app_->curr_modified_);
                }
              }
              /* type=3: bound + soure node */
              parallel_for(vid_t i = node_range[3]; i < node_range[4]; i++){
                vertex_t u(i);
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, app_->curr_modified_);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->curr_modified_);
                }
              }
            } else {
              ForEach(VertexRange<vid_t>(node_range[1], node_range[2]), [this](int tid, vertex_t u) {
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, app_->curr_modified_);
                }
              });
              ForEach(VertexRange<vid_t>(node_range[3], node_range[4]), [this](int tid, vertex_t u) {
                auto& delta = app_->deltas_[u];
                if (delta.value != app_->GetIdentityElement()) {
                  auto& value = app_->values_[u];
                  app_->CombineValueDelta(value, delta);
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, app_->curr_modified_);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, app_->curr_modified_);
                }
              });
            }
          }
        }
        exec_time += GetCurrentTime();
        LOG(INFO) << " pre_exec_time=" << exec_time;
        // check_result("init after");
        LOG(INFO) << "init after bitset.size=" << app_->curr_modified_.ParallelCount(thread_num());
      }
      
      LOG(INFO) << "extra_all_time=" << (GetCurrentTime()- extra_all_time);

      //debug
      {
#ifdef DEBUG
        const std::vector<vertex_t>& nodes_0 = all_nodes[0];
        vid_t node_0_size = nodes_0.size();
        size_t max_edge_0_num = 0;
        size_t edge_0_num = 0;
        for(vid_t i = 0; i < node_0_size; i++){
          vertex_t u(nodes_0[i]);
          auto oes = fragment_->GetOutgoingAdjList(u);
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
#endif
      }

      exec_time -= GetCurrentTime();
      // 连活跃队列都需要切换成新的id
      {
        ForEach(app_->next_modified_, inner_vertices,
              [&channels, vm_ptr, fid, this](int tid, vertex_t v) {
                app_->curr_modified_.Insert(vertex_t(oldGid2newGid[v.GetValue()]));
              });
      }
      /* The supernode bound vertex sends one round unconditionally. */
      ForEach(inner_vertices, [this](int tid, vertex_t u) {
        auto& delta = app_->deltas_[u];
        // 需要将超点内部消息传出去！！！
        if (((u.GetValue() >= node_range[1] && u.GetValue() < node_range[2]) || (u.GetValue() >= node_range[2] && u.GetValue() < node_range[3])) && delta.value != app_->GetIdentityElement()) {
            auto& value = app_->values_[u];
            // We don't cleanup delta with identity element, since we expect
            // the algorithm is monotonic
            adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
            app_->Compute(u, value, delta, oes, app_->curr_modified_);
          }
        }
      );
      exec_time += GetCurrentTime();
      continue; // 已经完成了curr_modified_/next_modified_的切换，不需要在交换了
    }
          } else {
            LOG(ERROR) << "Missing efile_update or efile_updated";
            break;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#Inc iter step: " << step;
            LOG(INFO) << "#Inc time: " << exec_time << " sec";
            LOG(INFO) << "#for_time_inc: " << for_time;
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

    // Analysis result
    double d_sum = 0;
    vertex_t source;
    bool native_source =
        fragment_->GetInnerVertex(FLAGS_sssp_source, source);
    vid_t max_id = native_source ? source.GetValue() : 0;
    for (auto v : fragment_->InnerVertices()) {
      if (app_->values_[v] < app_->GetIdentityElement()) {
        d_sum += app_->values_[v];
        if (app_->values_[v] > app_->values_[vertex_t(max_id)]) {
          max_id = v.GetValue();
        }
      }
    }
    LOG(INFO) << "max_d[" << fragment_->GetId(vertex_t(max_id)) << "]=" << app_->values_[vertex_t(max_id)];
    LOG(INFO) << "d_sum=" << d_sum;
    printf("#d_sum: %.10lf\n", d_sum);
    LOG(INFO) << "count=" << count;
    LOG(INFO) << "exec_time=" << exec_time;
    check_result("check finial realut");


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

  void print_result(){
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    LOG(INFO) << "-----------result---s------------";
    for (auto v : inner_vertices) {
      vertex_t p;
      fragment_->Gid2Vertex(deltas[v].parent_gid, p);
      LOG(INFO) << "oid=" << fragment_->GetId(v) << " id=" << v.GetValue() << ": value=" << values[v] << " delta=" << deltas[v].value << " parent_gid=" << p.GetValue() << std::endl;
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
      value_sum += values[v];
      delta_sum += deltas[v].value;
    }
    printf("---value_sum=%.10lf\n", value_sum);
    printf("---delta_sum=%.10lf\n", delta_sum);
  }

  void Output(std::ostream& os) {
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;

    // for (auto v : inner_vertices) {
    //   os << fragment_->GetId(v) << " " << values[v] << std::endl;
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
        char type = node_type[newId2oldId[u.GetValue()]];
        if (type == 0 || type == 1) {
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          edge_num += oes.Size();
          for (auto e : oes) {
            os << fragment_->GetId(vertex_t(newId2oldId[u.GetValue()])) << " "
               << fragment_->GetId(vertex_t(newId2oldId[e.neighbor.GetValue()])) 
               << " " << e.data << std::endl;
          }
        } else if (type == 2) {
          adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          edge_num += oes.Size();
          for (auto e : oes) {
            os << fragment_->GetId(vertex_t(newId2oldId[u.GetValue()])) << " " 
               << fragment_->GetId(vertex_t(newId2oldId[e.neighbor.GetValue()])) 
               << " " << e.data.value << std::endl;
          }
        } else if (type == 3) {
          adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          edge_num += oes_b.Size();
          for (auto e : oes_b) {
            os << fragment_->GetId(vertex_t(newId2oldId[u.GetValue()])) << " " 
               << fragment_->GetId(vertex_t(newId2oldId[e.neighbor.GetValue()])) 
               << " " << e.data << std::endl;
          }
          adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          edge_num += oes_s.Size();
          for (auto e : oes_s) {
            os << fragment_->GetId(vertex_t(newId2oldId[u.GetValue()])) << " " 
               << fragment_->GetId(vertex_t(newId2oldId[e.neighbor.GetValue()])) 
               << " " << e.data.value << std::endl;
          }
        }
      }
      LOG(INFO) << "edge_num=" << edge_num;
    }
  }

  void Finalize() { messages_.Finalize(); }

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
};

}  // namespace grape

#endif