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

      ForEachSimple(curr_modified, inner_vertices,
                    [this, &next_modified](int tid, vertex_t u) {
                      auto u_gid = fragment_->Vertex2Gid(u);
                      auto oes = fragment_->GetOutgoingAdjList(u);

                      for (auto e : oes) {
                        auto v = e.neighbor;

                        if (app_->DeltaParentGid(v) == u_gid && u != v) { // 注意考虑自环，且是source->source
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
      // LOG(INFO) << "  next_modified.size=" << next_modified.ParallelCount(8);
      // LOG(INFO) << "  curr_modified.size=" << curr_modified.ParallelCount(8);
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

    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    if(FLAGS_compress){
      auto added_edges = inc_fragment_builder.GetAddedEdgesGid();
      cpr_->inc_run(deleted_edges, added_edges, new_graph);
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
    app_->curr_modified_.Swap(next_modified);

    LOG(INFO) << " app_->curr_modified_.size()=" << app_->curr_modified_.ParallelCount(thread_num());
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

    if(compr_stage){
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
        if(type == 2 || type == 3){ // index
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          is_e_degree[i+1] = spnode.bound_delta.size();
        }
        if(type == 1 || type == 3){ // edge
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
        } else if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] += temp_cnt;
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
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
        if(type == 2 || type == 3){ // index
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
        if(type == 1 || type == 3){ // edge
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
        if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

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
        for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149

      // re-init curr_modified
      // app_->curr_modified_.Init(fragment_->Vertices());
      // for (auto u : inner_vertices) {  // 可与尝试并行！！！
      //   if (node_type[u.GetValue()] != 4) {
      //     app_->curr_modified_.Insert(u);
      //   }
      // }
      {
        exec_time -= GetCurrentTime();
        auto inner_vertices = fragment_->InnerVertices();
        /* The supernode entry vertex sends one round unconditionally. */
        ForEach(inner_vertices, [this](int tid, vertex_t u) {
          auto& delta = app_->deltas_[u];
          // 需要将超点内部消息传出去！！！
          if ((node_type[u.GetValue()] == 1 || node_type[u.GetValue()] == 3) && delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              // We don't cleanup delta with identity element, since we expect
              // the algorithm is monotonic
              adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
              app_->Compute(u, value, delta, oes, app_->curr_modified_);
            }
          }
        );
        exec_time += GetCurrentTime();
        LOG(INFO) << "pre_exec_time=" << exec_time;
      }
    }

    LOG(INFO) << "compr_stage=" << compr_stage;

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    // debug
    long long count = 0;
    double for_time = 0.d;
    while (true) {
      // LOG(INFO) << "curr_modified_.size()=" << app_->curr_modified_.ParallelCount(8);
      exec_time -= GetCurrentTime();
      ++step;
      
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
              app_->curr_modified_, inner_vertices, [this, &compr_stage, &count](int tid, vertex_t u) {
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];

                if (app_->CombineValueDelta(value, delta)) {
                  app_->Compute(u, last_value, delta, app_->next_modified_);
                  // debug
                  // { 
                  //   auto oes = fragment_->GetOutgoingAdjList(u);
                  //   count += oes.Size();
                  // }
                }
              });
        }
        if (compr_stage) {
          ForEachCilkOfBitset(
              app_->curr_modified_, inner_vertices, [this, &compr_stage, &count, &step](int tid, vertex_t u) {
                char type = node_type[u.GetValue()];
                auto& delta = app_->deltas_[u];
                auto& value = app_->values_[u];
                if (0 == type || 1 == type) {
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
                    // { 
                    //   count += oes.Size();
                    // }
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
                    // { 
                    //   count += oes.Size();
                    // }
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
                    // { 
                    //   count += oes_b.Size();
                    //   count += oes_s.Size();
                    // }
                  }
                }
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
                // { 
                //   auto oes = fragment_->GetOutgoingAdjList(u);
                //   count += oes.Size();
                // }
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
                    // { 
                    //   auto oes = fragment_->GetOutgoingAdjList(u);
                    //   count += oes.Size();
                    // }
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
                    // { 
                    //   count += oes.Size();
                    // }
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
                    // { 
                    //   count += oes.Size();
                    // }
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
                    // { 
                    //   count += oes_b.Size();
                    //   count += oes_s.Size();
                    // }
                  }
                }
              });
          
        }
      }
      for_time += GetCurrentTime();

      auto& channels = messages_.Channels();

      // send local delta to remote
      ForEach(app_->next_modified_, outer_vertices,
              [&channels, this](int tid, vertex_t v) {
                auto& delta_to_send = app_->deltas_[v];

                if (delta_to_send.value != app_->GetIdentityElement()) {
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

      exec_time += GetCurrentTime();

      bool terminate = messages_.ToTerminate();

      if (terminate) {
        if(compr_stage){
          LOG(INFO) << "start correct...";

          // debug
          // {
          //       LOG(INFO) << "=----------worker--s------------=";
          //       vertex_t v(1);
          //       LOG(INFO) << "gid=" << fragment_->Vertex2Gid(v) << " oid=" << fragment_->GetId(v) << " value=" << app_->values_[v] << " delta=" << app_->deltas_[v].value;
          //       LOG(INFO) << "=----------worker--e------------=";
          // }

          timer_next("correct deviation");
          compr_stage = false;
          corr_time -= GetCurrentTime();
          // supernode send by inner_delta
          // for(vid_t j = 0; j < cpr_->supernodes_num; j++){
          LOG(INFO) << "cpr_->supernodes_num=" << cpr_->supernodes_num;
          // granular_for(j, 0, cpr_->supernodes_num, (cpr_->supernodes_num > 1024), {
          parallel_for(vid_t j = 0; j < cpr_->supernodes_num; j++){
            supernode_t &spnode = cpr_->supernodes[j];
            auto u = spnode.id;
            auto& value = app_->values_[u];
            // We don't cleanup delta with identity element, since we expect
            // the algorithm is monotonic
            auto& delta = app_->deltas_[u];
            auto& oes = spnode.inner_delta;

            if (value != app_->GetIdentityElement()) { // right
            // if(value != app_->GetIdentityElement() && (cpr_->if_touch[j] || spnode.data != value)){ // error
              app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
            }
            // app_->CombineValueDelta(spnode.data, delta); // 不需要吧！！！
          }
          // })
          // inner nodes receive
          if (FLAGS_cilk) {
            std::vector<vertex_t>& nodes_4 = all_nodes[4];
            vid_t size = nodes_4.size();
            // parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            parallel_for (vid_t i = 0; i < size; i++){
              vertex_t u(nodes_4[i]);
              auto& value = app_->values_[u];
              auto& delta = app_->deltas_[u];
              app_->CombineValueDelta(value, delta);
            }
          } else {
            ForEach(inner_vertices, [this](int tid, vertex_t u) {
                if (node_type[u.GetValue()] == 4) {
                  auto& value = app_->values_[u];
                  auto& delta = app_->deltas_[u];
                  app_->CombineValueDelta(value, delta);
                }
              }
            );
          }

          // // debug
          // {
          //       LOG(INFO) << "=----------worker--s------------=";
          //       vertex_t v(1);
          //       LOG(INFO) << "gid=" << fragment_->Vertex2Gid(v) << " oid=" << fragment_->GetId(v) << " value=" << app_->values_[v] << " delta=" << app_->deltas_[v].value;
          //       LOG(INFO) << "=----------worker--e------------=";
          // }

          corr_time += GetCurrentTime();
          LOG(INFO) << "#corr_time: " << corr_time;
          LOG(INFO) << "#1st_step: " << step;
          app_->next_modified_.Swap(app_->curr_modified_);
          // print_result();
          continue;  // 似乎不需要continue!!!
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#iter step: " << step;
            LOG(INFO) << "#Batch time: " << exec_time;
          }
          exec_time = 0;
          step = 1;

          if (!FLAGS_efile_update.empty()) {
            // FLAGS_compress = false; // 测试
            compr_stage = FLAGS_compress; // use supernode
            timer_next("reloadGraph");
            deltaCompute();  // reload graph
            // compr_stage = false; // 测试
            timer_next("inc algorithm");
            // print_result();
            // break; // 测试
    if(compr_stage){
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
        for (int i = 0; i < 5; i++) {
          LOG(INFO) << "node_" << i << "=" << all_nodes[i].size();
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
        if(type == 2 || type == 3){ // index
          supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
          is_e_degree[i+1] = spnode.bound_delta.size();
        }
        if(type == 1 || type == 3){ // edge
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
        } else if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          size_t temp_cnt = oes.Size();
          ib_e_degree[i+1] += temp_cnt;
        }
      }
      LOG(INFO) << "csr_time_1=" << (GetCurrentTime()-csr_time_1); //1.25987

      /* get index start */
      double index_time = GetCurrentTime();
      for(vid_t i = 1; i <= inner_node_num; i++) {
        ib_e_degree[i] += ib_e_degree[i-1];
        is_e_degree[i] += is_e_degree[i-1];
      }
      bound_e_num = ib_e_degree[inner_node_num];
      source_e_num = is_e_degree[inner_node_num];
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
        if(type == 2 || type == 3){ // index
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
        if(type == 1 || type == 3){ // edge
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
        if (0 == type) { // edge
          auto oes = fragment_->GetOutgoingAdjList(u);
          auto it = oes.begin();
          auto out_degree = oes.Size();
          for(vid_t j = 0; j < out_degree; j++){
            auto& e = *(it + j);
            ib_e_[index_b] = e;
            index_b++;
          }
        }
      }
      is_e_offset_[inner_node_num] = &is_e_[source_e_num-1] + 1;
      ib_e_offset_[inner_node_num] = &ib_e_[bound_e_num-1] + 1;
      LOG(INFO) << "csr_time_2=" << (GetCurrentTime()-csr_time_2); //0.207281

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
        for (vid_t i = 0; i < inner_node_num; ++i) {
          std::sort(is_e_offset_[i], is_e_offset_[i + 1],
                  [](const nbr_index_t& lhs, const nbr_index_t& rhs) {
                    return lhs.neighbor.GetValue() < rhs.neighbor.GetValue();
                  });
        }
      }

      LOG(INFO) << " transfer_csr_time=" << (GetCurrentTime()- transfer_csr_time); //3.88149

      // re-init curr_modified
      // app_->curr_modified_.Init(fragment_->Vertices());
      // for (auto u : inner_vertices) {  // 可与尝试并行！！！
      //   if (node_type[u.GetValue()] != 4) {
      //     app_->curr_modified_.Insert(u);
      //   }
      // }
      {
        exec_time -= GetCurrentTime();
        auto inner_vertices = fragment_->InnerVertices();
        /* The supernode entry vertex sends one round unconditionally. */
        ForEach(inner_vertices, [this](int tid, vertex_t u) {
          auto& delta = app_->deltas_[u];
          // 需要将超点内部消息传出去！！！
          if ((node_type[u.GetValue()] == 1 || node_type[u.GetValue()] == 3) && delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              // We don't cleanup delta with identity element, since we expect
              // the algorithm is monotonic
              adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
              app_->Compute(u, value, delta, oes, app_->curr_modified_);
            }
          }
        );
        exec_time += GetCurrentTime();
        LOG(INFO) << "pre_exec_time=" << exec_time;
      }

      //debug
      {
        const std::vector<vertex_t>& nodes_0 = all_nodes[0];
        vid_t node_0_size = nodes_0.size();
        size_t max_edge_0_num = 0;
        size_t edge_0_num = 0;
        LOG(INFO) << "-----------------test node_0_edge:=================";
        for(vid_t i = 0; i < node_0_size; i++){
          vertex_t u(nodes_0[i]);
          auto oes = fragment_->GetOutgoingAdjList(u);
          max_edge_0_num = std::max(max_edge_0_num, oes.Size());
          edge_0_num += oes.Size();
          // LOG(INFO) << "---id=" << u.GetValue() << " size=" << oes.Size();
          // adj_list_t oes_2 = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          // LOG(INFO) << "---id=" << u.GetValue() << " oes_2_size=" << oes_2.Size();
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
    }
          } else {
            LOG(ERROR) << "Missing efile_update or efile_updated";
            break;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#Inc iter step: " << step;
            LOG(INFO) << "#Inc time: " << exec_time << " sec";
            // print_result();
          }
          break;
        }
      }

      // {
      //   LOG(INFO) << "----------------------------test-next_modified_------s----------------";
      //   ForEach(app_->next_modified_, inner_vertices,
      //         [&channels, this](int tid, vertex_t v) {
      //           auto& delta_to_send = app_->deltas_[v];

      //           LOG(INFO) << "id=" << v.GetValue() << " value=" << app_->values_[v] << " delta=" << delta_to_send.value;
      //         });
      //   LOG(INFO) << "----------------------------test-next_modified_-----e-----------------";
      // }


      app_->next_modified_.Swap(app_->curr_modified_);
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
    printf("d_sum=%.10lf\n", d_sum);
    LOG(INFO) << "count=" << count;
    LOG(INFO) << "for_time=" << for_time;


    MPI_Barrier(comm_spec_.comm());
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

  void Output(std::ostream& os) {
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;

    // for (auto v : inner_vertices) {
    //   os << fragment_->GetId(v) << " " << values[v] << std::endl;
    // }
    // 将超图写入文件
    {
      if (FLAGS_compress == false) {
        return ;
      }
      LOG(INFO) << "write supergraph...";
      for (auto u : inner_vertices) {
         char type = node_type[u.GetValue()];
        if (type == 0) {
          /* 0: out node */
          auto oes = fragment_->GetOutgoingAdjList(u);
          for (auto oe : oes) {
            os << fragment_->GetId(u) << " " << fragment_->GetId(oe.neighbor) << " " << oe.data << std::endl;
          }
        }
        else if (type == 1) {
          /* 1: bound node */
          adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          for (auto oe : oes) {
            os << fragment_->GetId(u) << " " << fragment_->GetId(oe.neighbor) << " " << oe.data << std::endl;
          }
        }
        else if (type == 2) {
          /* 2: source node */
          adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          for (auto oe : oes) {
            os << fragment_->GetId(u) << " " << fragment_->GetId(oe.neighbor) << " " << oe.data.value << std::endl;
          }
        }
        else if (type == 3) {
          /* 1: bound node */
          adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
          for (auto oe : oes_b) {
            os << fragment_->GetId(u) << " " << fragment_->GetId(oe.neighbor) << " " << oe.data << std::endl;
          }
          /* 2: source node */
          adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          for (auto oe : oes_s) {
            os << fragment_->GetId(u) << " " << fragment_->GetId(oe.neighbor) << " " << oe.data.value << std::endl;
          }
        }
      }
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
  VertexArray<bool, vid_t> is_spnode_in; // 需要删除
  VertexArray<bool, vid_t> is_nornode; // 需要删除
  // VertexArray<delta_t, vid_t> spnode_datas{};
  /* source to in_bound_node: index */
  Array<nbr_index_t, Allocator<nbr_index_t>> is_e_;
  Array<nbr_index_t*, Allocator<nbr_index_t*>> is_e_offset_;
  /* in_bound_node to out_bound_node: original edge */
  Array<nbr_t, Allocator<nbr_t>> ib_e_;
  Array<nbr_t*, Allocator<nbr_t*>> ib_e_offset_;
  /* each type of vertices */
  std::vector<std::vector<vertex_t>> all_nodes;
  std::vector<char> node_type; // all node's types, 0:out node, 1:bound node, 2:source node, 3:belong 1 and 2 at the same time, 4:inner node that needn't send message.

};

}  // namespace grape

#endif