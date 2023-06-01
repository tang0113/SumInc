
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

    // debug
    {
      vertex_t v(2114);
      LOG(INFO) << "----------reset after---------------";
      LOG(INFO) << "value=" <<  app_->values_[v];
      LOG(INFO) << "delta=" <<  app_->deltas_[v].value;
      LOG(INFO) << "  is=" << app_->curr_modified_.Exist(v);
      double delta_sum = 0;
      double value_sum = 0;
      for (auto v : inner_vertices) {
        delta_sum += app_->deltas_[v].value;
        value_sum += app_->values_[v];
      }
      LOG(INFO) << "  delta_sum=" << delta_sum;
      LOG(INFO) << "  value_sum=" << value_sum;
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
    if (FLAGS_compress) {
      app_->reInit(cpr_->all_node_num); // for mirror node
    }

    // copy to new graph
    for (auto v : inner_vertices) {
      app_->values_[v] = values[v];
      app_->deltas_[v] = deltas[v];
    }

    // debug
    {
      vertex_t v(2114);
      LOG(INFO) << "----------reset after---------------";
      LOG(INFO) << "value=" <<  app_->values_[v];
      LOG(INFO) << "delta=" <<  app_->deltas_[v].value;
      LOG(INFO) << "  is=" << app_->curr_modified_.Exist(v);
      double delta_sum = 0;
      double value_sum = 0;
      for (auto v : inner_vertices) {
        delta_sum += app_->deltas_[v].value;
        value_sum += app_->values_[v];
      }
      LOG(INFO) << "  delta_sum=" << delta_sum;
      LOG(INFO) << "  value_sum=" << value_sum;
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

    LOG(INFO) << " app_->curr_modified_.size()=" << app_->curr_modified_.ParallelCount(thread_num());;

    // print_result("提前计算前:");

    // if(FLAGS_compress){
    //   timer_next("inc pre compute");
    //   double inc_pre_compute = GetCurrentTime();
    //   // cpr_->precompute_spnode(fragment_);
    //   // cpr_->precompute_spnode_all(fragment_); // 应该是被touch到的超点需要
    //   cpr_->inc_precompute_spnode_mirror(fragment_, node_type);
    //   inc_pre_compute = GetCurrentTime() - inc_pre_compute;
    //   LOG(INFO) << "#inc_pre_compute: " << inc_pre_compute;
    // }

    // debug
    {
      vertex_t v(2114);
      LOG(INFO) << "----------reset after---------------";
      LOG(INFO) << "value=" <<  app_->values_[v];
      LOG(INFO) << "delta=" <<  app_->deltas_[v].value;
      LOG(INFO) << "  is=" << app_->curr_modified_.Exist(v);
      double delta_sum = 0;
      double value_sum = 0;
      for (auto v : inner_vertices) {
        delta_sum += app_->deltas_[v].value;
        value_sum += app_->values_[v];
      }
      LOG(INFO) << "  delta_sum=" << delta_sum;
      LOG(INFO) << "  value_sum=" << value_sum;
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

    // debug
    {
      vertex_t v(2114);
      LOG(INFO) << "-------------------------";
      LOG(INFO) << "vid=" <<  app_->values_[v];
      LOG(INFO) << "vid=" <<  app_->deltas_[v].value;
    }

    cpr_->sketch2csr_renumber(inner_node_num, node_type, oldId2newId, 
                              newId2oldId, oldGid2newGid, newGid2oldGid,
                              node_range, all_nodes, is_e_, is_e_offset_,
                              ib_e_, ib_e_offset_);

    // GetInnerIndex(); // build inner index's csr

    VertexRange<vid_t> allrange(0, node_range[4]);
    this->curr_modified_new.Init(allrange); // 不加mirror???
    this->next_modified_new.Init(allrange);

    /* exchage value and delta of old_id and new_id */
    double exchage_time = GetCurrentTime();
    fid_t fid = fragment_->fid();
    auto vm_ptr = fragment_->vm_ptr();
    // values_temp.Clear();
    // deltas_temp.Clear();
    app_->values_.Swap(values_temp);
    app_->deltas_.Swap(deltas_temp);
    parallel_for (vid_t i = 0; i < node_range[4]; i++) {
      vertex_t v(i);
      app_->values_[v] = values_temp[vertex_t(newId2oldId[i])];
      app_->deltas_[v] = deltas_temp[vertex_t(newId2oldId[i])];
      vid_t& v_parent_gid = app_->deltas_[v].parent_gid;
      fid_t v_fid = vm_ptr->GetFidFromGid(v_parent_gid);
      if (v_fid == fid) {
        v_parent_gid = oldGid2newGid[v_parent_gid];
      }
    }
    LOG(INFO) << "#exchage_time: " << (GetCurrentTime()- exchage_time);

    for(auto t : node_range) {
      LOG(INFO) << " node_type_range=" << t;
    }

    {
      // check_result("init before");
      exec_time -= GetCurrentTime();
      // app_->next_modified_.Swap(app_->curr_modified_);
      // Update the source id to the new id
      vertex_t source;
      bool native_source = fragment_->GetInnerVertex(FLAGS_sssp_source, source);
      if (native_source) {
        vid_t new_source_id = oldId2newId[source.GetValue()];
        // app_->curr_modified_.Insert(vertex_t(new_source_id));
        app_->curr_modified_.Insert(source); // old source node
        LOG(INFO) << "supernode... newid=" << new_source_id 
                  << " type4=" << node_range[4];
        LOG(INFO) << "this->Fc[source]=" << cpr_->Fc[source];
      }

      LOG(INFO) << "Send one round for supernode...";
      LOG(INFO) << "  active_node.size=" 
                << app_->curr_modified_.ParallelCount(8);
      /* send one round */
      ForEachCilkOfBitset(
        app_->curr_modified_, 
        VertexRange<vid_t>(node_range[0], node_range[5]), 
        [this](int tid, vertex_t u) {
          // LOG(INFO) << " ---------u.oid" << cpr_->v2Oid(u);
          u = vertex_t(oldId2newId[u.GetValue()]);
          if (u.GetValue() < node_range[2]) {
            auto& delta = app_->deltas_[u];
            if (delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              app_->CombineValueDelta(value, delta);
              adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
              app_->Compute(u, value, delta, oes, this->next_modified_new);
            }
          } else if (u.GetValue() < node_range[3]) {
            auto& delta = app_->deltas_[u];
            if (delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              app_->CombineValueDelta(value, delta);
              adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
              app_->ComputeByIndexDelta(u, value, delta, oes, this->next_modified_new);
            }
          } else if (u.GetValue() < node_range[4]) {
            auto& delta = app_->deltas_[u];
            if (delta.value != app_->GetIdentityElement()) {
              auto& value = app_->values_[u];
              app_->CombineValueDelta(value, delta);
              /* 1: bound node */
              adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
              app_->Compute(u, value, delta, oes_b, this->next_modified_new);
              /* 2: source node */
              adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
              app_->ComputeByIndexDelta(u, value, delta, oes_s, this->next_modified_new);
            }
          }
      });
      this->next_modified_new.Swap(this->curr_modified_new);
      exec_time += GetCurrentTime();
      LOG(INFO) << " pre_exec_time=" << exec_time;
      LOG(INFO) << " init after bitset.size=" << app_->curr_modified_.ParallelCount(thread_num());
      LOG(INFO) << " init after curr_modified_new.size=" 
                << this->curr_modified_new.ParallelCount(thread_num());
    }
    
    LOG(INFO) << "extra_all_time=" << (GetCurrentTime()- extra_all_time);
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());

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
      first_step(values_temp, deltas_temp, exec_time, false);
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

    while (true) {
      // LOG(INFO) << "step=" << step << " curr_modified_.size()=" << app_->curr_modified_.ParallelCount(8);
      exec_time -= GetCurrentTime();
      ++step;

      auto inner_vertices = fragment_->InnerVertices();
      auto outer_vertices = fragment_->OuterVertices();

      messages_.StartARound();
      // clean_bitset_time -= GetCurrentTime();
      // app_->next_modified_.ParallelClear(thread_num()); // 对于压缩图清理的范围可以缩小， 直接初始化为小区间！！！！
      if (compr_stage) {
        this->next_modified_new.ParallelClear(thread_num()); // 对于压缩图清理的范围可以缩小， 直接初始化为小区间！！！！
      } else {
        app_->next_modified_.ParallelClear(thread_num()); // ingress
      }
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
            this->curr_modified_new, 
            VertexRange<vid_t>(node_range[0], node_range[2]), 
            [this, &compr_stage, &count, &step, &threshold](int tid, vertex_t u) {
              // We don't cleanup delta with identity element, since we expect
              // the algorithm is monotonic
              auto& delta = app_->deltas_[u];
              auto& value = app_->values_[u];
              // if (delta.value <= threshold) {
                if (app_->CombineValueDelta(value, delta)) { // 这些判断是否有必要!
                  adj_list_t oes = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes, this->next_modified_new);
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
            this->curr_modified_new, 
            VertexRange<vid_t>(node_range[2], node_range[3]), 
            [this, &compr_stage, &count, &step, &threshold](int tid, vertex_t u) {
              auto& delta = app_->deltas_[u];
              auto& value = app_->values_[u];
              // if (delta.value <= threshold) {
                if (app_->CombineValueDelta(value, delta)) {
                  adj_list_index_t oes = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes, this->next_modified_new);
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
            this->curr_modified_new, 
            VertexRange<vid_t>(node_range[3], node_range[4]), 
            [this, &compr_stage, &count, &step, &threshold](int tid, vertex_t u) {
              auto& delta = app_->deltas_[u];
              auto& value = app_->values_[u];
              // if (delta.value <= threshold) {
                if (app_->CombineValueDelta(value, delta)) {
                  /* 1: bound node */
                  adj_list_t oes_b = adj_list_t(ib_e_offset_[u.GetValue()], ib_e_offset_[u.GetValue()+1]);
                  app_->Compute(u, value, delta, oes_b, this->next_modified_new);
                  /* 2: source node */
                  adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
                  app_->ComputeByIndexDelta(u, value, delta, oes_s, this->next_modified_new);
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
                    v_parent_gid = newGid2oldGid[v_parent_gid]; // 针对重编号
                  }
                  channels[tid].SyncStateOnOuterVertex(*fragment_, v,
                                                       delta_to_send);
                }
              });
      
      if (compr_stage) {
        if (!this->next_modified_new.PartialEmpty(0, node_range[4])) {
          messages_.ForceContinue();
        }
      } else {
        if (!app_->next_modified_.PartialEmpty(0, 
                                            fragment_->GetInnerVerticesNum())) {
          messages_.ForceContinue();
        }
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
          parallel_for (vid_t i = node_range[0]; i < node_range[4]; i++){
            vertex_t v(i);
            vertex_t old_v = vertex_t(newId2oldId[v.GetValue()]);
            values_temp[old_v] = app_->values_[v];
            deltas_temp[old_v] = app_->deltas_[v];
            vid_t& v_parent_gid = deltas_temp[old_v].parent_gid;
            fid_t v_fid = vm_ptr->GetFidFromGid(v_parent_gid);
            if (v_fid == fid) {
              v_parent_gid = newGid2oldGid[v_parent_gid];
            }
          }
          app_->values_.Swap(values_temp);
          app_->deltas_.Swap(deltas_temp);
          LOG(INFO) << "exchage_time: " << (GetCurrentTime() - exchage_time);

          // check_result("exchage data");
          // print_result("校正前:");

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
          // print_result();
          // app_->next_modified_.Swap(app_->curr_modified_); // 校正完应该收敛，无须交换吧！！！
          this->next_modified_new.Swap(this->curr_modified_new); // 针对Ingress做动态时, 用这个 
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
            for_time = 0;

            LOG(INFO) << "step=" << step << " f_send_delta_num=" << app_->f_send_delta_num;
            LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.ParallelCount(8);
            app_->f_send_delta_num = 0;
            app_->node_update_num = 0;
            app_->touch_nodes.ParallelClear(8);
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

            // print_result("重新编号前:");

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

      if (compr_stage) {
        this->next_modified_new.Swap(this->curr_modified_new); // 针对Ingress做动态时, 用这个 
      } else {
        app_->next_modified_.Swap(app_->curr_modified_); // 针对Ingress做动态时, 用这个
      }
    }

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
    LOG(INFO) << "d_sum=" << d_sum;
    printf("#d_sum: %.10lf\n", d_sum);
    LOG(INFO) << "count=" << count;
    LOG(INFO) << "#visited_num: " << visited_num;
    LOG(INFO) << "cpr_->old_node_num: " << fragment_->GetVerticesNum() ;
    LOG(INFO) << "#visited_rate: " << (visited_num * 1.0 / fragment_->GetVerticesNum() );
    LOG(INFO) << "exec_time=" << exec_time;
    check_result("check finial realut");

    // debug
    if (FLAGS_compress) {
      for (vid_t i = 0; i < cpr_->old_node_num; i++) {
        vertex_t u(i);
        value_t u_v = app_->values_[u];
        value_t u_d = app_->deltas_[u].value;
        if (u_v != u_d) {
          LOG(INFO) << " u_v=" << u_v << " u_d=" << u_d 
                    << " u_oid=" << cpr_->v2Oid(u);
        }
      }
      vid_t oid = 2195;
      if (oid < cpr_->old_node_num && false) {
        vertex_t u;
        bool native_source = fragment_->GetInnerVertex(oid, u);
        LOG(INFO) << " debug-- vid=" << u.GetValue() << " oid=" << oid;
        LOG(INFO) << " type=" << int(node_type[u.GetValue()]);
        LOG(INFO) << " value=" << app_->values_[u] 
                  << " delta=" << app_->deltas_[u].value;

        {
          vid_t spids_id = cpr_->id2spids[u];
          if (spids_id != cpr_->ID_default_value) {
            for (auto v : cpr_->supernode_ids[spids_id]) {
              LOG(INFO) << " include oid=" << cpr_->v2Oid(v)
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
              vid_t u = oldId2newId[v.GetValue()];
              // adj_list_t oes = adj_list_t(ib_e_offset_[u], ib_e_offset_[u+1]);
              adj_list_index_t oes = adj_list_index_t(is_e_offset_[u], is_e_offset_[u+1]);
              for (auto e : oes) {
                LOG(INFO) << "    " << cpr_->v2Oid(v) << " "
                          << cpr_->v2Oid(vertex_t(newId2oldId[e.neighbor.GetValue()])) 
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
      value_sum += values[v];
      delta_sum += deltas[v].value;
    }
    printf("---value_sum=%.10lf\n", value_sum);
    printf("---delta_sum=%.10lf\n", delta_sum);
  }

  void Output(std::ostream& os) {
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    for (auto v : inner_vertices) {
      // os << fragment_->GetId(v) << " " << values[v] << std::endl;
      os << fragment_->GetId(v) << " " << deltas[v].parent_gid << std::endl;
    }
    return ;
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
          // os << "----------\n";
          adj_list_index_t oes_s = adj_list_index_t(is_e_offset_[u.GetValue()], is_e_offset_[u.GetValue()+1]);
          edge_num += oes_s.Size();
          for (auto e : oes_s) {
            os << fragment_->GetId(vertex_t(newId2oldId[u.GetValue()])) << " " 
               << fragment_->GetId(vertex_t(newId2oldId[e.neighbor.GetValue()])) 
               << " " << e.data.value << std::endl;
          }
        }
        // os << "--------test------\n";
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
  DenseVertexSet<vid_t> curr_modified_new, next_modified_new;
};

}  // namespace grape

#endif