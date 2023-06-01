
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
                      app_->values_[u] = app_->GetIdentityElement();
                      app_->deltas_[u].Reset(app_->GetIdentityElement());
                      reset_vertices.Insert(u);
                    });

      ForEach(next_modified, outer_vertices,
              [&channels, this](int tid, vertex_t v) {
                grape::EmptyType dummy;
                channels[tid].SyncStateOnOuterVertex(*fragment_, v, dummy);
                app_->deltas_[v].Reset(app_->GetIdentityElement());
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

    if(FLAGS_compress){
      timer_next("inc pre compute");
      cpr_->precompute_spnode(fragment_);
    }
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());

    // // allocate dependency arrays
    // app_->Init(comm_spec_, fragment_);
    int step = 0;
    bool batch_stage = true;
    double exec_time = 0;
    bool compr_stage = FLAGS_compress; // true: supernode send

    if(compr_stage){
      auto inner_vertices = fragment_->InnerVertices();
      is_spnode_in.Init(inner_vertices, false);
      is_nornode.Init(inner_vertices, false);
      ForEach(inner_vertices, [this](int tid, vertex_t u) {
          if(cpr_->Fc[u] == cpr_->FC_default_value){
            is_nornode[u] = true;
          }
          else if(cpr_->Fc[u] >= 0){
            is_spnode_in[u] = true;
          }
        }
      );
      // re-init curr_modified
      app_->curr_modified_.Init(fragment_->Vertices());
      for (auto u : inner_vertices) {  // 可与尝试并行！！！
        if ((is_spnode_in[u] || is_nornode[u]) && app_->deltas_[u].value != app_->GetIdentityElement()) {
          app_->curr_modified_.Insert(u);
        }
      }
      // cout send message number
      // LOG(INFO) << "f_send_num_0=" << app_->f_send_num  << " f_send_delta_num_0=" << app_->f_send_delta_num << " sum_0=" << (app_->f_send_num + app_->f_send_delta_num);
      // LOG(INFO) << "g_num_1=" << app_->g_num;
      // app_->f_send_num = 0;
      // app_->f_send_delta_num = 0;
      // app_->g_num = 0;
    }
    LOG(INFO) << "compr_stage=" << compr_stage;

    messages_.Start();

    // Run an empty round, otherwise ParallelProcess will stuck
    messages_.StartARound();
    messages_.InitChannels(thread_num());
    messages_.FinishARound();

    while (true) {
      ++step;
      exec_time -= GetCurrentTime();
      auto inner_vertices = fragment_->InnerVertices();
      auto outer_vertices = fragment_->OuterVertices();

      messages_.StartARound();
      app_->next_modified_.ParallelClear(thread_num());

      {
        messages_.ParallelProcess<fragment_t, DependencyData<vid_t, value_t>>(
            thread_num(), *fragment_,
            [this](int tid, vertex_t v,
                   const DependencyData<vid_t, value_t>& msg) {
              if (app_->AccumulateTo(v, msg)) {
                app_->curr_modified_.Insert(v);
              }
            });
      }

      // Traverse outgoing neighbors
      if (FLAGS_cilk) {
        ForEachCilk(
            app_->curr_modified_, inner_vertices, [this, &compr_stage](int tid, vertex_t u) {
              if(compr_stage && is_spnode_in[u]){
                supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];
                auto& oes = spnode.bound_delta;

                if (app_->CombineValueDelta(value, delta)) {
                  app_->ComputeByIndexDelta(u, last_value, delta, oes, app_->next_modified_);
                  // app_->CombineValueDelta(spnode.data, delta);
                }
              }
              else if(compr_stage == false || is_nornode[u]){
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];

                if (app_->CombineValueDelta(value, delta)) {
                  app_->Compute(u, last_value, delta, app_->next_modified_);
                }
              }
            });
      } else {
        ForEachSimple(
            app_->curr_modified_, inner_vertices, [this, &compr_stage](int tid, vertex_t u) {
              if(compr_stage && is_spnode_in[u]){
                supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];
                auto& oes = spnode.bound_delta;

                if (app_->CombineValueDelta(value, delta)) {
                  app_->ComputeByIndexDelta(u, last_value, delta, oes, app_->next_modified_);
                  // app_->CombineValueDelta(spnode.data, delta);
                }
              }
              else if(compr_stage == false || is_nornode[u]){
                auto& value = app_->values_[u];
                auto last_value = value;
                // We don't cleanup delta with identity element, since we expect
                // the algorithm is monotonic
                auto& delta = app_->deltas_[u];

                if (app_->CombineValueDelta(value, delta)) {
                  app_->Compute(u, last_value, delta, app_->next_modified_);
                }
              }
            });
      }

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

      if (app_->next_modified_.Count() > 0) {
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
          timer_next("correct deviation");
          compr_stage = false;
          // supernode send by inner_delta
          // for(vid_t j = 0; j < cpr_->supernodes_num; j++){
          granular_for(j, 0, cpr_->supernodes_num, (cpr_->supernodes_num > 1024), {
            supernode_t &spnode = cpr_->supernodes[j];
            auto u = spnode.id;
            auto& value = app_->values_[u];
            // We don't cleanup delta with identity element, since we expect
            // the algorithm is monotonic
            auto& delta = app_->deltas_[u];
            auto& oes = spnode.inner_delta;

            if (value != app_->GetIdentityElement()){ // right
            // if(value != app_->GetIdentityElement() && (cpr_->if_touch[j] || spnode.data != value)){ // error
              app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
            }
            app_->CombineValueDelta(spnode.data, delta);
          })
          // inner nodes receive
          ForEach(inner_vertices, [this](int tid, vertex_t u) {
              if(!is_spnode_in[u] && !is_nornode[u]){
                auto& value = app_->values_[u];
                auto& delta = app_->deltas_[u];
                app_->CombineValueDelta(value, delta);
              }
            }
          );
          
          LOG(INFO) << "cps_graph_step=" << step << " correct deviation in supernode";
          // LOG(INFO) << "f_send_num=" << app_->f_send_num  << " f_send_delta_num=" << app_->f_send_delta_num << " sum=" << (app_->f_send_num + app_->f_send_delta_num);
          app_->next_modified_.Swap(app_->curr_modified_);
          continue;
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "iter step: " << step;
            LOG(INFO) << "Batch time: " << exec_time;
            // LOG(INFO) << "f_send_num_1=" << app_->f_send_num  << " f_send_delta_num_1=" << app_->f_send_delta_num << " sum_1=" << (app_->f_send_num + app_->f_send_delta_num);
            // LOG(INFO) << "g_num_1=" << app_->g_num;
            // app_->f_send_num = 0;
            // app_->f_send_delta_num = 0;
            // app_->g_num = 0;
          }
          exec_time = 0;
          step = 0;

          if (!FLAGS_efile_update.empty()) {
            compr_stage = FLAGS_compress; // use supernode
            timer_next("reloadGraph");
            deltaCompute();  // reload graph
            timer_next("inc algorithm");
            if(compr_stage){
              auto inner_vertices = fragment_->InnerVertices();
              is_spnode_in.Init(inner_vertices, false);
              is_nornode.Init(inner_vertices, false);
              ForEach(inner_vertices, [this](int tid, vertex_t u) {
                  if(cpr_->Fc[u] == cpr_->FC_default_value){
                    is_nornode[u] = true;
                  }
                  else if(cpr_->Fc[u] >= 0){
                    is_spnode_in[u] = true;
                  }
                }
              );
              /* The supernode entry vertex sends one round unconditionally. */
              ForEach(inner_vertices, [this](int tid, vertex_t u) {
                if (is_spnode_in[u] && app_->deltas_[u].value != app_->GetIdentityElement()) {
                    supernode_t &spnode = cpr_->supernodes[cpr_->Fc_map[u]];
                    auto& value = app_->values_[u];
                    // We don't cleanup delta with identity element, since we expect
                    // the algorithm is monotonic
                    auto& delta = app_->deltas_[u];
                    auto& oes = spnode.bound_delta;
                    app_->ComputeByIndexDelta(u, value, delta, oes, app_->next_modified_);
                  }
                }
              );
            }
            // LOG(INFO) << "deltaCompute: f_send_num_inc=" << app_->f_send_num  << " f_send_delta_num_inc=" << app_->f_send_delta_num << " sum_inc=" << (app_->f_send_num + app_->f_send_delta_num);
            // LOG(INFO) << "g_num_1=" << app_->g_num;
            // app_->f_send_num = 0;
            // app_->f_send_delta_num = 0;
            // app_->g_num = 0;
          } else {
            LOG(ERROR) << "Missing efile_update or efile_updated";
            break;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "iter step: " << step;
            LOG(INFO) << "Inc time: " << exec_time << " sec";
            // LOG(INFO) << "f_send_num_2=" << app_->f_send_num  << " f_send_delta_num_2=" << app_->f_send_delta_num << " sum_2=" << (app_->f_send_num + app_->f_send_delta_num);
            // LOG(INFO) << "g_num_2=" << app_->g_num;
            // app_->f_send_num = 0;
            // app_->f_send_delta_num = 0;
            // app_->g_num = 0;
          }
          break;
        }
      }

      app_->next_modified_.Swap(app_->curr_modified_);
    }
    MPI_Barrier(comm_spec_.comm());
  }

  void Output(std::ostream& os) {
    auto inner_vertices = fragment_->InnerVertices();
    auto& values = app_->values_;

    for (auto v : inner_vertices) {
      os << fragment_->GetId(v) << " " << values[v] << std::endl;
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
  VertexArray<bool, vid_t> is_spnode_in;
  VertexArray<bool, vid_t> is_nornode;
};

}  // namespace grape

#endif