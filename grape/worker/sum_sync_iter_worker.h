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
#include <cuda_runtime.h>
#include "freshman.h"
#include "my_worker.cuh"
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
      timer_next("init app_");
      app_->reInit(cpr_->all_node_num, *graph_); // for mirror node
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
   * 用于图变换前后值的修正，注意仅仅对受影响的点进行处理
   * type: -1表示回收(在旧图上)， 1表示重发(在新图上)
   * actives_nodes: 需要处理的活跃点
   */
  void AmendValue_active(int type, std::unordered_set<vertex_t>& actives_nodes) {
    LOG(INFO) << "test------------";
    auto inner_vertices = graph_->InnerVertices();
    auto outer_vertices = graph_->OuterVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;

    // 校正消息可以正负累积之后一次向外通信
    {
      auto it = actives_nodes.begin();
      // granular_for(j, 0, actives_nodes.size(), (actives_nodes.size() > 1024), {
      for (auto u : actives_nodes) {
        // auto& u = *(it + j);
        auto& value = values[u];
        auto delta = type * value;  // 发送回收值/发送补发值
        auto oes = graph_->GetOutgoingAdjList(u);

        app_->g_function(*graph_, u, value, delta, oes);
      }
    }
    LOG(INFO) << "test------------";
  }

  /**
   * 用于重新加载图，并完成图变换前后值的校正
   *
   */
  void reloadGraph() {
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

    //-----------------------------get active nodes-----------------------------
    double resivition_time_0 = GetCurrentTime();
    #if defined(DISTRIBUTED)
      LOG(INFO) << "Distributed vision...";
      std::unordered_set<vid_t> local_gid_set;
      for (auto v : fragment_->Vertices()) {
        local_gid_set.insert(fragment_->Vertex2Gid(v));
      }
    #else
      LOG(INFO) << "Single vision...";
    #endif
    size_t del_pair_num = deleted_edges.size();
    size_t add_pair_num = added_edges.size();
    std::unordered_set<vertex_t> actives_nodes; // vid set of origin node in edge
    VertexArray<bool, vid_t> is_update;
    is_update.Init(graph_->InnerVertices()); // is inner vertieces
    actives_nodes.reserve(del_pair_num+add_pair_num);
    LOG(INFO) << " test--actives_nodes.size=" << actives_nodes.size();
    for(vid_t i = 0; i < del_pair_num; i++) {
      auto pair = deleted_edges[i];
      vid_t u_gid = pair.first;

      #if defined(DISTRIBUTED)
        if (local_gid_set.find(u_gid) != local_gid_set.end()) {
          vertex_t u;
          CHECK(graph_->Gid2Vertex(u_gid, u));
          actives_nodes.insert(u);
          is_update[u] = true;
        }
      #else
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        actives_nodes.insert(u);
        is_update[u] = true;
      #endif
    }
    LOG(INFO) << " test--actives_nodes.size=" << actives_nodes.size();
    for(vid_t i = 0; i < add_pair_num; i++) {
      auto pair = added_edges[i];
      vid_t u_gid = pair.first;

      #if defined(DISTRIBUTED)
        if (local_gid_set.find(u_gid) != local_gid_set.end()) {
          vertex_t u;
          CHECK(graph_->Gid2Vertex(u_gid, u));
          actives_nodes.insert(u);
          is_update[u] = true;
        }
      #else
        vertex_t u;
        CHECK(graph_->Gid2Vertex(u_gid, u));
        actives_nodes.insert(u);
        is_update[u] = true;
      #endif
    }
    // recycled value on the old graph
    // if(FLAGS_compress){
    //   AmendValue_active(-1, actives_nodes);
    // } else {
    //   resivition_time_0 = GetCurrentTime();
      AmendValue(-1);
    // }
    LOG(INFO) << "#resivition_time_0: " << (GetCurrentTime() - resivition_time_0);

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

    LOG(INFO) << "test------------";
    // graph_ = inc_fragment_builder.Build();
    app_->rebuild_graph(*graph_);
    const std::shared_ptr<fragment_t>& new_graph = inc_fragment_builder.Build();
    // app_->iterate_begin(*graph_);
    app_->iterate_begin(*new_graph);
    LOG(INFO) << "test------------";

    print_active_edge("#AmendValue-1");
    if(FLAGS_compress){
      cpr_->inc_run(deleted_edges, added_edges, new_graph, is_update);
      print_active_edge("#inc_run_cmpIndex");
    }
    graph_ = new_graph;

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "New graph loaded" << " new_graph_edgeNum=" << (graph_->GetEdgeNum()/2);
    }
    app_->Init(comm_spec_, *graph_, false);
    if (FLAGS_compress) {
      app_->reInit(cpr_->all_node_num, *graph_); // for mirror node
    }
    {
      // Copy values to new graph
      for (auto v : iv) {
        app_->values_[v] = values[v];
        app_->deltas_[v] = deltas[v];
      }
    }

    // reissue value on the new graph
    double resivition_time_1 = GetCurrentTime();
    // if(FLAGS_compress){
    //   AmendValue_active(1, actives_nodes);
    // } else {
      AmendValue(1);
    // }
    LOG(INFO) << "#resivition_time_1: " << (GetCurrentTime() - resivition_time_1);
    print_active_edge("#AmendValue+1");
  }


  void first_step(bool is_inc) {
    auto inner_vertices = graph_->InnerVertices();
    vid_t inner_node_num = inner_vertices.end().GetValue() 
                            - inner_vertices.begin().GetValue();
    auto new_node_range = VertexRange<vid_t>(0, cpr_->all_node_num);
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    auto& is_e_ = cpr_->is_e_;
    auto& is_e_offset_ = cpr_->is_e_offset_;
    auto& ib_e_ = cpr_->ib_e_;
    auto& ib_e_offset_ = cpr_->ib_e_offset_;
    auto& sync_e_ = cpr_->sync_e_;
    auto& sync_e_offset_ = cpr_->sync_e_offset_;

    double node_type_time = GetCurrentTime();
    bound_node_values.clear();
    bound_node_values.Init(new_node_range, app_->default_v());
    spnode_datas.clear();
    spnode_datas.Init(new_node_range, app_->default_v());
    
    cpr_->get_nodetype_mirror(inner_node_num, node_type);

    all_nodes.clear();
    all_nodes.resize(7);
    for(vid_t i = inner_vertices.begin().GetValue(); 
      i < inner_vertices.end().GetValue(); i++) {
        all_nodes[node_type[i]].emplace_back(vertex_t(i));
    }
    LOG(INFO) << "node_type_time=" << (GetCurrentTime()-node_type_time); //0.313418

    
    for (int i = 0; i < all_nodes.size(); i++) {
      LOG(INFO) << "  type" << i << "=" << all_nodes[i].size()
                << " rate=" << (1.0 * all_nodes[i].size() / inner_node_num);
    }

    // cpr_->sketch2csr_divide(node_type);
    cpr_->sketch2csr_mirror(node_type);

    /* precompute supernode */
    timer_next("pre compute");
    double pre_compute = GetCurrentTime();
    cpr_->precompute_spnode_one(this->graph_, is_inc);
    parallel_for(vid_t j = 0; j < cpr_->cluster_ids.size(); j++){
      /* send to out nodes by bound edges: 不在原图上发,应该是sketch上 */
      std::vector<vertex_t> &node_set = cpr_->supernode_ids[j];
      for(auto u : node_set) {
        if (cpr_->supernode_out_bound[u.GetValue()]) {
          value_t value = values[u];
          vid_t i = u.GetValue();
          {
            /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
            auto oes = graph_->GetOutgoingAdjList(u);
            adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                        ib_e_offset_[i+1]);
            app_->g_function(*graph_, u, value, value, oes, adj);  // out degree neq now adjlist.size
            /* in-master */
            if (value != app_->default_v()) {
              adj_list_t sync_adj = adj_list_t(sync_e_offset_[i], 
                                                sync_e_offset_[i+1]);
              for (auto e : sync_adj) {
                vertex_t v = e.neighbor;
                // sync to mirror v
                app_->accumulate_atomic(deltas[v], value);
                // active mirror v
                value_t& old_delta = deltas[v];
                auto delta = atomic_exch(old_delta, app_->default_v());
                auto& value = values[v];
                adj_list_index_t adj = adj_list_index_t(is_e_offset_[v.GetValue()], 
                                                        is_e_offset_[v.GetValue()+1]);
                app_->g_index_function(*graph_, v, value, delta, adj, 
                                        bound_node_values);
                app_->accumulate_atomic(spnode_datas[v], delta);
              }
            }
          }
        }
      }
      /* out-mirror to master */
      for (auto u : cpr_->cluster_out_mirror_ids[j]) {
        vertex_t v = cpr_->mirrorid2vid[u];
        auto delta = atomic_exch(deltas[u], app_->default_v());
        this->app_->accumulate_atomic(deltas[v], delta);
        // LOG(INFO) << "oid=" << cpr_->v2Oid(v) << " value=" << deltas[u];
      }
    }
    cpr_->precompute_spnode_two();
    pre_compute = GetCurrentTime() - pre_compute;
    LOG(INFO) << "#pre_compute_" << int(is_inc) << ": " << pre_compute;
    print_active_edge("#pre_compute");
  }

  void Query() {
    MPI_Barrier(comm_spec_.comm());
    // app_->Init(comm_spec_, *graph_, false);

    #ifdef COUNT_ACTIVE_EDGE_NUM
      LOG(INFO) << "============== open count active edge num ================";
    #else
      LOG(INFO) << "=============== close count active edge num===============";
    #endif

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

    auto& is_e_ = cpr_->is_e_;
    auto& is_e_offset_ = cpr_->is_e_offset_;//SumInc
    auto& ib_e_ = cpr_->ib_e_;
    auto& ib_e_offset_ = cpr_->ib_e_offset_;
    auto& sync_e_ = cpr_->sync_e_;
    auto& sync_e_offset_ = cpr_->sync_e_offset_;

    int step = 1;
    bool batch_stage = true;
    short int convergence_id = 0;
    bool compr_stage = FLAGS_compress; // true: supernode send


    last_values.Init(inner_vertices);
    value_t init_value_sum = 0; 
    value_t init_delta_sum = 0;
    // for (auto v : inner_vertices) {
    //v is a vertex from begin to end , type is vid(int)
    parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
      vertex_t v(i);//v is a class var,init with i
      last_values[v] = values[v];
    }

  // {//values 2 last_values in GPU
  //   //use cuda instead of parallel_for.
  //   //1.define var
  //   vid_t num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
  //   vid_t *v_d,*v_h = (vid_t*)malloc(sizeof(vid_t)*num);
    
  //   value_t *last_values_d;
  //   value_t *values_d;
  //   // VertexArray<value_t, vid_t>  last_values_d;
  //   // VertexArray<value_t, vid_t>  values_d;
    
  //   //2.  mem allocate
  //   memset(v_h,0,num);
  //   for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++){
  //     v_h[i - inner_vertices.begin().GetValue()] = i;
  //   }
  //   cudaSetDevice(0);

  //   cudaMalloc(&v_d, sizeof(vid_t) * num);
  //   cudaMalloc(&last_values_d, sizeof(value_t) * num);
  //   cudaMalloc(&values_d, sizeof(value_t) * num);
  //   // check();
    
  //   //using a buffer for data transmition 
  //   last_values.fake2buffer();
  //   values.fake2buffer();

  //   //3. mem copy from cpu to gpu and run kernel func
  //   cudaMemcpy(v_d, v_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
  //   cudaMemcpy(last_values_d, last_values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
  //   cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
  //   tjn::value2last(last_values_d, values_d, v_d, num);

  //   //4. mem copy from gpu to cpu
  //   cudaMemcpy(last_values.data_buffer, last_values_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
  //   // check();
  //   last_values.buffer2fake();//buffer 

  //   //5. mem free
  //   cudaFree(last_values_d);
  //   cudaFree(v_d);
  //   cudaFree(values_d);
  //   free(v_h);
  // }

    


    if(compr_stage){
      first_step(false); // batch
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

    // print_result();


    #ifdef DEBUG
      double time_sum_0 = 0;
      double time_sum_1 = 0;
      double time_sum_2 = 0;
      double time_sum_3 = 0;
      double time_sum_4 = 0;
    #endif

    double time = 0;

    auto oeoffset = graph_->getOeoffset();//用于Ingress

    vid_t num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();

    vid_t *size_oe_d, *size_oe_h = (vid_t *)malloc(sizeof(vid_t) * num);//Ingress,用于记录每一个顶点的邻居数
    vid_t *size_ib_d, *size_ib_h = (vid_t *)malloc(sizeof(vid_t) * num);//SumInc,node type:SingleNode
    vid_t *size_is_d, *size_is_h = (vid_t *)malloc(sizeof(vid_t) * num);//SumInc,node type:OnlyInNode
    vid_t *size_sync_d, *size_sync_h = (vid_t *)malloc(sizeof(vid_t) * num);//SumInc,node type:OutMaster
    
    vid_t *cur_oeoff_d, *cur_oeoff_h = (vid_t *)malloc(sizeof(vid_t) * num);//Ingress,用于记录每一个顶点在邻居大链表中开始的偏移量
    vid_t *cur_iboff_d, *cur_iboff_h = (vid_t *)malloc(sizeof(vid_t) * num);
    vid_t *cur_isoff_d, *cur_isoff_h = (vid_t *)malloc(sizeof(vid_t) * num);
    vid_t *cur_syncoff_d, *cur_syncoff_h = (vid_t *)malloc(sizeof(vid_t) * num);
    
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
      for(int i=0;i<num;i++){
        cur_isoff_h[i] = is_offsize;
        is_offsize += is_e_offset_[i+1] - is_e_offset_[i];
        size_is_h[i] = is_e_offset_[i+1] - is_e_offset_[i];
      }
    }

    unsigned int sync_offsize = 0;
    if(compr_stage){
      for(int i=0;i<num;i++){
        cur_syncoff_h[i] = sync_offsize;
        sync_offsize += sync_e_offset_[i+1] - sync_e_offset_[i];
        size_sync_h[i] = sync_e_offset_[i+1] - sync_e_offset_[i];
      }
    }
    

    value_t *deltas_d;
    value_t *values_d;
    value_t *bound_node_values_d;
    value_t *spnode_datas_d;

    vid_t *oeoffset_d, *oeoffset_h = (vid_t *)malloc(sizeof(vid_t) * oe_offsize);//Ingress,记录每个顶点的邻居，形成一条链表
    vid_t *iboffset_d, *iboffset_h = (vid_t *)malloc(sizeof(vid_t) * ib_offsize);//SumInc
    vid_t *isoffset_d, *isoffset_h = (vid_t *)malloc(sizeof(vid_t) * is_offsize);//SumInc
    vid_t *syncoffset_d, *syncoffset_h = (vid_t *)malloc(sizeof(vid_t) * sync_offsize);//SumInc
    value_t *is_edata_d, *is_edata_h = (value_t *)malloc(sizeof(value_t) * is_offsize);//边数据
    char *node_type_d, *node_type_h = (char *)malloc(sizeof(char) * num);//SumInc,记录每个顶点的类型

    
    cudaSetDevice(0);

    cudaMalloc(&deltas_d, sizeof(value_t) * num);
    cudaMalloc(&values_d, sizeof(value_t) * num);
    cudaMalloc(&bound_node_values_d, sizeof(value_t) * num);
    cudaMalloc(&spnode_datas_d, sizeof(value_t) * num);
    // check();
    cudaMalloc(&oeoffset_d, sizeof(vid_t) * oe_offsize);
    cudaMalloc(&iboffset_d, sizeof(vid_t) * ib_offsize);
    cudaMalloc(&isoffset_d, sizeof(vid_t) * is_offsize);
    cudaMalloc(&syncoffset_d, sizeof(vid_t) * sync_offsize);
    cudaMalloc(&is_edata_d, sizeof(value_t) * is_offsize);
    // check();
    cudaMalloc(&cur_oeoff_d, sizeof(vid_t) * num);
    cudaMalloc(&cur_iboff_d, sizeof(vid_t) * num);
    cudaMalloc(&cur_isoff_d, sizeof(vid_t) * num);
    cudaMalloc(&cur_syncoff_d, sizeof(vid_t) * num);
    // check();

    cudaMalloc(&size_oe_d, sizeof(vid_t) * num);
    cudaMalloc(&size_ib_d, sizeof(vid_t) * num);
    cudaMalloc(&size_is_d, sizeof(vid_t) * num);
    cudaMalloc(&size_sync_d, sizeof(vid_t) * num);
    // check();
    cudaMalloc(&node_type_d, sizeof(char) * num);
    // bool free_need = false;
    // if(compr_stage){//启用压缩，则分配内存
    //   free_need = true;
    //   cudaMalloc(&node_type_d, sizeof(char) * num);//压缩， 顶点类型
    //   cudaMalloc(&bound_node_values_d, sizeof(value_t) * bound_node_values.size());//给bound_node分配内存
    //   cudaMalloc(&spnode_datas_d, sizeof(value_t) * spnode_datas.size());//给spnode分配内存
    // }
    int oe_curIndex = 0, ib_curIndex = 0, is_curIndex = 0, sync_curIndex = 0;
    for(int i = 0; i < num; i++){
      if(compr_stage)//启用压缩时node_type才有效
        node_type_h[i] = node_type[i];
      for(int j = 0;j < size_oe_h[i]; j++){
        oeoffset_h[oe_curIndex++] = oeoffset[i][j].neighbor.GetValue();
      }
      for(int j = 0;j < size_ib_h[i]; j++){
        iboffset_h[ib_curIndex++] = ib_e_offset_[i][j].neighbor.GetValue();
        
      }
      for(int j = 0;j < size_is_h[i];j++){
        is_edata_h[is_curIndex] = is_e_offset_[i][j].data;
        isoffset_h[is_curIndex++] = is_e_offset_[i][j].neighbor.GetValue();
        // std::cout<<"value is :"<<is_e_offset_[i][j].data;
      }
      for(int j = 0;j < size_sync_h[i];j++){
        syncoffset_h[sync_curIndex++] = sync_e_offset_[i][j].neighbor.GetValue();
      }
    }

    deltas.fake2buffer();
    values.fake2buffer();
    bound_node_values.fake2buffer();
    spnode_datas.fake2buffer();
    //将要用到的数据进行传输
    cudaMemcpy(oeoffset_d, oeoffset_h, sizeof(vid_t) * oe_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(iboffset_d, iboffset_h, sizeof(vid_t) * ib_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(isoffset_d, isoffset_h, sizeof(vid_t) * is_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(syncoffset_d, syncoffset_h, sizeof(vid_t) * sync_offsize, cudaMemcpyHostToDevice);
    cudaMemcpy(is_edata_d, is_edata_h, sizeof(value_t) * is_offsize, cudaMemcpyHostToDevice);
    // check();
    cudaMemcpy(cur_oeoff_d, cur_oeoff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_iboff_d, cur_iboff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_isoff_d, cur_isoff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(cur_syncoff_d, cur_syncoff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    // check();
    cudaMemcpy(deltas_d, deltas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(bound_node_values_d, bound_node_values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(spnode_datas_d, spnode_datas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
    // check();
    cudaMemcpy(size_oe_d, size_oe_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(size_ib_d, size_ib_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(size_is_d, size_is_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    cudaMemcpy(size_sync_d, size_sync_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
    // check();
    cudaMemcpy(node_type_d, node_type_h, sizeof(char) * num, cudaMemcpyHostToDevice);
    check();
    tjn::init(spnode_datas_d, bound_node_values_d, deltas_d, values_d, 
              oeoffset_d, iboffset_d, isoffset_d, syncoffset_d, 
              size_oe_d, size_ib_d, size_is_d, size_sync_d, 
              inner_vertices.begin().GetValue(), inner_vertices.end().GetValue(), 
              cur_oeoff_d, cur_iboff_d, cur_isoff_d, cur_syncoff_d,
              node_type_d, is_edata_d); 
              check();
    bool gpu_start = true;;
    while (true) {
      ++step;

      // LOG(INFO) << "step=" << step << " f_send_value_num=" << app_->f_send_value_num << " f_send_delta_num=" << app_->f_send_delta_num;
      // app_->f_send_value_num = 0;
      // app_->f_send_delta_num = 0;
      // LOG(INFO) << "step=" << step << " node_update_num=" << app_->node_update_num << " touch_nodes=" << app_->touch_nodes.size();
      // app_->node_update_num = 0;

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

      {//recieve message
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

      {//computation
        #ifdef DEBUG
          auto begin = GetCurrentTime();
        #endif
        // long long last_f = (app_->f_send_num + app_->f_send_value_num + app_->f_send_delta_num);
        if (FLAGS_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
          if(compr_stage == false){//no compression used,ingress

            // value_t priority;
            // if(FLAGS_portion < 1){
            //   priority = Scheduled(1000);
            // }

            #ifdef DEBUG
              send_time_0 -= GetCurrentTime();
            #endif

            // for(int i=0;i<32;i++){
            //   vertex_t u(i);
            //   printf("deltas_d0[%d] is %f\n", i, deltas[u]);
            // }
            // #pragma cilk gransize = 1;
            // printf("num is %d",inner_vertices.end().GetValue());
            if(!gpu_start){
              parallel_for(vid_t i = inner_vertices.begin().GetValue();
                        i < inner_vertices.end().GetValue(); i++) {
                vertex_t u(i);
                value_t& old_delta = deltas[u];
                // printf("deltas_d[%d] is %f\n", i, deltas[u]);
                // if(FLAGS_portion == 1 || old_delta >= priority){
                if (isChange(old_delta)) {
                  auto& value = values[u];
                  // printf("deltas_d[%d] is %f\n", i, deltas[u]);
                  
                  auto delta = atomic_exch(deltas[u], app_->default_v());//return deltas[u] to delta, and deltas[u] = 0, 0 = default_v()
                  auto oes = graph_->GetOutgoingAdjList(u);
                  
                  app_->g_function(*graph_, u, value, delta, oes);
                  app_->accumulate_atomic(value, delta);
                  
                  // printf("deltas_d[%d] is %f\n", i, deltas[u]);
                  #ifdef DEBUG
                    //n_edge += oes.Size();
                  #endif
                }
                // }
              }
              check_result();
            }
            
            // double time1 = GetCurrentTime();
            // auto oeoffset = graph_->getOeoffset();
            // vid_t num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();

            // vid_t *size_oe_d, *size_oe_h = (vid_t *)malloc(sizeof(vid_t) * num);
            // vid_t *cur_oeoff_d, *cur_oeoff_h = (vid_t *)malloc(sizeof(vid_t)  * num);
            // unsigned int oe_offsize = 0;
            // for(int i = 0;i < num; i++){
            //   cur_oeoff_h[i] = oe_offsize;
            //   oe_offsize += oeoffset[i+1] - oeoffset[i];
            //   size_oe_h[i] = oeoffset[i+1] - oeoffset[i];
            // }
            // value_t *deltas_d;
            // value_t *values_d;
            // vid_t *oeoffset_d, *oeoffset_h = (vid_t *)malloc(sizeof(vid_t)  * oe_offsize);
            
            // cudaSetDevice(0);
            // cudaMalloc(&deltas_d, sizeof(value_t) * num);
            // cudaMalloc(&values_d, sizeof(value_t) * num);
            // cudaMalloc(&oeoffset_d, sizeof(vid_t) * oe_offsize);
            // cudaMalloc(&cur_oeoff_d, sizeof(vid_t) * num);
            // cudaMalloc(&size_oe_d, sizeof(vid_t) * num);

            // // check();

            // int curIndex = 0;
            // for(int i = 0; i < num; i++){
            //   for(int j = 0;j < size_oe_h[i]; j++){
            //       oeoffset_h[curIndex++] = oeoffset[i][j].neighbor.GetValue();
            //   }

            // }

            // deltas.fake2buffer();
            // values.fake2buffer();

            // cudaMemcpy(oeoffset_d, oeoffset_h, sizeof(vid_t) * oe_offsize, cudaMemcpyHostToDevice);
            // cudaMemcpy(cur_oeoff_d, cur_oeoff_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
            // cudaMemcpy(deltas_d, deltas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
            // cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
            // cudaMemcpy(size_oe_d, size_oe_h, sizeof(vid_t) * num, cudaMemcpyHostToDevice);
            // // check();
            // double time2 = GetCurrentTime();
            // time += time2 - time1;
            // tjn::g_function_pr(deltas_d, values_d, oeoffset_d, size_oe_d, inner_vertices.begin().GetValue(), inner_vertices.end().GetValue(), cur_oeoff_d);
            // cudaDeviceSynchronize();
            // check();
            // double time3 = GetCurrentTime();
            // cudaMemcpy(deltas.data_buffer, deltas_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
            // cudaMemcpy(values.data_buffer, values_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
            
            // deltas.buffer2fake();
            // values.buffer2fake();
            // cudaFree(deltas_d);
            // cudaFree(values_d);
            // cudaFree(oeoffset_d);
            // cudaFree(size_oe_d);
            // cudaFree(cur_oeoff_d);

            // free(size_oe_h);
            // free(oeoffset_h);
            // free(cur_oeoff_h);
            // double time4 = GetCurrentTime();
            // time += time4 - time3;
            // printf("transtime is %f\n",time);
            
            if(gpu_start){
              // 一次传输
              values.fake2buffer();
              deltas.fake2buffer();
              spnode_datas.fake2buffer();
              bound_node_values.fake2buffer();
              cudaMemcpy(deltas_d, deltas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
              cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
              cudaMemcpy(spnode_datas_d, spnode_datas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
              cudaMemcpy(bound_node_values_d, bound_node_values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
              tjn::g_function_pr(inner_vertices.begin().GetValue(), inner_vertices.end().GetValue());
              cudaDeviceSynchronize();
              check();
              double time1 = GetCurrentTime();
              cudaMemcpy(deltas.data_buffer, deltas_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
              
              cudaMemcpy(values.data_buffer, values_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
              cudaMemcpy(bound_node_values.data_buffer, bound_node_values_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
              cudaMemcpy(spnode_datas.data_buffer, spnode_datas_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
              
              deltas.buffer2fake();
              values.buffer2fake();
              bound_node_values.buffer2fake();
              spnode_datas.buffer2fake();
              double time2 = GetCurrentTime();
              time += time2 - time1;
              printf("time is :%f",time);
              check_result();
            }
            

            
            #ifdef DEBUG
              send_time_0 += GetCurrentTime();
              time_sum_0 += send_time_0;
              //LOG(INFO) << "time0/edges=" << (send_time_0/(n_edge-last_n_edge)) << " edge=" << (n_edge-last_n_edge) << " node=" << node_0_size;
              std::cout << "N edge: " << n_edge << std::endl;
            #endif
          }

          if(compr_stage){//compression used,suminc
            // value_t priority;
            // if(FLAGS_portion < 1){
            //   priority = Scheduled(1000);
            // }
            if(1){
              if(gpu_start){
                check_result();
                values.fake2buffer();
                deltas.fake2buffer();
                spnode_datas.fake2buffer();
                bound_node_values.fake2buffer();
                cudaMemcpy(deltas_d, deltas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
                cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
                cudaMemcpy(spnode_datas_d, spnode_datas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
                cudaMemcpy(bound_node_values_d, bound_node_values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
                tjn::g_function_compr(inner_vertices.begin().GetValue(), inner_vertices.end().GetValue());
                cudaDeviceSynchronize();
                check();
                cudaMemcpy(deltas.data_buffer, deltas_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
                cudaMemcpy(values.data_buffer, values_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
                cudaMemcpy(spnode_datas.data_buffer, spnode_datas_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
                cudaMemcpy(bound_node_values.data_buffer, bound_node_values_d, sizeof(value_t) * num, cudaMemcpyDeviceToHost);
                check();

                deltas.buffer2fake();
                values.buffer2fake();
                spnode_datas.buffer2fake();
                bound_node_values.buffer2fake();
                check_result();
              }
              
              if(!gpu_start){
                check_result();
                parallel_for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {

                  vertex_t u(i);
                  switch (node_type[i]){
                  
                  case NodeType::SingleNode:
                    /* 1. out node */
                    {
                      value_t& old_delta = deltas[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        adj_list_t oes = adj_list_t(ib_e_offset_[i], 
                                                    ib_e_offset_[i+1]); 
                        app_->g_function(*graph_, u, value, delta, oes);
                        app_->accumulate_atomic(value, delta);
                      }
                    }
                    break;
                  case NodeType::OnlyInNode:
                    /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                    {
                      value_t& old_delta = deltas[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], 
                                                                is_e_offset_[i+1]);
                        app_->g_index_function(*graph_, u, value, delta, adj, 
                                                                  bound_node_values);
                        app_->accumulate_atomic(spnode_datas[u], delta);
                      }
                    }
                    break;
                  case NodeType::OnlyOutNode:
                    /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                    {
                      value_t& old_delta = bound_node_values[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        auto oes = graph_->GetOutgoingAdjList(u);
                        adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                    ib_e_offset_[i+1]);
                        app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                        app_->accumulate_atomic(value, delta);
                      }
                    }
                    break;
                  case NodeType::BothOutInNode:
                    /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                    {
                      value_t& old_delta = deltas[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], 
                                                                  is_e_offset_[i+1]);
                        app_->g_index_function(*graph_, u, value, delta, adj, 
                                                                  bound_node_values);
                        app_->accumulate_atomic(spnode_datas[u], delta);
                      }
                    }
                    /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                    {
                      value_t& old_delta = bound_node_values[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        auto oes = graph_->GetOutgoingAdjList(u);
                        adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                    ib_e_offset_[i+1]);
                        app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                        app_->accumulate_atomic(value, delta);
                      }
                    }
                    break;
                  case NodeType::OutMaster:
                    {
                      /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                      value_t& old_delta = bound_node_values[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        auto oes = graph_->GetOutgoingAdjList(u);
                        adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                    ib_e_offset_[i+1]);
                        app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                        app_->accumulate_atomic(value, delta);
                        /* in-master */
                        if (delta != app_->default_v()) {
                          adj_list_t sync_adj = adj_list_t(sync_e_offset_[i], 
                                                          sync_e_offset_[i+1]);
                          for (auto e : sync_adj) {
                            vertex_t v = e.neighbor;
                            // sync to mirror v
                            app_->accumulate_atomic(deltas[v], delta);
                            // active mirror v
                            value_t& old_delta = deltas[v];
                            auto delta = atomic_exch(old_delta, app_->default_v());
                            auto& value = values[v];
                            adj_list_index_t adj = adj_list_index_t(
                                                    is_e_offset_[v.GetValue()], 
                                                    is_e_offset_[v.GetValue()+1]);
                            app_->g_index_function(*graph_, v, value, delta, adj, 
                                                    bound_node_values);
                            app_->accumulate_atomic(spnode_datas[v], delta);
                          }
                        }
                      }
                    }
                    break;
                  case NodeType::BothOutInMaster:
                    /* 2. source node: source send message to inner_bound_node by inner_bound_index */
                    {
                      value_t& old_delta = deltas[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        adj_list_index_t adj = adj_list_index_t(is_e_offset_[i], 
                                                                is_e_offset_[i+1]);
                        app_->g_index_function(*graph_, u, value, delta, adj, 
                                                                  bound_node_values);
                        app_->accumulate_atomic(spnode_datas[u], delta);
                      }
                    }
                    {
                      /* 3. bound node: some node in is_spnode_in and supernode_out_bound at the same time. */
                      value_t& old_delta = bound_node_values[u];
                      if (isChange(old_delta)) {
                        auto delta = atomic_exch(old_delta, app_->default_v());
                        auto& value = values[u];
                        auto oes = graph_->GetOutgoingAdjList(u);
                        adj_list_t adj = adj_list_t(ib_e_offset_[i], 
                                                    ib_e_offset_[i+1]);
                        app_->g_function(*graph_, u, value, delta, oes, adj);  // out degree neq now adjlist.size
                        app_->accumulate_atomic(value, delta);
                        /* in-master */
                        if (delta != app_->default_v()) {
                          adj_list_t sync_adj = adj_list_t(sync_e_offset_[i], 
                                                          sync_e_offset_[i+1]);
                          for (auto e : sync_adj) {
                            vertex_t v = e.neighbor;
                            // sync to mirror v
                            app_->accumulate_atomic(deltas[v], delta);
                            // active mirror v
                            value_t& old_delta = deltas[v];
                            auto delta = atomic_exch(old_delta, app_->default_v());
                            auto& value = values[v];
                            adj_list_index_t adj = adj_list_index_t(
                                                    is_e_offset_[v.GetValue()], 
                                                    is_e_offset_[v.GetValue()+1]);
                            app_->g_index_function(*graph_, v, value, delta, adj, 
                                                    bound_node_values);
                            app_->accumulate_atomic(spnode_datas[v], delta);
                          }
                        }
                      }
                    }
                    break;
                  }
                }
                check_result();
              }
              
            
            /* out-mirror sync to master */
            vid_t size = cpr_->all_out_mirror.size();
            parallel_for (vid_t i = 0; i < size; i++) {
              vertex_t u =cpr_->all_out_mirror[i];
              value_t& old_delta = bound_node_values[u];
              if (isChange(old_delta)) {
                vertex_t v = cpr_->mirrorid2vid[u];
                auto delta = atomic_exch(bound_node_values[u], app_->default_v()); // send spnode_datas
                this->app_->accumulate_atomic(deltas[v], delta);
                // LOG(INFO) << "out-mirror -> master:" << cpr_->v2Oid(u) << "->"
                //           << cpr_->v2Oid(v) << " delta=" << delta;
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
          printf("v is %d",v.GetValue());
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

      if (termCheck(last_values, values, compr_stage) || step > FLAGS_pr_mr) {//达到阈值或达到迭代次数上限
        app_->touch_nodes.clear();
        if(compr_stage){
          LOG(INFO) << " start correct deviation...";
          // if (convergence_id < 1) {
          //   convergence_id++;
          //   continue;
          // }
          print_active_edge("#globalCompt");
          timer_next("correct deviation");
          // print_result();
          corr_time -= GetCurrentTime();
          // supernode send by inner_delta and inner_value
          // parallel_for(vid_t i = 0; i < cpr_->supernodes_num; i++){
          //   supernode_t &spnode = cpr_->supernodes[i];
          //   auto& oes_d = spnode.inner_delta;
          //   auto& oes_v = spnode.inner_value;
          //   auto& value = values[spnode.id];
          //   // auto delta = atomic_exch(spnode_datas[spnode.id], app_->default_v()); // csr
          //   auto& delta = spnode_datas[spnode.id];
          //   /* filter useless delta */
          //   if(delta != app_->default_v()){
          //     app_->g_index_func_delta(*graph_, spnode.id, value, delta, oes_d); //If the threshold is small enough when calculating the index, it can be omitted here
          //     app_->g_index_func_value(*graph_, spnode.id, value, delta, oes_v);
          //   }
          //   delta = app_->default_v();
          // }
          /* 注意: 加了mirror之后,与原来的不同,入口点累积的消息需要同时发送给入口点所在的
              超点和其入口mirror所在的超点.
           */
          parallel_for(vid_t i = 0; i < cpr_->all_node_num; i++) {

            // printf("i is %d\n",i);
            vertex_t u(i);
            auto& delta = spnode_datas[u];
            if(delta != app_->default_v()){
              vid_t cid = cpr_->id2spids[u];
              vid_t c_node_num = cpr_->supernode_ids[cid].size();
              
              if(isChange(delta, c_node_num)){
                vid_t sp_id = cpr_->Fc_map[u];
                
                supernode_t &spnode = cpr_->supernodes[sp_id];
                
                auto& value = values[spnode.id];

                auto& oes_d = spnode.inner_delta;
                auto& oes_v = spnode.inner_value;
                app_->g_index_func_delta(*graph_, spnode.id, value, delta, oes_d); //If the threshold is small enough when calculating the index, it can be omitted here
                app_->g_index_func_value(*graph_, spnode.id, value, delta, oes_v);
                delta = app_->default_v();
              }

            // const char type = node_type[i];
            // if(type == NodeType::OnlyInNode || type == NodeType::BothOutInNode){
            //   auto& delta = spnode_datas[u];
            //   auto& master_delta = master_datas[u];
            //   vid_t ids_id = cpr_->id2spids[u];         
            //   // LOG(INFO) << "spnode_datas[" << graph_->GetId(u) << "]=" << delta;
            //   for(auto mp : cpr_->shortcuts[i]) {
            //     vid_t sp_id = mp.second;
            //     supernode_t &spnode = cpr_->supernodes[sp_id];
            //     auto& oes_d = spnode.inner_delta;
            //     auto& oes_v = spnode.inner_value;
            //     auto& value = values[spnode.id];
            //     if (mp.first == ids_id) {
            //       if(delta != app_->default_v()){
            //         app_->g_index_func_delta(*graph_, spnode.id, value, delta, oes_d); //If the threshold is small enough when calculating the index, it can be omitted here
            //         app_->g_index_func_value(*graph_, spnode.id, value, delta, oes_v);
            //       }
            //     }
            //     if (mp.first != ids_id) { // im-mirror
            //       if(master_delta != app_->default_v()){
            //         // LOG(INFO) << "master_delta[" << graph_->GetId(u) << "]=" 
            //                   // << master_delta;
            //         app_->g_index_func_delta(*graph_, spnode.id, value, 
            //                                   master_delta, oes_d); //If the threshold is small enough when calculating the index, it can be omitted here
            //         app_->g_index_func_value(*graph_, spnode.id, value,
            //                                   master_delta, oes_v);
            //       }
            //     }
            //   }
            //   master_delta = app_->default_v();
              // delta = app_->default_v();
            }
          }
          
          #ifdef DEBUG
            LOG(INFO) << "one_step_time=" << one_step_time;
          #endif
          corr_time += GetCurrentTime();
          LOG(INFO) << "correct deviation in supernode";
          LOG(INFO) << "#first iter step: " << step;
          LOG(INFO) << "#first exec_time: " << exec_time;
          LOG(INFO) << "#corr_time: " << corr_time;
          print_active_edge("#localAss");
          compr_stage = false;
          // print_result();
          continue;//这里continue之后又使用ingress阶段算法
        }

        if (batch_stage) {
          batch_stage = false;

          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#iter step: " << step;
            LOG(INFO) << "#Batch time: " << exec_time << " sec";
            print_active_edge("#Batch");
          }
          exec_time = 0;
          corr_time = 0;
          step = 0;
          convergence_id = 0;

          if (!FLAGS_efile_update.empty()) {//检查更新
            LOG(INFO) << "----------------------------------------------------";
            LOG(INFO) << "------------------INC COMPUTE-----------------------";
            LOG(INFO) << "----------------------------------------------------";
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
            values.fake2buffer();
            deltas.fake2buffer();
            cudaMemcpy(deltas_d, deltas.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
            cudaMemcpy(values_d, values.data_buffer, sizeof(value_t) * num, cudaMemcpyHostToDevice);
            if(compr_stage){
              first_step(true);  // inc is true
            }
            continue;
          }
        } else {
          if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
            LOG(INFO) << "#Inc iter step: " << step;
            LOG(INFO) << "#Inc time: " << exec_time << " sec";
            print_active_edge("#curr");
          }
          break;
        }
      }
    }
    free(size_oe_h);
    free(size_ib_h);
    free(size_is_h);
    free(size_sync_h);

    free(oeoffset_h);
    free(iboffset_h);
    free(isoffset_h);
    free(syncoffset_h);

    free(cur_oeoff_h);
    free(cur_iboff_h);
    free(cur_isoff_h);
    free(cur_syncoff_h);

    free(node_type_h);

    cudaFree(deltas_d);
    cudaFree(values_d);
    cudaFree(bound_node_values_d);
    cudaFree(spnode_datas_d);

    cudaFree(oeoffset_d);
    cudaFree(iboffset_d);
    cudaFree(isoffset_d);
    cudaFree(syncoffset_d);

    cudaFree(cur_oeoff_d);
    cudaFree(cur_iboff_d);
    cudaFree(cur_isoff_d);
    cudaFree(cur_syncoff_d);

    cudaFree(size_oe_d);
    cudaFree(size_ib_d);
    cudaFree(size_is_d);
    cudaFree(size_sync_d);

    cudaFree(node_type_d);
    

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

    check_result();

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

  void print_result(){
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    LOG(INFO) << "-----------result---s------------";
    for (auto v : inner_vertices) {
      vertex_t p;
      LOG(INFO) << "oid=" << graph_->GetId(v) << " id=" << v.GetValue() 
                << ": value=" << values[v] << " delta=" << deltas[v];
    }
    LOG(INFO) << "-----------result---e------------";
  }

  void check_result(std::string position = ""){
    auto inner_vertices = graph_->InnerVertices();
    auto& values = app_->values_;
    auto& deltas = app_->deltas_;
    double value_sum = 0;
    double delta_sum = 0;
    LOG(INFO) << "----------check_result in " << position;
    for (auto v : inner_vertices) {
      if (values[v] != app_->default_v()) {
        value_sum += values[v];
      }
      if (deltas[v] != app_->default_v()) {
        delta_sum += deltas[v];
      }
    }
    printf("---value_sum=%.10lf\n", value_sum);
    printf("---delta_sum=%.10lf\n", delta_sum);
  }

  void Finalize() { messages_.Finalize(); }

 private:
  bool termCheck(VertexArray<value_t, vid_t>& last_values,
                 VertexArray<value_t, vid_t>& values, bool compr_stage) {
    terminate_checking_time_ -= GetCurrentTime();
    auto vertices = graph_->InnerVertices();
    double diff_sum = 0, global_diff_sum = 0;

    if (FLAGS_portion >= 1) {
      for (auto u : vertices) {
        diff_sum += fabs(app_->deltas_[u]);
      }
      LOG(INFO) << " use priority...";
    } else {
      for (auto u : vertices) {
        diff_sum += fabs(last_values[u] - values[u]);
        last_values[u] = values[u];
      }
    }

    communicator_.template Sum(diff_sum, global_diff_sum);

    if (comm_spec_.worker_id() == grape::kCoordinatorRank) {
      LOG(INFO) << "Diff: " << global_diff_sum<<"diff: "<<diff_sum;
    }

    double bound_value = 0;
    if (FLAGS_compress == true) {
      for (auto u : vertices) {
        bound_value += bound_node_values[u];
      }
    }
    if (global_diff_sum < FLAGS_termcheck_threshold 
        && bound_value > FLAGS_termcheck_threshold) {
      LOG(INFO) << "---------------------------------";
      LOG(INFO) << "  bound_value=" << bound_value;
      LOG(INFO) << "---------------------------------";
    }

    terminate_checking_time_ += GetCurrentTime();
    return global_diff_sum < FLAGS_termcheck_threshold; // 这个阈值应该修改为在sketch阶段的阈值应该是: sketch点数/总点数*阈值
  }

  bool isChange(value_t delta, vid_t c_node_num=1) {
    if (FLAGS_portion >= 1) {
      if (std::fabs(delta) * c_node_num 
            > FLAGS_termcheck_threshold/graph_->GetVerticesNum()) {
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  void print_active_edge(std::string position = "") {
    LOG(INFO) << position << "_f_index_count_num: " << app_->f_index_count_num;
    LOG(INFO) << position << "_f_send_delta_num: " << app_->f_send_delta_num;
    app_->f_index_count_num = 0;
    app_->f_send_delta_num = 0;
  }

  // VertexArray<vid_t, vid_t> spnode_ids;
  std::vector<char> node_type; // all node's types, 0:out node, 1:bound node, 2:source node, 3:belong 1 and 2 at the same time, 4:inner node that needn't send message.
  VertexArray<value_t, vid_t> bound_node_values; // 入口点发给出口点的delta
  std::shared_ptr<APP_T> app_;
  std::shared_ptr<fragment_t>& graph_;
  message_manager_t messages_;
  Communicator communicator_;
  CommSpec comm_spec_;
  double terminate_checking_time_;
  IterCompressor<APP_T, supernode_t>* cpr_;
  // std::vector<value_t> spnode_datas;
  VertexArray<value_t, vid_t> spnode_datas{}; // 入口点收到的delta累积值
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
