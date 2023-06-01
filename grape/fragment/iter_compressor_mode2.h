#ifndef GRAPE_FRAGMENT_ITER_COMPRESSOR_H_
#define GRAPE_FRAGMENT_ITER_COMPRESSOR_H_

#include "grape/graph/super_node.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "timer.h"
#include "flags.h"
#include <iomanip>
#include "grape/fragment/compressor_base.h"
#include <atomic>

// #define NUM_THREADS 52

namespace grape {

template <typename APP_T, typename SUPERNODE_T>
class IterCompressor : public CompressorBase <APP_T, SUPERNODE_T> {
    public:
    using fragment_t = typename APP_T::fragment_t;
    using value_t = typename APP_T::value_t;
    using vertex_t = typename APP_T::vertex_t;
    using vid_t = typename APP_T::vid_t;
    using supernode_t = SUPERNODE_T;
    using fc_t = int32_t;
    using nbr_t = typename fragment_t::nbr_t;
    using adj_list_t = typename fragment_t::adj_list_t;
    double supernode_termcheck_threshold = FLAGS_termcheck_threshold/10000; // to build index
    double supernode_termcheck_threshold_2 = FLAGS_termcheck_threshold/1000; // to pre compute
    const bool iter_compressor_flags_cilk = false;

    IterCompressor(std::shared_ptr<APP_T>& app,
                        std::shared_ptr<fragment_t>& graph):CompressorBase<APP_T, SUPERNODE_T>(app, graph){}

    void run(){
        /* init */
        // int thread_num = FLAGS_build_index_concurrency;
        int thread_num =  52; // batch阶段不进行计时，为了节省时间，此处线程开满！！！ 
        test_time.resize(thread_num);
        values_array.resize(thread_num);
        deltas_array.resize(thread_num);
        bounds_array.resize(thread_num);
        LOG(INFO) << "num=" << values_array.size() << " " << deltas_array.size() << " " << bounds_array.size()
        << " " << test_time.size();
        LOG(INFO) << "#build_index_concurrency: " << thread_num;
        double s = GetCurrentTime();
        parallel_for(int tid = 0; tid < thread_num; tid++){
            auto inner_vertices = this->graph_->InnerVertices();
            values_array[tid].Init(inner_vertices);
            deltas_array[tid].Init(inner_vertices);
            bounds_array[tid].Init(this->graph_->Vertices());
            test_time[tid].resize(4); // debug
        }
        LOG(INFO) << "init time=" << (GetCurrentTime()-s);
        this->app_->Init(this->comm_spec_, *(this->graph_), false);
        init_deltas.Init(this->graph_->Vertices(), this->app_->default_v()); // note: include out vertex
        this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);

        // 设置每个点收敛的阈值：
        supernode_termcheck_threshold = FLAGS_termcheck_threshold / this->graph_->GetVerticesNum();
        supernode_termcheck_threshold_2 = FLAGS_termcheck_threshold / this->graph_->GetVerticesNum();

        /* find supernode */
        timer_next("find supernode");
        {  
            LOG(INFO) << "start compress...";
            this->compress();
            LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish compress...";
            const vid_t spn_ids_num = this->supernode_ids.size();
            if (iter_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
                LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
                parallel_for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                    judge_out_bound_node(j, this->graph_);
                }
            }
            else{
#pragma omp parallel for num_threads(NUM_THREADS)
                for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                    judge_out_bound_node(j, this->graph_);
                }
#pragma omp barrier
            }
            /* build subgraph of supernode */
            build_subgraph(this->graph_);
        }

        /* calculate index for each structure */
        timer_next("calculate index");
        {
            /* parallel */
            /* Simulate thread pool */
            // vid_t spnode_id = 0;
            std::atomic<vid_t> spnode_id(0);
            std::atomic<vid_t> active_thread_num(thread_num);
            this->ForEach(this->supernodes_num, [this, &spnode_id, &active_thread_num](int tid) {
                // LOG(INFO) << "build index, tid=" << tid << " begin..." << this->supernodes_num;
                int i = 0, cnt = 0, step = 1;  // step need to be adjusted
                while(i < this->supernodes_num){
                    // i = __sync_fetch_and_add(&spnode_id, step);
                    i = spnode_id.fetch_add(step);
                    for(int j = i; j < i + step; j++){
                        if(j < this->supernodes_num){
                            build_iter_index(j, this->graph_, tid);
                            cnt++;
                        }
                        else{
                            break;
                        }
                    }
                }
                // debug
                {
                    active_thread_num.fetch_sub(1);
                    LOG(INFO) << "tid=" << tid 
                     << " time0=" << test_time[tid][0]
                     << " time1=" << test_time[tid][1]
                     << " time2=" << test_time[tid][2]
                     << " time1_max=" << test_time[tid][3] 
                     << " i=" << i << " spnode_id=" << spnode_id;
                }
                }, thread_num
            );
        }
        LOG(INFO) << "calculate finish!";

        /* precompute supernode */
        timer_next("pre compute");
        {
            const vid_t spn_ids_num = this->supernode_ids.size();
            if (iter_compressor_flags_cilk) {
    #ifdef INTERNAL_PARALLEL
            LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
    #endif
                parallel_for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                    run_to_convergence_for_precpt(j);
                    if(j % 1000000 == 0){
                        LOG(INFO) << "----id=" << j << " pre compute" << std::endl;
                    }
                }
            }
            else{
                /* Simulate thread pool */
                // std::atomic<vid_t> spn_ids_id(0);
                // int thread_num = NUM_THREADS; // use max thread_num
                // this->ForEach(spn_ids_num, [this, &spn_ids_id, &spn_ids_num](int tid) {
                //     // LOG(INFO) << "precompute, tid=" << tid << " begin..." << spn_ids_num;
                //     int i = 0, cnt = 0, step = 1;
                //     while(i < spn_ids_num){
                //         // i = __sync_fetch_and_add(&spn_ids_id, step);
                //         i = spn_ids_id.fetch_add(step);
                //         for(int j = i; j < i + step; j++){
                //             if(j < spn_ids_num){
                //                 run_to_convergence_for_precpt(j);
                //                 cnt++;
                //             }
                //             else{
                //                 break;
                //             }
                //         }
                //     }
                //     // LOG(INFO) << "pre compute tid=" << tid << " finish! cnt=" << cnt;
                //     }, thread_num
                // );
                /* cilk */
                clean_deltas();
                #pragma cilk grainsize = 1
                parallel_for(vid_t i = 0; i < spn_ids_num; i++){
                    run_to_convergence_for_precpt(i);
                }
            }
            auto& deltas = this->app_->deltas_;
            vid_t node_num = this->graph_->Vertices().end().GetValue();
            for(vid_t tid = 0; tid < FLAGS_build_index_concurrency; tid++){
                VertexArray<value_t, vid_t>& self_deltas = deltas_array[tid];
                parallel_for(vid_t i = 0; i < node_num; i++){
                    vertex_t v(i);                     
                    this->app_->accumulate(deltas[v], self_deltas[v]);
                }
            }
            // MPI_Barrier(this->comm_spec_.comm());
        }

        /* debug */
        // print();
        // timer_next("write spnodes");
        // write_spnodes("../Dataset/spnodes" + std::to_string(this->comm_spec_.worker_id()));
    }

    void clean_deltas(){
        double start = GetCurrentTime();
        vid_t node_num = this->graph_->Vertices().end().GetValue();
        for(vid_t tid = 0; tid < FLAGS_build_index_concurrency; tid++){
            VertexArray<value_t, vid_t>& self_deltas = deltas_array[tid];
            parallel_for(vid_t i = 0; i < node_num; i++){
                vertex_t v(i);                                                                                          
                self_deltas[v] = this->app_->default_v();
            }
        }
        LOG(INFO) << "---clean_delta_time=" << (GetCurrentTime()-start);
    }

    void build_subgraph(const std::shared_ptr<fragment_t>& new_graph){
        double subgraph_time = GetCurrentTime();
        const auto& inner_vertices = new_graph->InnerVertices();
        const vid_t spn_ids_num = this->supernode_ids.size();
        vid_t inner_node_num = inner_vertices.end().GetValue() - inner_vertices.begin().GetValue();
        std::vector<size_t> ia_oe_degree(inner_node_num+1, 0);
        std::vector<size_t> ib_oe_degree(inner_node_num+1, 0);
        vid_t ia_oe_num = 0; 
        vid_t ib_oe_num = 0; 
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
        // for(vid_t i = 0; i < spn_ids_num; i++){
            std::vector<vertex_t> &node_set = this->supernode_ids[i];
            vid_t temp_a = 0;
            vid_t temp_b = 0;
            for(auto v : node_set){
                auto ids_id = this->id2spids[v];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_degree[v.GetValue()+1]++;
                        temp_a++;
                    }
                    else{
                        ib_oe_degree[v.GetValue()+1]++;
                        temp_b++;
                    }
                }
            }
            atomic_add(ia_oe_num, temp_a);
            atomic_add(ib_oe_num, temp_b);
        }
        ia_oe_.clear();
        ia_oe_.resize(ia_oe_num);
        ia_oe_offset_.clear();
        ia_oe_offset_.resize(inner_node_num+1);
        ib_oe_.clear();
        ib_oe_.resize(ib_oe_num);
        ib_oe_offset_.clear();
        ib_oe_offset_.resize(inner_node_num+1);

        for(vid_t i = 1; i < inner_node_num; i++) {
            ia_oe_degree[i] += ia_oe_degree[i-1];
            ib_oe_degree[i] += ib_oe_degree[i-1];
        }

        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        // for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            vid_t index_a = ia_oe_degree[i];
            ia_oe_offset_[i] = &ia_oe_[index_a];
            vid_t index_b = ib_oe_degree[i];
            ib_oe_offset_[i] = &ib_oe_[index_b];
            if(this->Fc[u] != this->FC_default_value){
                auto ids_id = this->id2spids[u];
                const auto& oes = new_graph->GetOutgoingAdjList(u);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_[index_a] = oe;
                        index_a++;
                    }
                    else{
                        ib_oe_[index_b] = oe;
                        index_b++;
                    }
                }
            }
            // CHECK_EQ(index_s, ia_oe_degree[i+1]);
        }
        ia_oe_offset_[inner_node_num] = &ia_oe_[ia_oe_num-1] + 1;
        ib_oe_offset_[inner_node_num] = &ib_oe_[ib_oe_num-1] + 1;
        LOG(INFO) << "---subgraph_time=" << (GetCurrentTime()-subgraph_time);
    }

    void judge_out_bound_node(const vid_t superid, const std::shared_ptr<fragment_t>& new_graph){
        std::vector<vertex_t> &node_set = this->supernode_ids[superid]; 
        for(auto v : node_set){
            auto spids = this->id2spids[v];
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            for(auto& e : oes){
                if(this->id2spids[e.neighbor] != spids){
                    this->supernode_out_bound[v.GetValue()] = true;
                    break;
                }
            }
        }
    }

    /**
     * To compute indexes in parallel, use a VertexArray
    */
    void build_iter_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas */
        test_time[tid][0] -= GetCurrentTime();
        for(auto v : node_set){
            this->app_->init_c(v, deltas[v], *new_graph, source);
            this->app_->init_v(v, values[v]);
        }
        test_time[tid][0] += GetCurrentTime();
        /* iterative calculation */
        test_time[tid][1] -= GetCurrentTime();
        double b = GetCurrentTime();
        inc_run_to_convergence(tid, new_graph, node_set, source);
        test_time[tid][3] = std::max(test_time[tid][3], GetCurrentTime()-b);
        test_time[tid][1] += GetCurrentTime();
        /* build new index in supernodes */
        test_time[tid][2] -= GetCurrentTime();
        // fianl_build_iter_index2(tid, new_graph, node_set, spnode);  // 线程增多时，时间会增多？？？？
        fianl_build_iter_index_for_mode2(tid, new_graph, node_set, spnode);  // 线程增多时，时间会增多？？？？
        test_time[tid][2] += GetCurrentTime();
    }

    /* use a bounds_array */
    void fianl_build_iter_index_for_mode2(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[spnode.id];
        spnode.inner_value.clear();
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();

        for(auto v : node_set){
            auto& value = values[v];
            auto& delta = deltas[v];
            if(value != this->app_->default_v()){
                value_t rt_value;
                this->app_->g_revfunction(value, rt_value);
                if(this->supernode_out_bound[v.GetValue()]){
                    spnode.bound_delta.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
                }
                else {
                    spnode.inner_value.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
                }
            }
            /* inner_delta index, if there is absolute internal convergence, this index may not be established */
            if(delta != this->app_->default_v()){
                value_t rt_value;
                this->app_->g_revfunction(delta, rt_value);
                spnode.inner_delta.emplace_back(std::pair<vertex_t, value_t>(v, rt_value));
            }
        }
    }

    /* Used for precomputing */
    void run_to_convergence_for_precpt(const vid_t superid){
        std::vector<vertex_t> &node_set = this->supernode_ids[superid]; 
        int step = 0;
        float Diff = 0;
        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;
        size_t node_set_size = node_set.size();
        // auto spids = this->id2spids[node_set[0]];

        int thread_id = __cilkrts_get_worker_number();
        #ifdef DEBUG
            CHECK(thread_id < deltas_array.size());
        #endif
        VertexArray<value_t, vid_t>& self_deltas = deltas_array[thread_id];
        VertexArray<value_t, vid_t>& old_values = values_array[0];
        for(int i = 0; i < node_set_size; i++){
            vertex_t& u = node_set[i];
            old_values[u] = values[u];
            values[u] = this->app_->default_v();
        }

        double threshold_full = supernode_termcheck_threshold_2  * node_set_size;
        double threshold_filter = supernode_termcheck_threshold_2/10;
        // bool have_active = false;

        while (true) {
            step++;
            Diff = 0;
            // have_active = false;
            // receive & send
            for(auto v : node_set){
                auto to_send = deltas[v];
                if(to_send != this->app_->default_v()){
                // if(to_send >= threshold_filter){

                    auto last_value = values[v];
                    deltas[v] = this->app_->default_v();
                    auto& value = values[v];
                    const auto& oes = this->graph_->GetOutgoingAdjList(v);
                    /* inner edges */
                    const adj_list_t& a_oes = adj_list_t(ia_oe_offset_[v.GetValue()], ia_oe_offset_[v.GetValue()+1]);
                    auto a_out_degree = a_oes.Size();
                    if(a_out_degree > 0){
                        for(auto e : a_oes){
                            value_t outv = 0;
                            this->app_->g_function(*(this->graph_), v, value, to_send, oes, e, outv);
                            this->app_->accumulate(deltas[e.neighbor], outv); // inner nodes
                        }
                    }
                    this->app_->accumulate(value, to_send); // 这里不需要原子操作!!!
                    Diff += fabs(last_value - values[v]);
                    // have_active = true;
                }
            }
            // check convergence
            if(Diff <= threshold_full || step > 100){
            // if(have_active == false || step > 100){
            // if(Diff <= FLAGS_termcheck_threshold || step > 100){
                // if(step > 100){
                //     LOG(INFO) << "compress: step>200 Diff=" << Diff;
                // }
                break;
            }
        }

        /* bound edges */
        for(int i = 0; i < node_set_size; i++){
            vertex_t& v = node_set[i];
            const adj_list_t& b_oes = adj_list_t(ib_oe_offset_[v.GetValue()], ib_oe_offset_[v.GetValue()+1]);
            auto b_out_degree = b_oes.Size();
            value_t& value = values[v];
            value_t old_value = old_values[v];
            value_t to_send = value;
            this->app_->accumulate(value, old_value);
            if(b_out_degree > 0){
                const auto& oes = this->graph_->GetOutgoingAdjList(v);
                for(auto e : b_oes){
                    value_t outv = 0;
                    this->app_->g_function(*(this->graph_), v, value, to_send, oes, e, outv);
                    // this->app_->accumulate_atomic(deltas[e.neighbor], outv); // 或者先本地缓存,最后一次发出去，上面的atomic_exch都去掉
                    // this->app_->accumulate_atomic(init_deltas[e.neighbor], outv); // out nodes
                    // 上面使用了原子操作，可以通过写入各自线程的delta数组，去掉原子操作，最后合并一次！！！
                    this->app_->accumulate(self_deltas[e.neighbor], outv); // out nodes
                }
                
                // 注意只能在cilk开的线程数和计算索引线程数一样多时才能用！
                // auto it = oes.begin();
                // granular_for(j, 0, b_out_degree, (b_out_degree > 1024), {
                //     auto& e = *(it + j);
                //     value_t outv = 0;
                //     this->app_->g_function(*(this->graph_), v, value, to_send, oes, e, outv);
                //     this->app_->accumulate_atomic(init_deltas[e.neighbor], outv); // out nodes
                // })
            }
        }
    }

    void init_node(const std::vector<vertex_t> &node_set, const vertex_t& source){
        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;
        for(auto v : node_set){
            this->app_->init_c(v, deltas[v], *(this->graph_), source);
            this->app_->init_v(v, values[v]);
        }
    }

    void inc_run(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges, const std::shared_ptr<fragment_t>& new_graph){
        /* Switch data to support add new nodes */
        LOG(INFO) << "Switch data";
        VertexArray<fc_t, vid_t> old_Fc;
        auto old_vertices = this->graph_->Vertices();
        old_Fc.Init(old_vertices, this->FC_default_value);
        parallel_for(vid_t i = old_vertices.begin().GetValue(); i < old_vertices.end().GetValue(); i++) {
            vertex_t v(i);
            old_Fc[v] = this->Fc[v];
        }
        this->Fc.Init(new_graph->Vertices(), this->FC_default_value);
        // copy to new graph
        parallel_for(vid_t i = new_graph->Vertices().begin().GetValue(); i < new_graph->Vertices().end().GetValue(); i++) {
            vertex_t v(i);
            this->Fc[v] = old_Fc[v];
        }

        VertexArray<vid_t, vid_t> old_Fc_map;
        auto old_inner_vertices = this->graph_->InnerVertices();
        old_Fc_map.Init(old_inner_vertices);
        // for(auto v : old_inner_vertices){
        parallel_for(vid_t i = old_inner_vertices.begin().GetValue(); i < old_inner_vertices.end().GetValue(); i++) {
            vertex_t v(i);
            old_Fc_map[v] = this->Fc_map[v];
        }
        this->Fc_map.Init(new_graph->InnerVertices(), this->ID_default_value);
        // copy to new graph
        // for(auto v : new_graph->InnerVertices()) {
        parallel_for(vid_t i = new_graph->InnerVertices().begin().GetValue(); i < new_graph->InnerVertices().end().GetValue(); i++) {
            vertex_t v(i);
            this->Fc_map[v] = old_Fc_map[v];
        }

        VertexArray<vid_t, vid_t> old_id2spids;
        old_id2spids.Init(this->graph_->Vertices());
        // for(auto v : old_inner_vertices){
        parallel_for(vid_t i = old_inner_vertices.begin().GetValue(); i < old_inner_vertices.end().GetValue(); i++) {
            vertex_t v(i);
            old_id2spids[v] = this->id2spids[v];
        }
        this->id2spids.Init(new_graph->Vertices(), this->ID_default_value);
        // copy to new graph
        // for(auto v : new_graph->InnerVertices()) {
        parallel_for(vid_t i = new_graph->InnerVertices().begin().GetValue(); i < new_graph->InnerVertices().end().GetValue(); i++) {
            vertex_t v(i);
            this->id2spids[v] = old_id2spids[v];
        }

        init_deltas.Init(new_graph->Vertices(), this->app_->default_v()); // note: include out vertex

        /* find supernode */
        timer_next("inc compress");
        double inc_compress = GetCurrentTime();
        // this->inc_compress(deleted_edges, added_edges);
        this->parallel_inc_compress(deleted_edges, added_edges);
        inc_compress = GetCurrentTime()-inc_compress;
        LOG(INFO) << "#inc_compress: " << inc_compress;
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish inc compress...";
        // MPI_Barrier(this->comm_spec_.comm()); // 仅在测试find阶段时间时需要同步，后面可以删除

        timer_next("init bound_ids");
        /* init supernode_out_bound*/
        const vid_t spn_ids_num = this->supernode_ids.size();
        double begin = GetCurrentTime();
        this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);
        if (iter_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
            LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                judge_out_bound_node(j, new_graph);
            }
        }
        else{
#pragma omp parallel for num_threads(NUM_THREADS)
            for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                judge_out_bound_node(j, new_graph);
            }
#pragma omp barrier
        }
        /* build subgraph of supernode */
        build_subgraph(new_graph);
        LOG(INFO) << "init supernode_out_bound time=" << (GetCurrentTime() - begin);

        /* calculate index for each structure */
        timer_next("inc calculate index");
        double inc_calculate_index = GetCurrentTime();
        inc_compute_index(new_graph);
        inc_calculate_index = GetCurrentTime() - inc_calculate_index;
        LOG(INFO) << "#inc_calculate_index: " << inc_calculate_index;
        timer_next("graph switch");

        /* debug */
        // print();
        // timer_next("inc write pattern");
        // write_spnodes("../Dataset/inc_spnodes_incindex" + std::to_string(this->comm_spec_.worker_id()));
    }

    void inc_precompute_supernode(){
        /* precompute supernode */
        timer_next("inc pre compute");
        double inc_pre_compute = GetCurrentTime();
        vid_t spn_ids_num = this->supernode_ids.size();
        if (iter_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t j = 0; j < spn_ids_num; j++){  // parallel compute
                run_to_convergence_for_precpt(j);
            }
        }
        else{
            /* Simulate thread pool */
            // std::atomic<vid_t> spn_ids_id(0);
            // int thread_num = NUM_THREADS; // use max thread_num
            // this->ForEach(spn_ids_num, [this, &spn_ids_id, &spn_ids_num](int tid) {
            //     // LOG(INFO) << "precompute, tid=" << tid << " begin..." << spn_ids_num;
            //     int i = 0, cnt = 0, step = 1;;
            //     while(i < spn_ids_num){
            //         // i = __sync_fetch_and_add(&spn_ids_id, step);
            //         i = spn_ids_id.fetch_add(step);
            //         for(int j = i; j < i + step; j++){
            //             if(j < spn_ids_num){
            //                 run_to_convergence_for_precpt(j);
            //                 cnt++;
            //             }
            //             else{
            //                 break;
            //             }
            //         }
            //     }
            //     // LOG(INFO) << "tid=" << tid << " finish! cnt=" << cnt;
            //     }, thread_num
            // );

            clean_deltas();
            // #pragma cilk grainsize = 1
            parallel_for(vid_t i = 0; i < spn_ids_num; i++){
                run_to_convergence_for_precpt(i);
            }
        }
        // double t = GetCurrentTime();
        auto& deltas = this->app_->deltas_;
        vid_t node_num = this->graph_->Vertices().end().GetValue();
        // parallel_for(vid_t i = 0; i < node_num; i++) {
        //     vertex_t v(i);
        //     this->app_->accumulate(deltas[v], init_deltas[v]);
        // }
        for(vid_t tid = 0; tid < FLAGS_build_index_concurrency; tid++){
            VertexArray<value_t, vid_t>& self_deltas = deltas_array[tid];
            parallel_for(vid_t i = 0; i < node_num; i++) {
                vertex_t v(i);
                this->app_->accumulate(deltas[v], self_deltas[v]);
            }
        }
        // LOG(INFO) << "debug t_time=" << (GetCurrentTime()-t);
        inc_pre_compute = GetCurrentTime()-inc_pre_compute;
        LOG(INFO) << "#inc_pre_compute: " << inc_pre_compute;
        LOG(INFO) << "finish inc_precompute_supernode...";
    }

    void inc_compute_index(const std::shared_ptr<fragment_t>& new_graph){
        {
            double test_init_vector = GetCurrentTime();
            std::vector<vid_t> ids;
            // ids.reserve(this->inccalculate_spnode_ids.size());
            ids.insert(ids.begin(), this->inccalculate_spnode_ids.begin(), this->inccalculate_spnode_ids.end());
            LOG(INFO) << "---test_init_vector=" << (GetCurrentTime()-test_init_vector);
            int len = ids.size();
            /* Simulate thread pool */
            std::atomic<vid_t> spnode_id(0);
            int thread_num = FLAGS_build_index_concurrency;
            this->ForEach(len, [this, &spnode_id, &ids, &len, &new_graph](int tid) {
                int i = 0, cnt = 0;
                while(i < len){
                    // i = __sync_fetch_and_add(&spnode_id, 1);
                    i = spnode_id.fetch_add(1);
                    if(i < len){
                        vertex_t u(ids[i]);
                        if(this->Fc_map[u] != this->ID_default_value){
                            inc_build_iter_index(this->Fc_map[u], new_graph, tid); // inc
                            // build_iter_index(this->Fc_map[u], new_graph, tid); // re compute
                        }
                        cnt++;
                    }
                }
                }, thread_num
            );
            
            // #pragma cilk grainsize = 1
            // parallel_for(vid_t i = 0; i < len; i++){
            //     vertex_t u(ids[i]);
            //     if(this->Fc_map[u] != this->ID_default_value){
            //         int tid = __cilkrts_get_worker_number();
            //         #ifdef DEBUG
            //             //CHECK(tid < FLAGS_build_index_concurrency);
            //         #endif
            //         inc_build_iter_index(this->Fc_map[u], new_graph, tid); // inc
            //         // build_iter_index(this->Fc_map[u], new_graph, tid); // re compute
            //     }
            // }
        }
        // The newly added index must be recalculated
        {
            double test_init_vector = GetCurrentTime();
            std::vector<vid_t> ids;
            // ids.reserve(this->inccalculate_spnode_ids.size());
            ids.insert(ids.begin(), this->recalculate_spnode_ids.begin(), this->recalculate_spnode_ids.end());
            LOG(INFO) << "---test_init_vector=" << (GetCurrentTime()-test_init_vector);
            int len = ids.size();
            /* parallel */
            /* Simulate thread pool */
            std::atomic<vid_t> spnode_id(0);
            int thread_num = FLAGS_build_index_concurrency;
            this->ForEach(len, [this, &spnode_id, &ids, &len, &new_graph](int tid) {
                int i = 0, cnt = 0;
                while(i < len){
                    // i = __sync_fetch_and_add(&spnode_id, 1);
                    i = spnode_id.fetch_add(1);
                    if(i < len){
                        vertex_t u(ids[i]);
                        if(this->Fc_map[u] != this->ID_default_value){
                            build_iter_index(this->Fc_map[u], new_graph, tid);
                        }
                        cnt++;
                    }
                }
                }, thread_num
            );

            // #pragma cilk grainsize = 1
            // parallel_for(vid_t i = 0; i < len; i++){
            //     vertex_t u(ids[i]);
            //     if(this->Fc_map[u] != this->ID_default_value){
            //         int tid = __cilkrts_get_worker_number();
            //         #ifdef DEBUG
            //             //CHECK(tid < FLAGS_build_index_concurrency);
            //         #endif
            //         build_iter_index(this->Fc_map[u], new_graph, tid);
            //     }
            // }
        }
    }

    /* use a VertexArray */
    void inc_build_iter_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const vertex_t& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas by inner_delta and inner_value, 
           Note that the inner_value and inner_delta at this time have been converted into weights.
           g_revfunction()
        */
        for(auto v : node_set){
            values[v] = this->app_->default_v();
            deltas[v] = this->app_->default_v();
        }
        for(auto e : spnode.inner_value){
            values[e.first] = e.second;  // 这里直接使用了!!!
        }
        for(auto e : spnode.inner_delta){
            deltas[e.first] = e.second;
        }        
        // If you are using cos model two, you need to copy the following values.
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second;  // 这里直接使用了!!!
        }
        
        /* recycled value on the old graph */
        AmendValue(-1, tid, this->graph_, node_set, source);
        /* reissue value on the new graph */
        AmendValue(1, tid, new_graph, node_set, source);
        /* iterative calculation */
        inc_run_to_convergence(tid, new_graph, node_set, source);
        /* build new index in supernodes */
        // fianl_build_iter_index2(tid, new_graph, node_set, spnode);
        fianl_build_iter_index_for_mode2(tid, new_graph, node_set, spnode);
    }

    /* use a VertexArray */
    void AmendValue(int type, vid_t tid, const std::shared_ptr<fragment_t>& graph, const std::vector<vertex_t> &node_set, vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        const vid_t ids_id = this->id2spids[source];
        // send
        for(auto v : node_set){
            const auto& oes = graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            value_t to_send = value * type;
            if(value != this->app_->default_v()){
                // /*
                value_t outv = 0;
                for(auto& e : oes){
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        this->app_->g_function(*graph, v, value, to_send, oes, e, outv);
                        // this->app_->accumulate_atomic(deltas[e.neighbor], outv);
                        this->app_->accumulate(deltas[e.neighbor], outv);
                    }
                }
            }
        }
    }

    /* use a VertexArray */
    void inc_run_to_convergence(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<value_t, vid_t>& deltas = deltas_array[tid];
        // const vid_t ids_id = this->id2spids[source];
        int step = 0;
        float Diff = 0;
        //double now_termcheck_threshold = supernode_termcheck_threshold / node_set.size();
        size_t node_set_size = node_set.size();
        while (true) {
            step++;
            Diff = 0;
            // receive & send
            // for(auto v : node_set){
            for(vid_t i = 0; i < node_set_size; i++){
                const vertex_t& v = node_set[i];
                // auto to_send = atomic_exch(deltas[v], this->app_->default_v());
                auto to_send = deltas[v];
                auto last_value = values[v];
                deltas[v] = this->app_->default_v();
                if(to_send != this->app_->default_v()){
                    const auto& old_oes = new_graph->GetOutgoingAdjList(v);
                    const adj_list_t& oes = adj_list_t(ia_oe_offset_[v.GetValue()], ia_oe_offset_[v.GetValue()+1]);
                    auto& value = values[v];
                    value_t outv = 0;
                    for(auto& e : oes){
                    // for(auto& e : old_oes){
                    //     if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                            this->app_->g_function(*new_graph, v, value, to_send, old_oes, e, outv);
                            // this->app_->accumulate_atomic(deltas[e.neighbor], outv);
                            this->app_->accumulate(deltas[e.neighbor], outv); // 单线程
                        // }
                    }
                    // this->app_->accumulate_atomic(value, to_send);
                    this->app_->accumulate(value, to_send); // 单线程
                }
                Diff += fabs(last_value - values[v]);
            }
            // check convergence
            // if(Diff <= now_termcheck_threshold || step > 100){
            // if(Diff * node_set_size <= supernode_termcheck_threshold || step > 100){
            if(Diff <= supernode_termcheck_threshold * node_set_size || step > 100){
                // if(step > 100){
                //     LOG(INFO) << "compress: step>200 Diff=" << Diff;
                // }
                break;
            }
        }
    }

public:
    VertexArray<value_t, vid_t> init_deltas;
    std::vector<VertexArray<value_t, vid_t>> values_array; // use to calulate indexes in parallel
    std::vector<VertexArray<value_t, vid_t>> deltas_array;
    std::vector<VertexArray<value_t, vid_t>> bounds_array; // delta of bound vertex
    std::vector<std::vector<double>> test_time; // test time
    /* inner all nodes */
    Array<nbr_t, Allocator<nbr_t>> ia_oe_;
    Array<nbr_t*, Allocator<nbr_t*>> ia_oe_offset_;
    /* in_bound_node to out_bound_node */
    Array<nbr_t, Allocator<nbr_t>> ib_oe_;
    Array<nbr_t*, Allocator<nbr_t*>> ib_oe_offset_;
};

}  // namespace grape
#endif  // GRAPE_FRAGMENT_ITER_COMPRESSOR_H_