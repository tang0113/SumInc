#ifndef GRAPE_FRAGMENT_TRAV_COMPRESSOR_H_
#define GRAPE_FRAGMENT_TRAV_COMPRESSOR_H_

#include "grape/graph/super_node.h"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include "timer.h"
#include "flags.h"
#include <iomanip>
#include "grape/fragment/compressor_base.h"

// #define NUM_THREADS 52

namespace grape {

template <typename APP_T, typename SUPERNODE_T>
class TravCompressor : public CompressorBase <APP_T, SUPERNODE_T> {
    public:
    using fragment_t = typename APP_T::fragment_t;
    using value_t = typename APP_T::value_t;
    using vertex_t = typename APP_T::vertex_t;
    using vid_t = typename APP_T::vid_t;
    using supernode_t = SUPERNODE_T;
    using delta_t = typename APP_T::delta_t;
    using fc_t = int32_t;
    using nbr_t = typename fragment_t::nbr_t;
    using adj_list_t = typename fragment_t::adj_list_t;
    size_t total_node_set = 0;
    const bool trav_compressor_flags_cilk = false;

    TravCompressor(std::shared_ptr<APP_T>& app,
                        std::shared_ptr<fragment_t>& graph):CompressorBase<APP_T, SUPERNODE_T>(app, graph){}

    void run(){
        /* init */
        int thread_num = FLAGS_build_index_concurrency;
        // int thread_num =  52; // batch阶段不进行计时，为了节省时间，此处线程开满！！！ 
        test_time.resize(thread_num);
        values_array.resize(thread_num);
        deltas_array.resize(thread_num);
        LOG(INFO) << "num=" << values_array.size() << " " << deltas_array.size() 
        << " " << test_time.size();
        LOG(INFO) << "#build_index_concurrency: " << thread_num;
        double s = GetCurrentTime();
        parallel_for(int tid = 0; tid < thread_num; tid++){
            auto inner_vertices = this->graph_->InnerVertices();
            values_array[tid].Init(inner_vertices);
            deltas_array[tid].Init(inner_vertices);
            for (auto v : this->graph_->Vertices()) {
                deltas_array[tid][v] = this->app_->GetInitDelta(v);
            }
            test_time[tid].resize(4); // debug
        }
        LOG(INFO) << "init time=" << (GetCurrentTime()-s);
        // this->app_->Init(this->comm_spec_, *(this->graph_), false);
        // init_deltas.Init(this->graph_->Vertices()); // note: include out vertex
        // for (auto v : this->graph_->Vertices()) {
        //     init_deltas[v] = this->app_->GetInitDelta(v);
        // }
        this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);

        /* find supernode */
        timer_next("find supernode");
        {  
            LOG(INFO) << "start compress...";
            this->compress();
            LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish compress...";
            const vid_t spn_ids_num = this->supernode_ids.size();
            if (trav_compressor_flags_cilk) {
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

        // LOG(INFO) << "测试测试用了return!!!!!!!!!!!!!!!!!!!!!!!!!!";
        // return;

        /* calculate index for each structure */
        timer_next("calculate index");
        double calculate_index = GetCurrentTime();
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
                            build_trav_index(i, this->graph_, tid);
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
        calculate_index = GetCurrentTime() - calculate_index;
        LOG(INFO) << "#calculate_index: " << calculate_index;
        /* init supernode.data */
        if_touch.resize(this->supernodes_num);
        parallel_for(int i = 0; i < this->supernodes_num; i++){
            if_touch[i] = 1;
            this->supernodes[i].data = this->app_->GetIdentityElement();
        }
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish calculate index...";

        /* precompute supernode */
        timer_next("pre compute");
        double pre_compute = GetCurrentTime();
        precompute_spnode(this->graph_);
        pre_compute = GetCurrentTime() - pre_compute;
        LOG(INFO) << "#pre_compute: " << pre_compute;
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish precompute supernode...";

        // MPI_Barrier(this->comm_spec_.comm());

        /* debug */
        // this->print();
        // timer_next("write spnodes");
        // this->write_spnodes("../Dataset/spnodes" + std::to_string(this->comm_spec_.worker_id()));
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
        vid_t ia_oe_num = 0; 
        vid_t ib_oe_num = 0; 
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
        // for(vid_t i = 0; i < spn_ids_num; i++){
            std::vector<vertex_t> &node_set = this->supernode_ids[i];
            vid_t temp_a = 0;
            for(auto v : node_set){
                auto ids_id = this->id2spids[v];
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_degree[v.GetValue()+1]++;
                        temp_a++;
                    }
                }
            }
            atomic_add(ia_oe_num, temp_a);
        }
        ia_oe_.clear();
        ia_oe_.resize(ia_oe_num);
        ia_oe_offset_.clear();
        ia_oe_offset_.resize(inner_node_num+1);

        for(vid_t i = 1; i < inner_node_num; i++) {
            ia_oe_degree[i] += ia_oe_degree[i-1];
        }

        parallel_for(vid_t i = 0; i < inner_node_num; i++) {
        // for(vid_t i = inner_vertices.begin().GetValue(); i < inner_vertices.end().GetValue(); i++) {
            vertex_t u(i);
            vid_t index_a = ia_oe_degree[i];
            ia_oe_offset_[i] = &ia_oe_[index_a];
            if(this->Fc[u] != this->FC_default_value){
                auto ids_id = this->id2spids[u];
                const auto& oes = new_graph->GetOutgoingAdjList(u);
                for(auto& oe : oes){
                    if(this->id2spids[oe.neighbor] == ids_id){ // inner edge
                        ia_oe_[index_a] = oe;
                        index_a++;
                    }
                }
            }
            // CHECK_EQ(index_s, ia_oe_degree[i+1]);
        }
        ia_oe_offset_[inner_node_num] = &ia_oe_[ia_oe_num-1] + 1;
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

    /* use a VertexArray */
    // void build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, VertexArray<value_t, vid_t>& values,  VertexArray<delta_t, vid_t>& deltas){
    void build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        spnode.data = this->app_->GetIdentityElement();
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas */
        test_time[tid][0] -= GetCurrentTime();
        for(auto v : node_set){
            deltas[v] = this->app_->GetInitDelta(v, source);
            values[v] = this->app_->GetInitValue(v);
        }
        test_time[tid][0] += GetCurrentTime();
        /* iterative calculation */
        test_time[tid][1] -= GetCurrentTime();
        double b = GetCurrentTime();
        std::unordered_set<vertex_t> next_modified;
        next_modified.insert(node_set.begin(), node_set.end());
        run_to_convergence(tid, new_graph, node_set, next_modified, source);
        test_time[tid][3] = std::max(test_time[tid][3], GetCurrentTime()-b);
        test_time[tid][1] += GetCurrentTime();
        /* build new index in supernodes */
        test_time[tid][2] -= GetCurrentTime();
        fianl_build_trav_index(tid, new_graph, node_set, spnode);
        test_time[tid][2] += GetCurrentTime();
    }

    /* use VertexArray */
    void run_to_convergence(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, std::unordered_set<vertex_t>& next_modified_, vertex_t source){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        std::unordered_set<vertex_t> curr_modified_;
        int step = 0;
        const vid_t ids_id = this->id2spids[source];
        
        while (true) {
            step++;
            curr_modified_.clear();
            // receive & send
            for(auto v : next_modified_){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                auto& value = values[v];
                auto& to_send = deltas[v];
                if (this->app_->CombineValueDelta(value, to_send)) {
                    auto out_degree = oes.Size();
                    if(out_degree > 0){
                        /* Note: Because set is not thread-safe, parallelism cannot be used */
                        /*
                        auto it = oes.begin();
                        granular_for(j, 0, out_degree, (out_degree > 1024), {
                            auto& e = *(it + j);
                            // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                            if(ids_id == this->id2spids[e.neighbor]){  // Only sent to internal vertices
                                delta_t outv;
                                this->app_->Compute(v, value, to_send, oes, e, outv);
                                if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                                    curr_modified_.insert(e.neighbor);
                                }
                            }
                        }) 
                        */
                        for(auto e : oes){
                            // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                            if(ids_id == this->id2spids[e.neighbor]){  // Only sent to internal vertices
                                delta_t outv;
                                this->app_->Compute(v, value, to_send, oes, e, outv);
                                if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                                    curr_modified_.insert(e.neighbor);
                                }
                            }
                        }
                    }
                }
            }
            // check convergence
            if(curr_modified_.size() == 0 || step > 2000){
                if(step > 2000){
                    LOG(INFO) << "run_to_convergence: step>2000";
                }
                break;
            }
            next_modified_.swap(curr_modified_);
        }
    }

    /* use VertexArray */
    void fianl_build_trav_index(vid_t tid, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::unordered_map<vid_t, delta_t> bound_map;
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        // for(auto v : node_set){
        //     auto& value = values[v];
        //     auto& delta = deltas[v];
        //     if(value != this->app_->GetIdentityElement()){
        //         delta_t rt_delta;
        //         this->app_->revCompute(delta, rt_delta);
        //         spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
        //         const auto& oes = new_graph->GetOutgoingAdjList(v);
        //         delta_t outv;
        //         for(auto& e : oes){
        //             if(ids_id != this->id2spids[e.neighbor]){  // Ensure that e.neighbor is the boundary vertex
        //                 this->app_->Compute(v, value, delta, oes, e, outv);
        //                 if(bound_map.find(e.neighbor.GetValue()) == bound_map.end()){
        //                     bound_map[e.neighbor.GetValue()] = outv;
        //                 }
        //                 else{
        //                     this->app_->AccumulateDelta(bound_map[e.neighbor.GetValue()], outv);
        //                 }
        //             }
        //         }
        //     }
        // }
        /* 针对cc算法需要判断超点内部是否连通，如果不连通则不需要建立索引，而连通与否无法通过默认值判断 */
        if (FLAGS_application == "cc") {
            // 针对cc单独处理
            vid_t co_id = values[source];
            for(auto v : node_set){
                auto& value = values[v];
                auto& delta = deltas[v];
                if(delta.value == co_id){
                    delta_t rt_delta;
                    this->app_->revCompute(delta, rt_delta);
                    if(this->supernode_out_bound[v.GetValue()]){
                        spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                    }
                    else {
                        spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta)); 
                    }
                }
                else {
                    // LOG(INFO) << "not same com....";
                }
            }

            //debug
            {   
                if (this->id2spids[source] == this->id2spids[vertex_t(1)]) {
                    LOG(INFO) << "=----------build index--------------=";
                    LOG(INFO) << "source id=" << source.GetValue() << " oid=" << new_graph->GetId(source);
                    for (auto v : node_set) {
                        LOG(INFO) << "gid=" << new_graph->Vertex2Gid(v) << " oid=" << new_graph->GetId(v) << " value=" << values[v] << " delta=" << deltas[v].value;
                    }
                    // LOG(INFO) << "inner edge:";
                    // for (auto v : node_set) { 
                    //     const adj_list_t& inner_oes = adj_list_t(ia_oe_offset_[v.GetValue()], ia_oe_offset_[v.GetValue()+1]);
                    //     for(auto e : inner_oes){
                    //         LOG(INFO) << "vOid=" << new_graph->GetId(v) << "->" << new_graph->GetId(e.neighbor) << " =>>" <<  "gid=" << new_graph->Vertex2Gid(v) << "->" << new_graph->Vertex2Gid(e.neighbor);
                    //     }
                    // }
                }
            }
        }
        else {
            for(auto v : node_set){
                auto& value = values[v];
                auto& delta = deltas[v];
                if(value != this->app_->GetIdentityElement()){
                    delta_t rt_delta;
                    this->app_->revCompute(delta, rt_delta);
                    if(this->supernode_out_bound[v.GetValue()]){
                        spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                    }
                    else {
                        spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta)); 
                    }
                }
            }
            // this->print();
        }
    }

    void precompute_spnode(const std::shared_ptr<fragment_t>& new_graph){
        /* if the source vertex is within the supernode and isn't the entry point. */
        double inc_pre_compute = GetCurrentTime();
        if(FLAGS_application == "sssp"){
            vertex_t source;
            bool native_source = new_graph->GetInnerVertex(FLAGS_sssp_source, source);
            if(native_source && this->Fc[source] < 0){ // inner node, must include bound node
                // vid_t src_id = -this->Fc[source]-1;
                // std::vector<vertex_t>& src = this->supernode_source[src_id];
                // vid_t spid = this->Fc_map[src[0]];
                // run_to_convergence_for_precpt(spid, new_graph);
                vid_t spids = this->id2spids[source];
                run_to_convergence_for_precpt(spids, new_graph);
                LOG(INFO) << " sssp precompute_spnode ...";
            }
        } else if (FLAGS_application == "cc") {
            const vid_t spn_ids_num = this->supernode_ids.size();
            LOG(INFO) << "application cc spn_ids_num=" << spn_ids_num;
            #pragma cilk grainsize = 1
            parallel_for(vid_t i = 0; i < spn_ids_num; i++){
                run_to_convergence_for_precpt(i, new_graph);
            }
        } else {
            LOG(INFO) << "No this application.";
            exit(0);
        }
    }

    void precompute_spnode_all(const std::shared_ptr<fragment_t>& new_graph){
        const vid_t spn_ids_num = this->supernode_ids.size();
        LOG(INFO) << "precompute_spnode_all spn_ids_num=" << spn_ids_num;
        #pragma cilk grainsize = 1
        parallel_for(vid_t i = 0; i < spn_ids_num; i++){
            run_to_convergence_for_precpt(i, new_graph);
        }
    }

    void Output(const std::shared_ptr<fragment_t>& new_graph) {
        auto inner_vertices = new_graph->InnerVertices();
        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;
        LOG(INFO) << "--------------";
        for (auto v : inner_vertices) {
            LOG(INFO) << "gid=" << new_graph->Vertex2Gid(v) << " oid=" << new_graph->GetId(v) << " value=" << values[v] << " delta=" << deltas[v].value << std::endl;
        }
        LOG(INFO) << "++++++++++++++";
    }

    /**
     * spids: index of supernode_ids[]
    */
    void run_to_convergence_for_precpt(const vid_t spids, const std::shared_ptr<fragment_t>& new_graph){
        // supernode_t& spnode = this->supernodes[spid];
        // std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];
        // const vid_t ids_id = this->id2spids[spnode.id];

        std::vector<vertex_t> &node_set = this->supernode_ids[spids];
        const vid_t ids_id = spids;

        // std::unordered_map<vid_t, delta_t> send_delta;
        std::unordered_set<vertex_t> curr_modified_;
        std::unordered_set<vertex_t> next_modified_;
        int step = 0;
        next_modified_.insert(node_set.begin(), node_set.end());

        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;

        //debug
        // {   
        //     if (spids == this->id2spids[vertex_t(1)]) {
        //         LOG(INFO) << "=------------------------=";
        //         for (auto v : node_set) {
        //             LOG(INFO) << "gid=" << new_graph->Vertex2Gid(v) << " oid=" << new_graph->GetId(v) << " value=" << values[v] << " delta=" << deltas[v].value;
        //         }
        //         LOG(INFO) << "inner edge:";
        //         for (auto v : node_set) { 
        //             const adj_list_t& inner_oes = adj_list_t(ia_oe_offset_[v.GetValue()], ia_oe_offset_[v.GetValue()+1]);
        //             for(auto e : inner_oes){
        //                 LOG(INFO) << "vOid=" << new_graph->GetId(v) << "->" << new_graph->GetId(e.neighbor) << " =>>" <<  "gid=" << new_graph->Vertex2Gid(v) << "->" << new_graph->Vertex2Gid(e.neighbor);
        //             }
        //         }
        //     }
        // }

        // Output(new_graph);
        while (true) {
            step++;
            curr_modified_.clear();
            // receive & send
            for(auto v : next_modified_){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                const adj_list_t& inner_oes = adj_list_t(ia_oe_offset_[v.GetValue()], ia_oe_offset_[v.GetValue()+1]);
                auto& value = values[v];
                auto& to_send = deltas[v];

                if (this->app_->CombineValueDelta(value, to_send)) {
                    // auto out_degree = oes.Size();
                    auto out_degree = inner_oes.Size();
                    if(out_degree > 0){
                        // auto it = oes.begin();
                        // granular_for(j, 0, out_degree, (out_degree > 1024), {
                        //     auto& e = *(it + j);
                        //     delta_t outv;
                        //     this->app_->Compute(v, value, to_send, oes, e, outv);
                        //     bool is_update = this->app_->AccumulateDelta(deltas[e.neighbor], outv);
                        //     // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                        //     if(is_update && ids_id == this->id2spids[e.neighbor]){  // Only sent to internal vertices)
                        //         curr_modified_.insert(e.neighbor);
                        //     }
                        // })

                        // for(auto e : oes){
                        for(auto e : inner_oes){ // Only sent to internal points
                            if (ids_id == this->id2spids[e.neighbor]) {
                                delta_t outv;
                                this->app_->Compute(v, value, to_send, oes, e, outv);
                                bool is_update = this->app_->AccumulateDelta(deltas[e.neighbor], outv);
                                // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                                if(is_update){  // Only sent to internal vertices)
                                // 上面的可以删除判断内部点，因为已经使用了内部边！
                                    curr_modified_.insert(e.neighbor);
                                }
                            }
                        }
                    }
                }
            }
            // check convergence
            if(curr_modified_.size() == 0 || step > 2000){
                if(step > 2000){
                    LOG(INFO) << "run_to_convergence: step>2000";
                }
                break;
            }
            next_modified_.swap(curr_modified_);
        }

        //debug
        // {   
        //     Output(new_graph);
        //     LOG(INFO) << "-----------spids=" << spids;
        //     if (spids == this->id2spids[vertex_t(10)]) {
        //         LOG(INFO) << "=----------pre compute--s------------=";
        //         for (auto v : node_set) {
        //             LOG(INFO) << "gid=" << new_graph->Vertex2Gid(v) << " oid=" << new_graph->GetId(v) << " value=" << values[v] << " delta=" << deltas[v].value;
        //         }
        //         LOG(INFO) << "=----------pre compute--e------------=";
        //         // LOG(INFO) << "inner edge:";
        //         // for (auto v : node_set) { 
        //         //     const adj_list_t& inner_oes = adj_list_t(ia_oe_offset_[v.GetValue()], ia_oe_offset_[v.GetValue()+1]);
        //         //     for(auto e : inner_oes){
        //         //         LOG(INFO) << "vOid=" << new_graph->GetId(v) << "->" << new_graph->GetId(e.neighbor) << " =>>" <<  "gid=" << new_graph->Vertex2Gid(v) << "->" << new_graph->Vertex2Gid(e.neighbor);
        //         //     }
        //         // }
        //     }
        // }
    }

    void inc_run(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges, const std::shared_ptr<fragment_t>& new_graph){
        /* Switch data to support add new nodes */
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

        // init_deltas.Init(new_graph->Vertices()); // note: include out vertex
        // for (auto v : new_graph->Vertices()) {
        //     init_deltas[v] = this->app_->GetInitDelta(v);
        // }

        /* find supernode */
        timer_next("inc compress");
        double inc_compress = GetCurrentTime();
        this->inc_trav_compress(deleted_edges, added_edges);
        inc_compress = GetCurrentTime()-inc_compress;
        LOG(INFO) << "#inc_compress: " << inc_compress;
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish inc compress...";

        timer_next("init bound_ids");
        /* init supernode_out_bound*/
        const vid_t spn_ids_num = this->supernode_ids.size();
        double begin = GetCurrentTime();
        this->supernode_out_bound.clear();
        this->supernode_out_bound.resize(this->graph_->GetVerticesNum(), 0);
        if (trav_compressor_flags_cilk) {
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
        // this->write_spnodes("../Dataset/inc_spnodes_incindex" + std::to_string(this->comm_spec_.worker_id()));
    }

    void inc_trav_compress(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges){
        fid_t fid = this->graph_->fid();
        auto vm_ptr = this->graph_->vm_ptr();
        this->inccalculate_spnode_ids.clear();
        this->recalculate_spnode_ids.clear();
        reset_edges.clear();
        LOG(INFO) << "spnode_num=" << this->supernodes_num;
        LOG(INFO) << "deal deleted_edges...";
        for(auto& pair : deleted_edges) {
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(u_fid == fid && this->Fc[u] != this->FC_default_value){
                reset_edges.template emplace_back(u.GetValue(), v.GetValue());
            }
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                vid_t src_id = this->Fc[v] < 0 ? (-this->Fc[v]-1) : this->Fc[v];
                std::vector<vertex_t>& src = this->supernode_source[src_id];
                vid_t del_id = this->Fc_map[src[0]];
                supernode_t& spnode = this->supernodes[del_id];
                // const vid_t ids_id = this->id2spids[spnode.id]; // 2022-2-9
                const vid_t ids_id = this->id2spids[v];
                if(ids_id != this->id2spids[u] && src.size() > 1){
                    CHECK(this->Fc[v] >= 0);
                    const auto& ies = this->graph_->GetIncomingAdjList(v);
                    bool hava_out_inadj = false;
                    for(auto& e : ies){
                        auto& nb = e.neighbor;
                        // if(nb != u && std::find(spids.begin(), spids.end(), nb) == spids.end()){
                        if(nb != u && ids_id != this->id2spids[nb]){
                            hava_out_inadj = true;
                            break;
                        }
                    }
                    if(hava_out_inadj == false){
                        this->delete_supernode(v);
                    }
                }
            }
        }
        LOG(INFO) << "deal added_edges...";
        for(auto& pair : added_edges){
            auto u_gid = pair.first;
            auto v_gid = pair.second;
            fid_t u_fid = vm_ptr->GetFidFromGid(u_gid),
                  v_fid = vm_ptr->GetFidFromGid(v_gid);
            // u -> v
            // LOG(INFO) << u_gid << "->" << v_gid;
            vertex_t u;
            CHECK(this->graph_->Gid2Vertex(u_gid, u));
            if(u_fid == fid && this->Fc[u]!= this->FC_default_value){
                // LOG(INFO) << graph_->GetId(u);
                vid_t src_id = this->Fc[u] < 0 ? (-this->Fc[u]-1) : this->Fc[u];
                for(auto source : this->supernode_source[src_id]){
                    this->inccalculate_spnode_ids.insert(source.GetValue());
                }
            }
            vertex_t v;
            CHECK(this->graph_->Gid2Vertex(v_gid, v));
            if(v_fid == fid && this->Fc[v] != this->FC_default_value){
                vid_t src_id = this->Fc[v] < 0 ? (-this->Fc[v]-1) : this->Fc[v];
                std::vector<vertex_t>& src = this->supernode_source[src_id];
                supernode_t& spnode = this->supernodes[this->Fc_map[src[0]]];
                auto& spids = this->supernode_ids[spnode.ids];
                const vid_t ids_id = this->id2spids[spnode.id];
                if(this->Fc[v] < 0 && ids_id != this->id2spids[u]){ // not a source, build a new spnode
                    this->Fc[v] = src_id;
                    this->supernode_source[src_id].emplace_back(v);
                    // build a new spnode idnex
                    vid_t supernoed_id = this->supernodes_num;
                    this->Fc_map[v] = supernoed_id;
                    this->supernodes[supernoed_id].id = v;
                    // supernodes[supernoed_id].ids.insert(supernodes[supernoed_id].ids.begin(), spnode.ids.begin(), spnode.ids.end());
                    this->supernodes[supernoed_id].ids = spnode.ids;
                    this->supernodes_num++;
                    this->recalculate_spnode_ids.insert(v.GetValue());
                }
            }
        }
        // for(auto u : inccalculate_spnode_ids){
        //     LOG(INFO) << "inccalculate_spnode_ids:" << u << ":" << graph_->GetId(vertex_t(u));
        // }
        LOG(INFO) << "spnode_num=" << this->supernodes_num << " inccalculate_spnode_ids.size=" << this->inccalculate_spnode_ids.size() << " recalculate_spnode_ids.size=" << this->recalculate_spnode_ids.size() << " %=" << ((this->inccalculate_spnode_ids.size()+this->recalculate_spnode_ids.size())*1.0/this->supernodes_num);
        LOG(INFO) << "reset_edges.size=" << reset_edges.size();
    }


    void inc_compute_index(const std::shared_ptr<fragment_t>& new_graph){
        /* case 1: Reset the index value according to the reset_edges */
        reset_spnode_edges.resize(this->supernodes_num);
        had_reset.clear();
        had_reset.resize(this->supernodes_num, 0);
        if_touch.clear();
        if_touch.resize(this->supernodes_num, 0);
        for(auto pair : reset_edges){
            auto u_id = pair.first;
            auto v_id = pair.second;
            vertex_t u(u_id);
            vid_t src_id = this->Fc[u] < 0 ? (-this->Fc[u]-1) : this->Fc[u];
            auto& srcs = this->supernode_source[src_id];
            for(auto source: srcs){
                reset_spnode_edges[this->Fc_map[source]].push_back(std::pair<vid_t, vid_t>(u_id, v_id));
                // inccalculate_spnode_ids 和 reset_spnode_edges的交集
                // if(this->inccalculate_spnode_ids.find(source.GetValue()) != this->inccalculate_spnode_ids.end()){
                //     this->inccalculate_spnode_ids.erase(source.GetValue());
                // }
            }
        }
        {
            std::vector<vid_t> ids;
            vid_t i = 0;
            for(auto e : reset_spnode_edges){
                if(int(e.size()) > 0){
                    ids.emplace_back(i);
                }
                i++;
            }
            int len = ids.size();
            if (trav_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
                parallel_for(vid_t i = 0; i < len; i++){
                    int tid = __cilkrts_get_worker_number();
                    reset_inc_build_tarv_index(ids[i], new_graph, tid); // iterative: pr, php
                    if(i % 1000000 == 0){
                        LOG(INFO) << "----id=" << i << " computing index" << std::endl;
                    }
                }
            }
            else{
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                    for(vid_t i = begin; i < end; i++){
                        // if_touch[ids[i]] = true;  //测试
                        // build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                        reset_inc_build_tarv_index(ids[i], new_graph, tid);
                        // build_trav_index(ids[i], new_graph, tid); // 仅仅测时使用！！！！
                    }
                    // LOG(INFO) << "tid=" << tid << " finish reset_inc_build_tarv_index!";
                    }, FLAGS_build_index_concurrency
                );
                MPI_Barrier(this->comm_spec_.comm()); // 是否需要？？？
            }
            LOG(INFO) << "finish reset_inc_build_tarv_index... len=" << len;
        }

        /* case 2: inc-recalculate the index value according to the inccalculate_spnode_ids */
        {
            std::vector<vid_t> ids;
            ids.insert(ids.begin(), this->inccalculate_spnode_ids.begin(), this->inccalculate_spnode_ids.end());
            int len = ids.size();
            if (trav_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
                parallel_for(vid_t i = 0; i < len; i++){
                    vertex_t u(ids[i]);
                    int tid = __cilkrts_get_worker_number();
                    // Note: It needs to be judged here, because the index of u as the entry may have been deleted.
                    if(this->Fc_map[u] != this->ID_default_value){
                        inc_build_trav_index(this->Fc_map[u], new_graph, tid);
                    }
                }
            }
            else{
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                    for(vid_t i = begin; i < end; i++){
                        vertex_t u(ids[i]);
                        // build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                        if(this->Fc_map[u] != this->ID_default_value && !had_reset[this->Fc_map[u]]){
                            // if_touch[this->Fc_map[u]] = true; // 测试
                            inc_build_trav_index(this->Fc_map[u], new_graph, tid);
                            // reset_inc_build_tarv_index(this->Fc_map[u], new_graph, tid); // 仅仅测时用，全部重算！！！
                            // build_trav_index(this->Fc_map[u], new_graph, tid); // 仅仅测时用，全部重算！！！
                        }
                    }
                    // LOG(INFO) << "tid=" << tid << " finish inc_build_trav_index!";
                    }, FLAGS_build_index_concurrency
                );
                MPI_Barrier(this->comm_spec_.comm());
            }
            LOG(INFO) << "finish inc_build_trav_index... len=" << len;
        }
        
        /* case 3: recalculate the newly created index according to the recalculate_spnode_ids */
        {
            std::vector<vid_t> ids;
            ids.insert(ids.begin(), this->recalculate_spnode_ids.begin(), this->recalculate_spnode_ids.end());
            int len = ids.size();
            if (trav_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
                parallel_for(vid_t i = 0; i < len; i++){
                    vertex_t u(ids[i]);
                    if(this->Fc_map[u] != this->ID_default_value){
                        int tid = __cilkrts_get_worker_number();
                        build_trav_index(this->Fc_map[u], new_graph, tid);
                    }
                    if(i % 1000000 == 0){
                        LOG(INFO) << "----id=" << i << " computing index" << std::endl;
                    }
                }
            }
            else{
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                    for(vid_t i = begin; i < end; i++){
                        vertex_t u(ids[i]);
                        // build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                        if(this->Fc_map[u] != this->ID_default_value){
                            // if_touch[this->Fc_map[u]] = true; // 测试
                            build_trav_index(this->Fc_map[u], new_graph, tid);
                        }
                    }
                    // LOG(INFO) << "tid=" << tid << " finish inc_build_trav_index!";
                    }, FLAGS_build_index_concurrency
                );
                MPI_Barrier(this->comm_spec_.comm());
            }
            LOG(INFO) << "finish recalculate build_trav_index... len=" << len;
        }
    }

    /* use a VertexArray */
    void reset_inc_build_tarv_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            // values[v] = this->app_->GetIdentityElement();
            // deltas[v].Reset(this->app_->GetIdentityElement());
            values[v] = this->app_->GetInitValue(v);
            deltas[v] = this->app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
            this->app_->CombineValueDelta(values[v], deltas[v]);
        }
        // 对于mode2这里需要对inner_delta和bound_delta都要处理
        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        
        /* reset by dependency */
        std::unordered_set<vertex_t> curr_modified, next_modified;
        for (auto& pair : reset_spnode_edges[spid]) {
            vid_t u_id = pair.first, v_id = pair.second;

            vertex_t u(u_id), v(v_id);
            auto parent_gid = deltas[v].parent_gid;

            if (parent_gid == this->graph_->Vertex2Gid(u)) {
                curr_modified.insert(v);
            }
        }
        if(curr_modified.size() == 0){ // bound_delta maybe have been chaged, so, can't return.
            // reset_spnode_edges[spid].clear(); // Corresponds to the conditions of stage 2
            /* build new bound_delta index in supernodes */
            // fianl_build_trav_index_bound(tid, new_graph, node_set, spnode); // 是否需要？？
            return; // 考虑有没有问题，会不会有新的出口点啥的？
        }
        while (curr_modified.size() > 0){
            for(auto u : curr_modified){
                auto u_gid = this->graph_->Vertex2Gid(u);
                auto oes = this->graph_->GetOutgoingAdjList(u);

                for (auto e : oes) {
                    auto v = e.neighbor;
                    auto parent_gid = deltas[v].parent_gid;
                    if (parent_gid == u_gid) {
                        next_modified.insert(v);
                    }
                }

                // values[u] = this->app_->GetIdentityElement();
                // deltas[u].Reset(this->app_->GetIdentityElement());
                values[u] = this->app_->GetInitValue(u);
                deltas[u] = this->app_->GetInitDelta(u); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
                this->app_->CombineValueDelta(values[u], deltas[u]);
            }

            curr_modified.clear();
            curr_modified.swap(next_modified);
        }

        // Start a round without any condition on new_graph
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            auto& delta = deltas[v];

            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                // auto it = oes.begin();
                // granular_for(j, 0, out_degree, (out_degree > 1024), {
                //     auto& e = *(it + j);
                //     // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                //     if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                //         delta_t outv;
                //         this->app_->Compute(v, value, delta, oes, e, outv);
                //         if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                //             next_modified.insert(e.neighbor);
                //         }
                //     }
                // })
                for(auto e : oes){
                    // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                } 
            }
        }
        
        had_reset[spid] = 1; // Corresponds to the conditions of stage 2
        if_touch[spid] = 1;
        /* iterative calculation */
        run_to_convergence(tid, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(tid, new_graph, node_set, spnode);
    }

    /* use a VertexArray */
    void inc_build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            // values[v] = this->app_->GetIdentityElement();
            // deltas[v].Reset(this->app_->GetIdentityElement());
            values[v] = this->app_->GetInitValue(v);
            deltas[v] = this->app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
            this->app_->CombineValueDelta(values[v], deltas[v]);
        }

        // 对于mode2这里需要对inner_delta和bound_delta都要处理
        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }

        // Start a round without any condition on new_graph
        std::unordered_set<vertex_t> next_modified;
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            auto& delta = deltas[v];

            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                // auto it = oes.begin();
                // granular_for(j, 0, out_degree, (out_degree > 1024), {
                //     auto& e = *(it + j);
                //     // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                //     if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                //         delta_t outv;
                //         this->app_->Compute(v, value, delta, oes, e, outv);
                //         if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                //             next_modified.insert(e.neighbor);
                //         }
                //     }
                // })
                for(auto e : oes){
                    // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                }
            }
        }

        /* iterative calculation */
        // 注意：此时内部变了，所以需要重新建立索引！！！之前是分了两种情况，可以只更新外部索引和全部更新索引！
        // if(next_modified.size() > 0){ // 注意：看看是否能改成直接判空？？？？
            if_touch[spid] = 1;
            run_to_convergence(tid, new_graph, node_set, next_modified, source);
            /* build new index in supernodes */
            fianl_build_trav_index(tid, new_graph, node_set, spnode);
        // }
        // else{
        // 可以分为建立所有索引和边界索引, 在mode2的情况下，没有这种需求了！！！！
            // fianl_build_trav_index_bound(tid, new_graph, node_set, spnode); // 是否需要？
        // }
    }

    /* use a VertexArray */
    void inc_build_trav_index2(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, vid_t tid){
        VertexArray<value_t, vid_t>& values = values_array[tid];
        VertexArray<delta_t, vid_t>& deltas = deltas_array[tid];
        supernode_t& spnode = this->supernodes[spid];
        // spnode.data = this->app_->GetIdentityElement(); // ??
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            // values[v] = this->app_->GetIdentityElement();
            // deltas[v].Reset(this->app_->GetIdentityElement()); // 应该用初始化函数！

            values[v] = this->app_->GetInitValue(v);
            deltas[v] = this->app_->GetInitDelta(v); // When resetting value/delta, the CC should be set to gid instead of the default value, and delta should be applied to the value at the same time.
            this->app_->CombineValueDelta(values[v], deltas[v]);
        }
        // 对于mode2这里需要对inner_delta和bound_delta都要处理
        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }
        for(auto e : spnode.bound_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
            deltas[e.first] = e.second;
        }

        // // Start a round without any condition on new_graph
        std::unordered_set<vertex_t> next_modified;
        //debug
        // {
        //     next_modified.insert(node_set.begin(), node_set.end());
        // }

        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v];
            auto& delta = deltas[v];
            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                // auto it = oes.begin();
                // granular_for(j, 0, out_degree, (out_degree > 1024), {
                //     auto& e = *(it + j);
                //     // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                //     if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                //         delta_t outv;
                //         this->app_->Compute(v, value, delta, oes, e, outv);
                //         if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                //             next_modified.insert(e.neighbor);
                //         }
                //     }
                // })
                for(auto e : oes){
                    // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                }
            }
        }

        /* iterative calculation */
        // 注意：此时内部变了，所以需要重新建立索引！！！之前是分了两种情况，可以只更新外部索引和全部更新索引！
        // if(next_modified.size() > 0){ // 注意：看看是否能改成直接判空？？？？
            if_touch[spid] = 1;
            run_to_convergence(tid, new_graph, node_set, next_modified, source);
            /* build new index in supernodes */
            fianl_build_trav_index(tid, new_graph, node_set, spnode);
        // }
        // else{
        // 可以分为建立所有索引和边界索引, 在mode2的情况下，没有这种需求了！！！！
            // fianl_build_trav_index_bound(tid, new_graph, node_set, spnode); // 是否需要？
        // }
    }

public:
    std::vector<std::pair<vid_t, vid_t>> reset_edges;
    std::vector<std::vector<std::pair<vid_t, vid_t>> > reset_spnode_edges;
    std::vector<short int> had_reset; // Mark whether the super point is reset and calculated
    std::vector<short int> if_touch; // Mark whether to update the inside of the superpoint
    std::vector<VertexArray<value_t, vid_t>> values_array; // use to calulate indexes in parallel
    std::vector<VertexArray<delta_t, vid_t>> deltas_array;

    // VertexArray<delta_t, vid_t> init_deltas;
    std::vector<std::vector<double>> test_time; // test time
    /* inner all nodes */
    Array<nbr_t, Allocator<nbr_t>> ia_oe_;
    Array<nbr_t*, Allocator<nbr_t*>> ia_oe_offset_;
    /* in_bound_node to out_bound_node */
    // Array<nbr_t, Allocator<nbr_t>> ib_oe_;
    // Array<nbr_t*, Allocator<nbr_t*>> ib_oe_offset_;
};

}  // namespace grape
#endif  // GRAPE_FRAGMENT_TRAV_COMPRESSOR_H_