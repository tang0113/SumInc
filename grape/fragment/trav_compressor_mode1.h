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
    size_t total_node_set = 0;
    bool trav_compressor_flags_cilk = false;

    TravCompressor(std::shared_ptr<APP_T>& app,
                        std::shared_ptr<fragment_t>& graph):CompressorBase<APP_T, SUPERNODE_T>(app, graph){}

    void run(){
        /* find supernode */
        timer_next("find supernode");
        this->compress();
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish compress...";
        // MPI_Barrier(this->comm_spec_.comm()); // debug

        /* calculate index for each structure */
        timer_next("calculate index");
        if (trav_compressor_flags_cilk) {
#ifdef INTERNAL_PARALLEL
          LOG(FATAL) << "Ingress is not compiled with -DUSE_CILK";
#endif
            parallel_for(vid_t i = 0; i < this->supernodes_num; i++){
                build_trav_index(i, this->graph_);
            }
        }
        else{
            /*
// #pragma omp parallel for num_threads(NUM_THREADS) //schedule(dynamic,1)
            for(vid_t i = 0; i < this->supernodes_num; i++){
                build_trav_index(i, this->graph_);
            }
// #pragma omp barrier
            */
           /* parallel */
           int thread_num = FLAGS_build_index_concurrency;
           values_array.resize(thread_num);
           deltas_array.resize(thread_num);
           this->ForEachIndex(this->supernodes_num, [this](int tid, vid_t begin, vid_t end) {
                LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                auto inner_vertices = this->graph_->InnerVertices();
                values_array[tid].Init(inner_vertices);
                deltas_array[tid].Init(inner_vertices);
                for(vid_t i = begin; i < end; i++){
                    build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                }
                LOG(INFO) << "tid=" << tid << " finish!";
                }, thread_num
            );
        }
        /* init supernode.data */
        if_touch.resize(this->supernodes_num);
        #pragma omp parallel for num_threads(this->thread_num())
        for(int i = 0; i < this->supernodes_num; i++){
            if_touch[i] = 1;
            this->supernodes[i].data = this->app_->GetIdentityElement();
        }
        MPI_Barrier(this->comm_spec_.comm());
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish calculate index...";

        /* precompute supernode */
        timer_next("pre compute");
        precompute_spnode(this->graph_);
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish precompute supernode...";

        MPI_Barrier(this->comm_spec_.comm());

        /* debug */
        // this->print();
        // timer_next("write spnodes");
        // this->write_spnodes("../Dataset/spnodes" + std::to_string(this->comm_spec_.worker_id()));
    }

    /* use a VertexArray */
    void build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, VertexArray<value_t, vid_t>& values,  VertexArray<delta_t, vid_t>& deltas){
        supernode_t& spnode = this->supernodes[spid];
        spnode.data = this->app_->GetIdentityElement();
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* init values/deltas */
        for(auto v : node_set){
            deltas[v] = this->app_->GetInitDelta(v, source);
            values[v] = this->app_->GetInitValue(v);
        }
        /* iterative calculation */
        std::unordered_set<vertex_t> next_modified;
        next_modified.insert(node_set.begin(), node_set.end());
        run_to_convergence(values, deltas, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(values, deltas, new_graph, node_set, spnode);
    }

    /* use local variable map */
    void build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph){
        supernode_t& spnode = this->supernodes[spid];
        spnode.data = this->app_->GetIdentityElement();
        const auto& source = spnode.id;
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];
        std::unordered_map<vid_t, delta_t> deltas;
        std::unordered_map<vid_t, value_t> values;

        /* init values/deltas */
        for(auto v : node_set){
            deltas[v.GetValue()] = this->app_->GetInitDelta(v, source);
            values[v.GetValue()] = this->app_->GetInitValue(v);
        }
        /* iterative calculation */
        std::unordered_set<vertex_t> next_modified;
        next_modified.insert(node_set.begin(), node_set.end());
        run_to_convergence(values, deltas, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(values, deltas, new_graph, node_set, spnode);
    }

    void run_to_convergence(std::unordered_map<vid_t, value_t>& values,  std::unordered_map<vid_t, delta_t>& deltas, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, std::unordered_set<vertex_t>& next_modified_, vertex_t source){
        std::unordered_set<vertex_t> curr_modified_;
        int step = 0;
        const vid_t ids_id = this->id2spids[source];
        
        while (true) {
            step++;
            // receive
            curr_modified_.clear();
            for(auto v : next_modified_){
                auto& value = values[v.GetValue()];
                auto& delta = deltas[v.GetValue()];

                if (this->app_->CombineValueDelta(value, delta)) {
                  curr_modified_.insert(v);
                }
            }
            next_modified_.clear();
            // send
            for(auto v : curr_modified_){
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                auto& value = values[v.GetValue()];
                auto& to_send = deltas[v.GetValue()];

                auto out_degree = oes.Size();
                if(out_degree > 0){
                    auto it = oes.begin();
                    granular_for(j, 0, out_degree, (out_degree > 1024), {
                        auto& e = *(it + j);
                        // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                        if(ids_id == this->id2spids[e.neighbor]){  // Only sent to internal vertices){  // Ensure that e.neighbor is the boundary vertex
                            delta_t outv;
                            this->app_->Compute(v, value, to_send, oes, e, outv);
                            if(this->app_->AccumulateDelta(deltas[e.neighbor.GetValue()], outv)){
                                next_modified_.insert(e.neighbor);
                            }
                        }
                    })
                    for(auto e : oes){
                        if(ids_id == this->id2spids[e.neighbor]){  // Only sent to internal vertices){  // Ensure that e.neighbor is the boundary vertex
                            delta_t outv;
                            this->app_->Compute(v, value, to_send, oes, e, outv);
                            if(this->app_->AccumulateDelta(deltas[e.neighbor.GetValue()], outv)){
                                next_modified_.insert(e.neighbor);
                            }
                        }
                    }
                }
            }
            // check convergence
            if(next_modified_.size() == 0 || step > 2000){
                if(step > 2000){
                    LOG(INFO) << "run_to_convergence: step>2000";
                }
                break;
            }
        }
    }

    /* use VertexArray */
    void run_to_convergence(VertexArray<value_t, vid_t>& values,  VertexArray<delta_t, vid_t>& deltas, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, std::unordered_set<vertex_t>& next_modified_, vertex_t source){
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

    void fianl_build_trav_index(std::unordered_map<vid_t, value_t>& values,  std::unordered_map<vid_t, delta_t>& deltas, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::unordered_map<vid_t, delta_t> bound_map;
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        for(auto v : node_set){
            auto& value = values[v.GetValue()];
            auto& delta = deltas[v.GetValue()];
            if(value != this->app_->GetIdentityElement()){
                delta_t rt_delta;
                this->app_->revCompute(delta, rt_delta);
                spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                delta_t outv;
                for(auto& e : oes){
                    if(ids_id != this->id2spids[e.neighbor]){  // Ensure that e.neighbor is the boundary vertex
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(bound_map.find(e.neighbor.GetValue()) == bound_map.end()){
                            bound_map[e.neighbor.GetValue()] = outv;
                        }
                        else{
                            this->app_->AccumulateDelta(bound_map[e.neighbor.GetValue()], outv);
                        }
                    }
                }
            }
        }
        for(auto kv : bound_map){
            delta_t rt_value;
            this->app_->revCompute(kv.second, rt_value);
            vertex_t u(kv.first);
            spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(u, rt_value)); 
        }
    }

    /* use VertexArray */
    void fianl_build_trav_index(VertexArray<value_t, vid_t>& values,  VertexArray<delta_t, vid_t>& deltas, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::unordered_map<vid_t, delta_t> bound_map;
        spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        for(auto v : node_set){
            auto& value = values[v];
            auto& delta = deltas[v];
            if(value != this->app_->GetIdentityElement()){
                delta_t rt_delta;
                this->app_->revCompute(delta, rt_delta);
                spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                delta_t outv;
                for(auto& e : oes){
                    if(ids_id != this->id2spids[e.neighbor]){  // Ensure that e.neighbor is the boundary vertex
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(bound_map.find(e.neighbor.GetValue()) == bound_map.end()){
                            bound_map[e.neighbor.GetValue()] = outv;
                        }
                        else{
                            this->app_->AccumulateDelta(bound_map[e.neighbor.GetValue()], outv);
                        }
                    }
                }
            }
        }
        for(auto kv : bound_map){
            delta_t rt_value;
            this->app_->revCompute(kv.second, rt_value);
            vertex_t u(kv.first);
            spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(u, rt_value)); 
        }
    }

    /* just build bound_delta */
    void fianl_build_trav_index_bound(std::unordered_map<vid_t, value_t>& values,  std::unordered_map<vid_t, delta_t>& deltas, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::unordered_map<vid_t, delta_t> bound_map;
        // spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        for(auto v : node_set){
            auto& value = values[v.GetValue()];
            auto& delta = deltas[v.GetValue()];
            if(value != this->app_->GetIdentityElement()){
                // delta_t rt_delta;
                // this->app_->revCompute(delta, rt_delta);
                // spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                delta_t outv;
                for(auto& e : oes){
                    if(ids_id != this->id2spids[e.neighbor]){  // Ensure that e.neighbor is the boundary vertex
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(bound_map.find(e.neighbor.GetValue()) == bound_map.end()){
                            bound_map[e.neighbor.GetValue()] = outv;
                        }
                        else{
                            this->app_->AccumulateDelta(bound_map[e.neighbor.GetValue()], outv);
                        }
                    }
                }
            }
        }
        for(auto kv : bound_map){
            delta_t rt_value;
            this->app_->revCompute(kv.second, rt_value);
            vertex_t u(kv.first);
            spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(u, rt_value)); 
        }
    }

    /* just build bound_delta, use VertexArray */
    void fianl_build_trav_index_bound(VertexArray<value_t, vid_t>& values,  VertexArray<delta_t, vid_t>& deltas, const std::shared_ptr<fragment_t>& new_graph, const std::vector<vertex_t> &node_set, supernode_t& spnode){
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::unordered_map<vid_t, delta_t> bound_map;
        // spnode.inner_delta.clear();
        spnode.bound_delta.clear();
        for(auto v : node_set){
            auto& value = values[v];
            auto& delta = deltas[v];
            if(value != this->app_->GetIdentityElement()){
                // delta_t rt_delta;
                // this->app_->revCompute(delta, rt_delta);
                // spnode.inner_delta.emplace_back(std::pair<vertex_t, delta_t>(v, rt_delta));
                const auto& oes = new_graph->GetOutgoingAdjList(v);
                delta_t outv;
                for(auto& e : oes){
                    if(ids_id != this->id2spids[e.neighbor]){  // Ensure that e.neighbor is the boundary vertex
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(bound_map.find(e.neighbor.GetValue()) == bound_map.end()){
                            bound_map[e.neighbor.GetValue()] = outv;
                        }
                        else{
                            this->app_->AccumulateDelta(bound_map[e.neighbor.GetValue()], outv);
                        }
                    }
                }
            }
        }
        for(auto kv : bound_map){
            delta_t rt_value;
            this->app_->revCompute(kv.second, rt_value);
            vertex_t u(kv.first);
            spnode.bound_delta.emplace_back(std::pair<vertex_t, delta_t>(u, rt_value)); 
        }
    }


    void precompute_spnode(const std::shared_ptr<fragment_t>& new_graph){
        /* if the source vertex is within the supernode and isn't the entry point. */
        if(FLAGS_application == "sssp"){
            vertex_t source;
            bool native_source = new_graph->GetInnerVertex(FLAGS_sssp_source, source);
            if(native_source && this->Fc[source] < 0){
                vid_t src_id = -this->Fc[source]-1;
                std::vector<vertex_t>& src = this->supernode_source[src_id];
                vid_t spid = this->Fc_map[src[0]];
                run_to_convergence_for_precpt(spid, new_graph);
                LOG(INFO) << " sssp precompute_spnode ...";
            }
        }
    }

    void run_to_convergence_for_precpt(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph){
        supernode_t& spnode = this->supernodes[spid];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];
        const vid_t ids_id = this->id2spids[spnode.id];
        // std::unordered_map<vid_t, delta_t> send_delta;
        std::unordered_set<vertex_t> curr_modified_;
        std::unordered_set<vertex_t> next_modified_;
        int step = 0;
        next_modified_.insert(node_set.begin(), node_set.end());

        auto& values = this->app_->values_;
        auto& deltas = this->app_->deltas_;
        
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
                        for(auto e : oes){
                            delta_t outv;
                            this->app_->Compute(v, value, to_send, oes, e, outv);
                            bool is_update = this->app_->AccumulateDelta(deltas[e.neighbor], outv);
                            // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                            if(is_update && ids_id == this->id2spids[e.neighbor]){  // Only sent to internal vertices)
                                curr_modified_.insert(e.neighbor);
                            }
                        }
                    }
                }
            }
            // check convergence
            if(next_modified_.size() == 0 || step > 2000){
                if(step > 2000){
                    LOG(INFO) << "run_to_convergence: step>2000";
                }
                break;
            }
            next_modified_.swap(curr_modified_);
        }
    }

    void inc_run(std::vector<std::pair<vid_t, vid_t>>& deleted_edges, std::vector<std::pair<vid_t, vid_t>>& added_edges, const std::shared_ptr<fragment_t>& new_graph){
        /* Switch data */
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
        for(auto v : old_inner_vertices){
            old_Fc_map[v] = this->Fc_map[v];
        }
        this->Fc_map.Init(new_graph->InnerVertices(), this->ID_default_value);
        // copy to new graph
        for(auto v : new_graph->InnerVertices()) {
            this->Fc_map[v] = old_Fc_map[v];
        }

        VertexArray<vid_t, vid_t> old_id2spids;
        old_id2spids.Init(this->graph_->Vertices());
        for(auto v : old_inner_vertices){
            old_id2spids[v] = this->id2spids[v];
        }
        this->id2spids.Init(new_graph->Vertices(), this->ID_default_value);
        // copy to new graph
        for(auto v : new_graph->InnerVertices()) {
            this->id2spids[v] = old_id2spids[v];
        }

        /* find supernode */
        timer_next("inc compress");  
        this->inc_trav_compress(deleted_edges, added_edges);
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish inc compress...";

        /* calculate index for each structure */
        timer_next("inc calculate index");
        inc_compute_index(new_graph);
        LOG(INFO) << "work_id=" << this->comm_spec_.worker_id() << " finish inc calculate index...";

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
                const vid_t ids_id = this->id2spids[spnode.id];
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
                    reset_inc_build_tarv_index(ids[i], new_graph); // iterative: pr, php
                    if(i % 1000000 == 0){
                        LOG(INFO) << "----id=" << i << " computing index" << std::endl;
                    }
                }
            }
            else{
// #pragma omp parallel for num_threads(NUM_THREADS) //schedule(dynamic,1)
//                 for(vid_t i = 0; i < len; i++){
//                     reset_inc_build_tarv_index(ids[i], new_graph);
//                     if(i % 1000000 == 0){
//                         LOG(INFO) << "----id=" << i << " computing index" << std::endl;
//                     }
//                 }
// #pragma omp barrier
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                    for(vid_t i = begin; i < end; i++){
                        // if_touch[ids[i]] = true;  //测试
                        // build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                        reset_inc_build_tarv_index(ids[i], new_graph, values_array[tid], deltas_array[tid]);
                    }
                    // LOG(INFO) << "tid=" << tid << " finish reset_inc_build_tarv_index!";
                    }, FLAGS_build_index_concurrency
                );
                MPI_Barrier(this->comm_spec_.comm());
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
                    // Note: It needs to be judged here, because the index of u as the entry may have been deleted.
                    if(this->Fc_map[u] != this->ID_default_value){
                        inc_build_trav_index(this->Fc_map[u], new_graph);
                    }
                }
            }
            else{
// #pragma omp parallel for num_threads(NUM_THREADS) //schedule(dynamic,1)
//                 for(vid_t i = 0; i < len; i++){
//                     vertex_t u(ids[i]);
//                     // if(this->Fc_map[u] != this->ID_default_value && reset_spnode_edges[this->Fc_map[u]].size() == 0){ // Has not been deleted, and has not been reset, the reset phase must include incremental calculations
//                     if(this->Fc_map[u] != this->ID_default_value && !had_reset[this->Fc_map[u]]){ // Has not been deleted, and has not been reset, the reset phase must include incremental calculations
//                         inc_build_trav_index(this->Fc_map[u], new_graph);
//                     }
//                     if(i % 1000000 == 0){
//                         LOG(INFO) << "----id=" << i << " computing index" << std::endl;
//                     }
//                 }
// #pragma omp barrier
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                    for(vid_t i = begin; i < end; i++){
                        vertex_t u(ids[i]);
                        // build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                        if(this->Fc_map[u] != this->ID_default_value && !had_reset[this->Fc_map[u]]){
                            // if_touch[this->Fc_map[u]] = true; // 测试
                            inc_build_trav_index(this->Fc_map[u], new_graph, values_array[tid], deltas_array[tid]);
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
                        build_trav_index(this->Fc_map[u], new_graph);
                    }
                    if(i % 1000000 == 0){
                        LOG(INFO) << "----id=" << i << " computing index" << std::endl;
                    }
                }
            }
            else{
// #pragma omp parallel for num_threads(NUM_THREADS) //schedule(dynamic,1)
//                 for(vid_t i = 0; i < len; i++){
//                     vertex_t u(ids[i]);
//                     if(this->Fc_map[u] != this->ID_default_value){
//                         build_trav_index(this->Fc_map[u], new_graph);
//                     }
//                     if(i % 1000000 == 0){
//                         LOG(INFO) << "----id=" << i << " computing index" << std::endl;
//                     }
//                 }
// #pragma omp barrier
                this->ForEachIndex(len, [this, &ids, &new_graph](int tid, vid_t begin, vid_t end) {
                    // LOG(INFO) << "build index, tid=" << tid << " range: "<< begin << "->" << end;
                    for(vid_t i = begin; i < end; i++){
                        vertex_t u(ids[i]);
                        // build_trav_index(i, this->graph_, values_array[tid], deltas_array[tid]);
                        if(this->Fc_map[u] != this->ID_default_value){
                            // if_touch[this->Fc_map[u]] = true; // 测试
                            build_trav_index(this->Fc_map[u], new_graph, values_array[tid], deltas_array[tid]);
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

    void reset_inc_build_tarv_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph){
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];
        std::unordered_map<vid_t, delta_t> deltas;
        std::unordered_map<vid_t, value_t> values;

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            values[v.GetValue()] = this->app_->GetIdentityElement();
            deltas[v.GetValue()].Reset(this->app_->GetIdentityElement());
        }
        for(auto e : spnode.inner_delta){
            values[e.first.GetValue()] = e.second.value;
        }
        for(auto e : spnode.inner_delta){
            deltas[e.first.GetValue()] = e.second;
        }
        
        /* reset by dependency */
        std::unordered_set<vertex_t> curr_modified, next_modified;
        for (auto& pair : reset_spnode_edges[spid]) {
            vid_t u_id = pair.first, v_id = pair.second;

            vertex_t u(u_id), v(v_id);
            auto parent_gid = deltas[v_id].parent_gid;

            if (parent_gid == this->graph_->Vertex2Gid(u)) {
                curr_modified.insert(v);
            }
        }
        if(curr_modified.size() == 0){ // bound_delta maybe have been chaged, so, can't return.
            // reset_spnode_edges[spid].clear(); // Corresponds to the conditions of stage 2
            /* build new bound_delta index in supernodes */
            fianl_build_trav_index_bound(values, deltas, new_graph, node_set, spnode);
            return;
        }
        while (curr_modified.size() > 0){
            for(auto u : curr_modified){
                auto u_gid = this->graph_->Vertex2Gid(u);
                auto oes = this->graph_->GetOutgoingAdjList(u);

                for (auto e : oes) {
                    auto v = e.neighbor;
                    auto parent_gid = deltas[v.GetValue()].parent_gid;
                    if (parent_gid == u_gid) {
                        next_modified.insert(v);
                    }
                }

                values[u.GetValue()] = this->app_->GetIdentityElement();
                deltas[u.GetValue()].Reset(this->app_->GetIdentityElement());
            }

            curr_modified.clear();
            curr_modified.swap(next_modified);
        }

        // Start a round without any condition on new_graph
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v.GetValue()];
            auto& delta = deltas[v.GetValue()];

            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                // auto it = oes.begin();
                // granular_for(j, 0, out_degree, (out_degree > 1024), {
                //     auto& e = *(it + j);
                //     // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                //     if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                //         delta_t outv;
                //         this->app_->Compute(v, value, delta, oes, e, outv);
                //         if(this->app_->AccumulateDelta(deltas[e.neighbor.GetValue()], outv)){
                //             next_modified.insert(e.neighbor);
                //         }
                //     }
                // })
                for(auto e : oes){
                    // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor.GetValue()], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                }
            }
        }
        
        had_reset[spid] = 1; // Corresponds to the conditions of stage 2
        if_touch[spid] = 1;
        /* iterative calculation */
        run_to_convergence(values, deltas, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(values, deltas, new_graph, node_set, spnode);
    }

    void inc_build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph){
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];
        std::unordered_map<vid_t, delta_t> deltas;
        std::unordered_map<vid_t, value_t> values;

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            values[v.GetValue()] = this->app_->GetIdentityElement();
            deltas[v.GetValue()].Reset(this->app_->GetIdentityElement());
        }
        for(auto e : spnode.inner_delta){
            values[e.first.GetValue()] = e.second.value;  // The value of delta is used directly.
        }
        for(auto e : spnode.inner_delta){
            deltas[e.first.GetValue()] = e.second;
        }

        // Start a round without any condition on new_graph
        std::unordered_set<vertex_t> next_modified;
        for(auto v : node_set){
            const auto& oes = new_graph->GetOutgoingAdjList(v);
            auto& value = values[v.GetValue()];
            auto& delta = deltas[v.GetValue()];

            auto out_degree = oes.Size();
            if(out_degree > 0 && delta.value != this->app_->GetIdentityElement()){
                // auto it = oes.begin();
                // granular_for(j, 0, out_degree, (out_degree > 1024), {
                //     auto& e = *(it + j);
                //     // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                //     if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                //         delta_t outv;
                //         this->app_->Compute(v, value, delta, oes, e, outv);
                //         if(this->app_->AccumulateDelta(deltas[e.neighbor.GetValue()], outv)){
                //             next_modified.insert(e.neighbor);
                //         }
                //     }
                // })
                for(auto e : oes){
                    // if(std::find(node_set.begin(), node_set.end(), e.neighbor) != node_set.end()){ // Only sent to internal vertices
                    if(ids_id == this->id2spids[e.neighbor]){ // Only sent to internal vertices
                        delta_t outv;
                        this->app_->Compute(v, value, delta, oes, e, outv);
                        if(this->app_->AccumulateDelta(deltas[e.neighbor.GetValue()], outv)){
                            next_modified.insert(e.neighbor);
                        }
                    }
                }
            }
        }
        if(next_modified.size() > 0){
            if_touch[spid] = 1;
        }
        /* iterative calculation */
        run_to_convergence(values, deltas, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(values, deltas, new_graph, node_set, spnode);
    }

    /* use a VertexArray */
    void reset_inc_build_tarv_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, VertexArray<value_t, vid_t>& values,  VertexArray<delta_t, vid_t>& deltas){
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            values[v] = this->app_->GetIdentityElement();
            deltas[v].Reset(this->app_->GetIdentityElement());
        }
        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;
        }
        for(auto e : spnode.inner_delta){
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
            fianl_build_trav_index_bound(values, deltas, new_graph, node_set, spnode);
            return;
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

                values[u] = this->app_->GetIdentityElement();
                deltas[u].Reset(this->app_->GetIdentityElement());
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
        run_to_convergence(values, deltas, new_graph, node_set, next_modified, source);
        /* build new index in supernodes */
        fianl_build_trav_index(values, deltas, new_graph, node_set, spnode);
    }

    /* use a VertexArray */
    void inc_build_trav_index(const vid_t spid, const std::shared_ptr<fragment_t>& new_graph, VertexArray<value_t, vid_t>& values,  VertexArray<delta_t, vid_t>& deltas){
        supernode_t& spnode = this->supernodes[spid];
        const auto& source = spnode.id;
        const vid_t ids_id = this->id2spids[source];
        std::vector<vertex_t> &node_set = this->supernode_ids[spnode.ids];

        /* 
            init values/deltas by inner_delta and inner_value, 
            Note that the inner_value and inner_delta at this time have been converted into weights 
        */
        for(auto v : node_set){
            values[v] = this->app_->GetIdentityElement();
            deltas[v].Reset(this->app_->GetIdentityElement());
        }
        for(auto e : spnode.inner_delta){
            values[e.first] = e.second.value;  // The value of delta is used directly.
        }
        for(auto e : spnode.inner_delta){
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
        if(next_modified.size() > 0){
            if_touch[spid] = 1;
            run_to_convergence(values, deltas, new_graph, node_set, next_modified, source);
            /* build new index in supernodes */
            fianl_build_trav_index(values, deltas, new_graph, node_set, spnode);
        }
        else{
        // 可以分为建立所有索引和边界索引
            fianl_build_trav_index_bound(values, deltas, new_graph, node_set, spnode);
        }
    }


public:
    std::vector<std::pair<vid_t, vid_t>> reset_edges;
    std::vector<std::vector<std::pair<vid_t, vid_t>> > reset_spnode_edges;
    std::vector<short int> had_reset; // Mark whether the super point is reset and calculated
    std::vector<short int> if_touch; // Mark whether to update the inside of the superpoint
    std::vector<VertexArray<value_t, vid_t>> values_array; // use to calulate indexes in parallel
    std::vector<VertexArray<delta_t, vid_t>> deltas_array;
};

}  // namespace grape
#endif  // GRAPE_FRAGMENT_TRAV_COMPRESSOR_H_