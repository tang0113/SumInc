#ifndef ANALYTICAL_APPS_CC_CC_INGRESS_H_
#define ANALYTICAL_APPS_CC_CC_INGRESS_H_

#include "grape/app/traversal_app_base.h"

namespace grape {

template <typename FRAG_T, typename VALUE_T>
class CCIngress : public TraversalAppBase<FRAG_T, VALUE_T> {
 public:
  using fragment_t = FRAG_T;
  using vid_t = typename fragment_t::vid_t;
  using vertex_t = typename fragment_t::vertex_t;
  using value_t = typename TraversalAppBase<FRAG_T, VALUE_T>::value_t;
  using delta_t = typename TraversalAppBase<FRAG_T, VALUE_T>::delta_t;
  using edata_t = typename fragment_t::edata_t;
  using adj_list_t = typename fragment_t::adj_list_t;
  using adj_list_index_t = AdjList<vid_t, delta_t>;

  value_t GetInitValue(const vertex_t& v) const override {
    return GetIdentityElement();
  }

  delta_t GetInitDelta(const vertex_t& v) const override {
    return this->GenDelta(v, this->fragment().Vertex2Gid(v));
  }

  bool CombineValueDelta(value_t& lhs, const delta_t& rhs) override {
    if (lhs > rhs.value) {
      lhs = rhs.value;
      return true;
    }
    return false;
  }

  bool AccumulateDeltaAtomic(delta_t& lhs, const delta_t& rhs) override {
    return lhs.SetIfLessThanAtomic(rhs);
  }

  void Compute(const vertex_t& u, const value_t& value, const delta_t& delta,
               DenseVertexSet<vid_t>& modified) override {
    auto gid = delta.value;
    auto oes = this->fragment().GetOutgoingAdjList(u);

    if (FLAGS_cilk) {
      auto out_degree = oes.Size();
      auto it = oes.begin();

      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        delta_t delta_to_send = this->GenDelta(u, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.neighbor;
        delta_t delta_to_send = this->GenDelta(u, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      }
    }
  }

  value_t GetIdentityElement() const override {
    return std::numeric_limits<value_t>::max();
  }

  // to support compress
  bool AccumulateDelta(delta_t& lhs, const delta_t& rhs) override {
    return lhs.SetIfLessThan(rhs); 
  }

  delta_t GetInitDelta(const vertex_t& v, const vertex_t& source) const override {
    return this->GenDelta(v, this->fragment().Vertex2Gid(v));
  }

  void Compute(const vertex_t& v, const value_t& value, const delta_t& delta,
                const adj_list_t& oes, const Nbr<vid_t, edata_t>& oe, delta_t& outv) override {
     auto gid = delta.value;
     outv = this->GenDelta(v, gid);
  }

  void Compute(const vertex_t& u, const value_t& value, const delta_t& delta,
               const adj_list_t& oes,
               DenseVertexSet<vid_t>& modified) override {
    auto gid = delta.value;
    if (FLAGS_cilk) {
      auto out_degree = oes.Size();
      auto it = oes.begin();

      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        delta_t delta_to_send = this->GenDelta(u, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.neighbor;
        delta_t delta_to_send = this->GenDelta(u, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      }
    }
  }

  void ComputeByIndexDelta(const vertex_t& u, const value_t& value, const delta_t& delta,
               const std::vector<std::pair<vertex_t, delta_t>>& oes,
               DenseVertexSet<vid_t>& modified) override {
    auto gid = delta.value;
    // this->f_send_delta_num += oes.size();

    if (FLAGS_cilk) {
      auto out_degree = oes.size();
      auto it = oes.begin();
      
      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.first;
        // auto new_dist = e.second.value + dist;
        delta_t delta_to_send = this->GenDelta(e.second.parent_gid, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.first;
        // auto new_dist = e.second.value + dist;
        delta_t delta_to_send = this->GenDelta(e.second.parent_gid, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      }
    }
  }
  

  void ComputeByIndexDelta(const vertex_t& u, const value_t& value, const delta_t& delta,
               const adj_list_index_t& oes,
               DenseVertexSet<vid_t>& modified) override {
    auto gid = delta.value;
    if (FLAGS_cilk) {
      auto out_degree = oes.Size();
      auto it = oes.begin();
      
      granular_for(j, 0, out_degree, (out_degree > 1024), {
        auto& e = *(it + j);
        auto v = e.neighbor;
        // auto new_dist = e.data.value + dist;
        delta_t delta_to_send = this->GenDelta(e.data.parent_gid, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      })
    } else {
      for (auto e : oes) {
        auto v = e.neighbor;
        // auto new_dist = e.data.value + dist;
        delta_t delta_to_send = this->GenDelta(e.data.parent_gid, gid);

        if (this->AccumulateToAtomic(v, delta_to_send)) {
          modified.Insert(v);
        }
      }
    }
  }

  void revCompute(delta_t& delta, delta_t& rt_delta) override {
    rt_delta = delta;
  }

};

}  // namespace grape

#endif  // ANALYTICAL_APPS_CC_CC_INGRESS_H_
