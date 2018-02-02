/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/benchmark.hpp"


namespace caffe {

/**
 * @brief Connects Layer%s together into a directed acyclic graph (DAG)
 *        specified by a NetParameter.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param, const Net* root_net = NULL);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL,
      const Net* root_net = NULL, std::string engine = "");
  virtual ~Net() {}

  /// @brief Initialize a network with a NetParameter.
  void Init(const NetParameter& param);

  /**
   * @brief Run Forward and return the result.
   *
   */
  const vector<Blob<Dtype>*>& Forward(Dtype* loss = NULL);
  /// @brief DEPRECATED; use Forward() instead.
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL) {
    LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: ForwardPrefilled() "
        << "will be removed in a future version. Use Forward().";
    return Forward(loss);
  }

  /**
   * The From and To variants of Forward and Backward operate on the
   * (topological) ordering by which the net is specified. For general DAG
   * networks, note that (1) computing from one layer to another might entail
   * extra computation on unrelated branches, and (2) computation starting in
   * the middle may be incorrect if all of the layers of a fan-in are not
   * included.
   */
  Dtype ForwardFromTo(int start, int end);
  Dtype ForwardFrom(int start);
  Dtype ForwardTo(int end);
  /// @brief DEPRECATED; set input blobs then use Forward() instead.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);

  /**
   * @brief Zeroes out the diffs of all net parameters.
   *        Should be run before Backward.
   */
  void ClearParamDiffs(int learnable_param_id);
  void ClearParamDiffs();

  /**
   * The network backward should take no input and output, since it solely
   * computes the gradient w.r.t the parameters, and the data has already been
   * provided during the forward pass.
   */
  void Backward();
  void BackwardFromTo(int start, int end);
  void BackwardFrom(int start);
  void BackwardTo(int end);

  /**
   * @brief Reshape all layers from bottom to top.
   *
   * This is useful to propagate changes to layer sizes without running
   * a forward pass, e.g. to compute output feature size.
   */
  void Reshape();

  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }

  /// @brief Updates the network weights based on the diff values computed.
  void Update();
  /**
   * @brief Shares weight data of owner blobs with shared blobs.
   *
   * Note: this is called by Net::Init, and thus should normally not be
   * called manually.
   */
  void ShareWeights();

  /**
   * @brief For an already initialized net, implicitly copies (i.e., using no
   *        additional memory) the pre-trained layers from another Net.
   */
  void ShareTrainedLayersWith(const Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  /**
   * @brief For an already initialized net, copies the pre-trained layers from
   *        another Net.
   */
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  void CopyTrainedLayersFromBinaryProto(const string trained_filename);
  void CopyTrainedLayersFromHDF5(const string trained_filename);
  /// @brief Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false) const;
  /// @brief Writes the net to an HDF5 file.
  void ToHDF5(const string& filename, bool write_diff = false) const;

  /// @brief returns the network name.
  inline const string& name() const { return name_; }
  /// @brief returns the layer names
  inline const vector<string>& layer_names() const { return layer_names_; }
  /// @brief returns the blob names
  inline const vector<string>& blob_names() const { return blob_names_; }
  /// @brief returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() const {
    return blobs_;
  }
  /// @brief returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() const {
    return layers_;
  }
  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }
  /**
   * @brief returns the bottom vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& bottom_vecs() const {
    return bottom_vecs_;
  }
  /**
   * @brief returns the top vecs for each layer -- usually you won't
   *        need this unless you do per-layer checks such as gradients.
   */
  inline const vector<vector<Blob<Dtype>*> >& top_vecs() const {
    return top_vecs_;
  }
  /// @brief returns the ids of the top blobs of layer i
  inline const vector<int> & top_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, top_id_vecs_.size()) << "Invalid layer id";
    return top_id_vecs_[i];
  }
  /// @brief returns the ids of the bottom blobs of layer i
  inline const vector<int> & bottom_ids(int i) const {
    CHECK_GE(i, 0) << "Invalid layer id";
    CHECK_LT(i, bottom_id_vecs_.size()) << "Invalid layer id";
    return bottom_id_vecs_[i];
  }
  inline const vector<vector<bool> >& bottom_need_backward() const {
    return bottom_need_backward_;
  }
  inline const vector<Dtype>& blob_loss_weights() const {
    return blob_loss_weights_;
  }
  inline const vector<bool>& layer_need_backward() const {
    return layer_need_backward_;
  }
  /// @brief returns the parameters
  inline const vector<shared_ptr<Blob<Dtype> > >& params() const {
    return params_;
  }
  inline const vector<Blob<Dtype>*>& learnable_params() const {
    return learnable_params_;
  }

  vector<int> get_layer_learnable_param_ids(int layer_id) const;

  /// @brief returns the learnable parameter learning rate multipliers
  inline const vector<float>& params_lr() const { return params_lr_; }
  inline const vector<bool>& has_params_lr() const { return has_params_lr_; }
  /// @brief returns the learnable parameter decay multipliers
  inline const vector<float>& params_weight_decay() const {
    return params_weight_decay_;
  }
  inline const vector<bool>& has_params_decay() const {
    return has_params_decay_;
  }
  const map<string, int>& param_names_index() const {
    return param_names_index_;
  }
  inline const vector<int>& param_owners() const { return param_owners_; }
  inline const vector<string>& param_display_names() const {
    return param_display_names_;
  }

  inline const pair<int, int>& param_layer_indices(int param_id) {
    return param_layer_indices_[param_id];
  }

  /// @brief Input and output blob numbers
  inline int num_inputs() const { return net_input_blobs_.size(); }
  inline int num_outputs() const { return net_output_blobs_.size(); }
  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
  inline const vector<Blob<Dtype>*>& output_blobs() const {
    return net_output_blobs_;
  }
  inline const vector<int>& input_blob_indices() const {
    return net_input_blob_indices_;
  }
  inline const vector<int>& output_blob_indices() const {
    return net_output_blob_indices_;
  }
  bool has_blob(const string& blob_name) const;
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name) const;
  bool has_layer(const string& layer_name) const;
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name) const;

  void set_debug_info(const bool value) { debug_info_ = value; }

  // Helpers for Init.
  /**
   * @brief Remove layers that the user specified should be excluded given the current
   *        phase, level, and stage.
   */
  static void FilterNet(const NetParameter& param,
      NetParameter* param_filtered);

  /**
   * @brief Remove or Replace layers that the user specified should be excluded to increase
   *        computational performance.
   */
  static void CompileNet(const NetParameter& param,
    NetParameter* param_compiled);

  /**
  * @brief This is rule that analyze layer if it is of type Scale and if that is the case
  *        and previous layer which serves as input layer to Scale Layer is MKLBatchNorm
  *        then scale layer can be dropped
  */
  // TODO: Make it decent C++ anonymous function etc.
  static void CompilationRuleOne(const NetParameter& param,
                                 NetParameter* param_compiled);
  /**
  * @brief This is rule that analyze layer if it is of type MKLDNNReLU and if that is the case
  *        and previous layer which serves as input layer to MKLDNNReLU Layer is MKLDNNConvolution
  *        then MKLDNNReLU layer can be dropped
  */


  static void CompilationRuleTwo(const NetParameter& param,
                                 NetParameter* param_compiled);

  /**
  * @brief This is rule analyze if layer is of type MKLBatchNorm
  *        and is to perform in place computation 
  *        if positive then make it doing out-ofplace computation
  */
  static void CompilationRuleThree(const NetParameter& param,
                             NetParameter* param_compiled);

  /**
  * @brief This is rule analyze for conv/elt/relu fusion.
  */
  static void CompilationRuleFour(const NetParameter& param,
                             NetParameter* param_compiled);
  /**
   * @brief If find "Conv--BN--Scale" in current network, merge BN and Scale layer into Convolution
   * layers, this optimization only works in caffe TEST phase now.
   */

  static void GetBlobConsumers(std::vector<const LayerParameter*> &cnsmer_blobs,
                                                const string& blob_name_to_find,
                                                const NetParameter& param,
                                                int layer_id);

  static void GetNeedToCancelInplaceLayers(
      vector<vector<const LayerParameter*>>& layer_pairs,    
      std::map<string, int>& specified_layer_blob_name_to_index,      
      std::map<string, int>& inplace_blob_name_to_index,
      vector<string>& each_blob_list,
      const NetParameter& param);

  static void ParseNetInplaceStatus(
      std::map<string, int>& inplace_blob_name_to_index,
      std::map<string, int>& specified_layer_blob_name_to_index,      
      vector<vector<string>>& specified_layer_input_blob_names,
      NetParameter* param, const string& specified_layer_type);

  /// @brief return whether NetState state meets NetStateRule rule
  static bool StateMeetsRule(const NetState& state, const NetStateRule& rule,
      const string& layer_name);
  /**
   * @brief Look at the layer activations and parameters to find the maximum
   * absolute values. The following layers are considered: Convolution,
   * InnerProduct.
   *
   * @param layer_name The layers that should be quantized to fixed point.
   * @param max_in The highest layer input.
   * @param max_out The highest layer output.
   * @param max_param The highest layer parameter.
   *
   * For layer parameters, the biases are ignored.
   */
  void RangeInLayers(vector<string>* layer_name, vector<Dtype>* max_in,
      vector<Dtype>* max_out, vector<vector<Dtype>>* max_param, string scaling);
  /**
   * @brief Find the maximum value in a blob.
   */
  vector<Dtype> FindMax(Blob<Dtype>* blob, bool is_single=true);
  inline const map<string,int>& blob_names_index() const {
    return blob_names_index_;
  }

#ifdef CAFFE_PER_LAYER_TIMINGS
  /* Timers for performance measurements */
  Timer timer;
#ifdef FW_OVERLAP_OPT
  Timer wait_timer;
#endif
  std::vector<double> forward_time_per_layer;
  std::vector<double> backward_time_per_layer;
  std::vector<double> update_time_per_layer;
  double cleardiffs_time_per_iter;
#ifdef USE_MLSL
  std::vector<double> startcomm_time_per_layer;
  std::vector<double> waitcomm_time_per_layer;

  std::vector<double> startcomm_start_time_per_layer;
  std::vector<double> waitcomm_start_time_per_layer;
  std::vector<double> startcomm_stop_time_per_layer;
  std::vector<double> waitcomm_stop_time_per_layer;

#ifdef FW_OVERLAP_OPT
  std::vector<double> first_waitcomm_start_time_per_layer;
  std::vector<double> first_waitcomm_stop_time_per_layer;
  std::vector<double> first_update_start_time_per_layer;
  std::vector<double> first_update_stop_time_per_layer;
#endif

#endif

  std::vector<double> forward_time_per_layer_total;
  std::vector<double> backward_time_per_layer_total;
  std::vector<double> update_time_per_layer_total;
  double cleardiffs_time_per_iter_total;
#ifdef USE_MLSL
  std::vector<double> startcomm_time_per_layer_total;
  std::vector<double> waitcomm_time_per_layer_total;
#endif

  std::vector<double> forward_start_time_per_layer;
  std::vector<double> backward_start_time_per_layer;
  std::vector<double> update_start_time_per_layer;

  std::vector<double> forward_stop_time_per_layer;
  std::vector<double> backward_stop_time_per_layer;
  std::vector<double> update_stop_time_per_layer;

  void InitTimers();
  void ResetTimers();
  void PrintTimers(bool printTotal);

  void PrintPayloadSize();
  void SaveTimeline();

#endif /* CAFFE_PER_LAYER_TIMINGS */

 protected:
  // Helpers for Init.
  /// @brief Append a new top blob to the net.
  void AppendTop(const NetParameter& param, const int layer_id,
                 const int top_id, set<string>* available_blobs,
                 map<string, int>* blob_name_to_idx);
  /// @brief Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id, set<string>* available_blobs,
                   map<string, int>* blob_name_to_idx);
  /// @brief Append a new parameter blob to the net.
  void AppendParam(const NetParameter& param, const int layer_id,
                   const int param_id);

  /// @brief Helper for displaying debug info in Forward.
  void ForwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Backward.
  void BackwardDebugInfo(const int layer_id);
  /// @brief Helper for displaying debug info in Update.
  void UpdateDebugInfo(const int param_id);
  bool bn_scale_remove_;
  bool bn_scale_merge_;
  vector<string> kept_bn_layers_;
  /// @brief The network name
  string name_;
  /// @brief The engine name
  string engine_name_;
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;
  /// @brief the blobs storing intermediate results between the layer.
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<bool> blob_need_backward_;
  /// bottom_vecs stores the vectors containing the input for each layer.
  /// They don't actually host the blobs (blobs_ does), so we simply store
  /// pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<bool> > bottom_need_backward_;
  /// top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  /// Vector of weight in the loss (or objective) function of each net blob,
  /// indexed by blob_id.
  vector<Dtype> blob_loss_weights_;
  vector<vector<int> > param_id_vecs_;
  vector<int> param_owners_;
  vector<string> param_display_names_;
  vector<pair<int, int> > param_layer_indices_;
  map<string, int> param_names_index_;
  /// blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<int> net_output_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  /// The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;
  vector<Blob<Dtype>*> learnable_params_;
  /**
   * The mapping from params_ -> learnable_params_: we have
   * learnable_param_ids_.size() == params_.size(),
   * and learnable_params_[learnable_param_ids_[i]] == params_[i].get()
   * if and only if params_[i] is an "owner"; otherwise, params_[i] is a sharer
   * and learnable_params_[learnable_param_ids_[i]] gives its owner.
   */
  vector<int> learnable_param_ids_;
  /// the learning rate multipliers for learnable_params_
  vector<float> params_lr_;
  vector<bool> has_params_lr_;
  /// the weight decay multipliers for learnable_params_
  vector<float> params_weight_decay_;
  vector<bool> has_params_decay_;
  /// The bytes of memory used by this net
  size_t memory_used_;
  /// Whether to compute and display debug info for the net.
  bool debug_info_;
  /// The root net that actually holds the shared layers in data parallelism
  const Net* const root_net_;
  DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
