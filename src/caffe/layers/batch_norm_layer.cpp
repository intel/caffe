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

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0]=1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the bias correction to zero.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } else {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  num_stats_batches_ = 1;
  stats_batch_size_ = bottom[0]->shape(0);
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  if (!use_global_stats_ && param.stats_batch_size() > 0) {
    CHECK_EQ(bottom[0]->shape(0) % param.stats_batch_size(), 0);
    num_stats_batches_ = bottom[0]->shape(0) / param.stats_batch_size();
    stats_batch_size_ = param.stats_batch_size();
  }

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
  sz[0]=stats_batch_size_;
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count(2);
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*stats_batch_size_;
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::replicate(Dtype* buffer_to_write,
                                      int num_batches,
                                      unsigned int batch_offset_incr,
                                      unsigned int channel_offset_incr,
                                      const Dtype* data_to_be_replicated) {
#ifdef _OPENMP
  #if defined(_MSC_EXTENSIONS)
    #pragma omp parallel for
  #else
    #pragma omp parallel for collapse(2)
  #endif
#endif
  for (int j = 0; j< channels_; ++j) {
    for (int n = 0; n < num_batches; ++n) {
      caffe_set(channel_offset_incr, data_to_be_replicated[j],
        buffer_to_write + j * channel_offset_incr + n * batch_offset_incr);
    }
  }
}

template <typename Dtype>
template <typename FuncTy>
void BatchNormLayer<Dtype>::replicate_to_op(Dtype* buffer_to_write,
                                      int num_batches,
                                      unsigned int batch_offset_incr,
                                      unsigned int channel_offset_incr,
                                      const Dtype* data_to_be_replicated,
                                      FuncTy op_func) {
#ifdef _OPENMP
  #if defined(_MSC_EXTENSIONS)
    #pragma omp parallel for
  #else
    #pragma omp parallel for collapse(2)
  #endif
#endif
  for (int j = 0; j< channels_; ++j) {
    for (int n = 0; n < num_batches; ++n) {
      Dtype value = data_to_be_replicated[j];
      Dtype* buffer_offsetted =
        buffer_to_write + j * channel_offset_incr + n * batch_offset_incr;
      for (int k = 0; k < channel_offset_incr; ++k) {
        buffer_offsetted[k] = op_func(buffer_offsetted[k], value);
      }
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::ForwardStatsBatch_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top, int stats_batch_idx) {
  long data_stats_count = stats_batch_size_ * bottom[0]->count(1);
  long data_offset = stats_batch_idx * data_stats_count;
  const Dtype* bottom_data = bottom[0]->cpu_data() + data_offset;
  Dtype* top_data = top[0]->mutable_cpu_data() + data_offset;
  Dtype* temp_data = temp_.mutable_cpu_data() + data_offset;
  Dtype* x_norm_data = x_norm_.mutable_cpu_data() + data_offset;
  int num = stats_batch_size_;
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);

  if (bottom[0] != top[0]) {
    caffe_copy(data_stats_count, bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    // compute mean
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        mean_.mutable_cpu_data());
  }

  // subtract mean
  replicate_to_op(top_data,
                  num,
                  spatial_dim*channels_,
                  spatial_dim,
                  mean_.cpu_data(),
                  std::minus<Dtype>());

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(data_stats_count, top_data, Dtype(2),
        temp_data);  // (X-EX)^2
    caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), temp_data,
        spatial_sum_multiplier_.cpu_data(), 0.,
        num_by_chans_.mutable_cpu_data());
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
    int m = bottom[0]->count()/num_stats_batches_/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_cpu_axpby(variance_.count(), bias_correction_factor,
        variance_.cpu_data(), moving_average_fraction_,
        this->blobs_[1]->mutable_cpu_data());
  }

  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
             variance_.mutable_cpu_data());

  // replicate variance to input size
  this->replicate(temp_data,
                  num,
                  spatial_dim*channels_,
                  spatial_dim,
                  variance_.cpu_data());

  caffe_div(data_stats_count, top_data, temp_data, top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(data_stats_count, top_data,
      x_norm_data);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::BackwardStatsBatch_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom,
    int stats_batch_idx) {
  long data_stats_count = stats_batch_size_ * bottom[0]->count(1);
  long data_offset = stats_batch_idx * data_stats_count;
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff() + data_offset;
  } else {
    caffe_copy(data_stats_count, top[0]->cpu_diff() + data_offset,
               x_norm_.mutable_cpu_diff() + data_offset);
    top_diff = x_norm_.cpu_diff() + data_offset;
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff() + data_offset;
  if (use_global_stats_) {
    caffe_div(data_stats_count, top_diff, temp_.cpu_data() + data_offset, bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.cpu_data() + data_offset;
  int num = stats_batch_size_;
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_mul(data_stats_count, top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  this->replicate(bottom_diff,
                  num,
                  spatial_dim*channels_,
                  spatial_dim,
                  mean_.cpu_data());

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(data_stats_count, top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y

  replicate_to_op(bottom_diff,
                  num,
                  spatial_dim*channels_,
                  spatial_dim,
                  mean_.cpu_data(),
                  std::plus<Dtype>());

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(data_stats_count, Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(data_stats_count, bottom_diff, temp_.cpu_data() + data_offset, bottom_diff);
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < num_stats_batches_; i++) {
    ForwardStatsBatch_cpu(bottom, top, i);
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < num_stats_batches_; i++) {
    BackwardStatsBatch_cpu(top, propagate_down, bottom, i);
  }
}


#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);
}  // namespace caffe
