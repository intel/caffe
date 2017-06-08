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
#include <omp.h>

#include "caffe/filler.hpp"
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
    this->blobs_.resize(5);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    this->blobs_[3].reset(new Blob<Dtype>(sz));
    sz[0]=1;
    this->blobs_[4].reset(new Blob<Dtype>(sz));
    FillerParameter scale_filler_param(param.scale_filler());
    FillerParameter shift_filler_param(param.shift_filler());
    if (!param.has_scale_filler()) {
        scale_filler_param.set_type("constant");
        scale_filler_param.set_value(1);
    }
    if (!param.has_shift_filler()) {
        shift_filler_param.set_type("constant");
        shift_filler_param.set_value(0);
    }
    shared_ptr<Filler<Dtype> > scale_filler(GetFiller<Dtype>(scale_filler_param));
    shared_ptr<Filler<Dtype> > shift_filler(GetFiller<Dtype>(shift_filler_param));
    scale_filler->Fill(this->blobs_[0].get());
    shift_filler->Fill(this->blobs_[1].get());
    for (int i = 2; i < 5; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);
  mean_.Reshape(1, channels_, 1, 1);
  variance_.Reshape(1, channels_, 1, 1);
  x_norm_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void batch_mean(Dtype* batch_data, int N, int C, int HW, Dtype* batch_mean) {
    int CHW = C * HW;
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
        batch_mean[i] = 0;
        Dtype* data = batch_data + i * HW;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < HW; ++k) {
                batch_mean[i] += data[k];
            }
            data += CHW;
        }
    }

    int NHW = N * HW;
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
        //LOG(INFO) << "thread id: " << omp_get_thread_num();
        batch_mean[i] /= NHW;
    }
}

template <typename Dtype>
void batch_variance(Dtype* batch_data, int N, int C, int HW, const Dtype* batch_mean, Dtype* batch_variance) {
    int CHW = C * HW;
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
        batch_variance[i] = 0;
        Dtype* data = batch_data + i * HW;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < HW; ++k) {
                batch_variance[i] += data[k];
            }
            data += CHW;
        }
    }

    int NHW = N * HW;
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
        batch_variance[i] /= NHW;
        batch_variance[i] -= batch_mean[i] * batch_mean[i];
    }
}

template<typename Dtype>
void batch_norm(Dtype* bottom_data, const Dtype* batch_mean, const Dtype* batch_variance, \
        const Dtype* scale, const Dtype* shift, Dtype eps, int N, int C, int HW, \
        Dtype* x_norm, Dtype* top_data) {
    int CHW = C * HW;
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
        Dtype factor = 1.0 / sqrt(eps + batch_variance[i]);
        Dtype* bottom = bottom_data + i * HW;
        Dtype* top = top_data + i * HW;
        Dtype* norm = x_norm + i * HW;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < HW; ++k) {
                norm[k] = (bottom[k] - batch_mean[i]) * factor;
                top[k] = scale[i] * norm[k] + shift[i];
            }
            bottom += CHW;
            top += CHW;
            norm += CHW;
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count() / (num * channels_);

  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  const Dtype* shift_data = this->blobs_[1]->cpu_data();
  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype factor = this->blobs_[4]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[4]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), factor,
        this->blobs_[2]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), factor,
        this->blobs_[3]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    // compute mean
      batch_mean(bottom[0]->mutable_cpu_data(), num, channels_, spatial_dim, \
              mean_.mutable_cpu_data());
  }

  if (!use_global_stats_) {
    // compute variance using var(X) = E(X^2) - (EX)^2
    caffe_powx(bottom[0]->count(), bottom_data, Dtype(2),
        x_norm_.mutable_cpu_data());  // X^2
    // compute variance
    batch_variance(x_norm_.mutable_cpu_data(), num, channels_, spatial_dim, \
            mean_.cpu_data(), variance_.mutable_cpu_data());
    // compute and save moving average
    this->blobs_[4]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[4]->mutable_cpu_data()[0] += 1;

    caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
        moving_average_fraction_, this->blobs_[2]->mutable_cpu_data());
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_cpu_axpby(variance_.count(), bias_correction_factor,
        variance_.cpu_data(), moving_average_fraction_,
        this->blobs_[3]->mutable_cpu_data());
  }

  // subtract mean and normalize variance
  batch_norm(bottom[0]->mutable_cpu_data(), mean_.cpu_data(), variance_.cpu_data(), \
         scale_data, shift_data, eps_, num, channels_, spatial_dim, x_norm_.mutable_cpu_data(), \
         top_data);
}

template<typename Dtype>
void batch_diff(Dtype* top_diff, Dtype* x_norm, int N, int C, int HW, Dtype* scale_diff, \
       Dtype* shift_diff) {
    int CHW = C * HW;
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
        scale_diff[i] = 0;
   //     shift_diff[i] = 0;
        Dtype* top = top_diff + i * HW;
        Dtype* norm = x_norm + i * HW;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < HW; ++k) {
                scale_diff[i] += top[k] * norm[k];
   //             shift_diff[i] += top[k];
            }
            top += CHW;
            norm += CHW;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
   //     scale_diff[i] = 0;
        shift_diff[i] = 0;
        Dtype* top = top_diff + i * HW;
        Dtype* norm = x_norm + i * HW;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < HW; ++k) {
   //             scale_diff[i] += top[k] * norm[k];
                shift_diff[i] += top[k];
            }
            top += CHW;
            norm += CHW;
        }
    }
}

template<typename Dtype>
void batch_x_diff(Dtype* top_diff, Dtype* x_norm, const Dtype* variance, \
        const Dtype* scale_diff, const Dtype* shift_diff, const Dtype* scale, \
        Dtype eps, int N, int C, int HW, Dtype* bottom_diff) {
    int CHW = C * HW;
    int m = N * HW;
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) {
        Dtype factor = scale[i] / sqrt(variance[i] + eps);
        Dtype* top = top_diff + i * HW;
        Dtype* norm = x_norm + i * HW;
        Dtype* bottom = bottom_diff + i * HW;
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < HW; ++k) {
                bottom[k] = factor * (top[k] - (shift_diff[i] + norm[k] * scale_diff[i]) / m);
            }
            top += CHW;
            norm += CHW;
            bottom += CHW;
        }
    }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype* top_diff = top[0]->mutable_cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  Dtype* norm_data = x_norm_.mutable_cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count() / (num * channels_);
  const Dtype* scale_data = this->blobs_[0]->cpu_data();
  Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* shift_diff = this->blobs_[1]->mutable_cpu_diff();
  
  // compute scale diff and shift diff
  batch_diff(top_diff, norm_data, num, channels_, spatial_dim, scale_diff, shift_diff);
  // compute bottom diff
  batch_x_diff(top_diff, norm_data, variance_.cpu_data(), scale_diff, \
        shift_diff, scale_data, eps_, num, channels_, spatial_dim, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);
}  // namespace caffe
