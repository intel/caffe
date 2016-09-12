#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  if (bottom[0] != top[0]) {
      caffe_copy(count, bottom_data, top_data);
  }
  if (negative_slope == 0) {
      #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
          top_data[i] *= (bottom_data[i] > 0);
      }
  }
  else {
      #pragma omp parallel for
      for (int i = 0; i < count; ++i) {
          top_data[i] *= ((bottom_data[i] >= 0) + (bottom_data[i] < 0) * negative_slope);
      }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    if (bottom[0] != top[0]) {
        caffe_copy(count, top_diff, bottom_diff);
    }
    if (negative_slope == 0) {
        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            if (bottom_data[i] < 0) {
                bottom_diff[i] = 0;
            }
        }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            if (bottom_data[i] < 0) {
                bottom_diff[i] *= negative_slope;
            }
        }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
