#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


#ifdef USE_CUDA
template<typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
                               const unsigned int* mask,
                               const unsigned int threshold, const float scale,
                               Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    if (this->phase_ == TRAIN) {
      unsigned int* mask =
          static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
      caffe_gpu_rng_uniform(count, (unsigned intc*) (mask));  // NOLINT
      // set thresholds
      // NOLINT_NEXT_LINE(whitespace/operators)
      float scale_val = scale_train_ ? scale_ : 1.f;
      DropoutForward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
                                        CAFFE_CUDA_NUM_THREADS)(
          count, bottom_data, mask, uint_thres_, scale_val, top_data);
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(count, bottom_data, top_data);
      if (! scale_train_) {
        caffe_scal<Dtype>(count, 1. / scale_, top_data);
      }
    }
#endif  // USE_CUDA
  }
}

#ifdef USE_CUDA
template<typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
                                const unsigned int* mask,
                                const unsigned int threshold, const float scale,
                                Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}
#endif  // USE_CUDA

template<typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
      if (this->phase_ == TRAIN) {
        const unsigned int* mask = static_cast<const unsigned int*>(rand_vec_
            .gpu_data());
        const int count = bottom[0]->count();
        // NOLINT_NEXT_LINE(whitespace/operators)
        DropoutBackward<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS)(
            count, top_diff, mask, uint_thres_, scale_, bottom_diff);
        CUDA_POST_KERNEL_CHECK;
      } else {
        caffe_copy(top[0]->count(), top_diff, bottom_diff);
      }
#endif  // USE_CUDA
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
