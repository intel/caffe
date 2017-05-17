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

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/fast_rcnn_layers.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA

extern "C" const char _cl_simpler_nms_layer_start;
extern "C" const char _cl_simpler_nms_layer_end;

namespace caffe {

template <typename Dtype>
void SimplerNMSLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  std::vector<simpler_nms_proposal_t> sorted_proposals_confidence;

  if (this->device_->backend() == BACKEND_CUDA) {
    NOT_IMPLEMENTED;
  } else {
    const Dtype* bottom_cls_forward_prob = bottom[0]->gpu_data();
    const Dtype* bottom_deltas_pred = bottom[1]->gpu_data();
    int image_width = bottom[2]->cpu_data()[1];
    int image_height = bottom[2]->cpu_data()[0];
    int scaled_min_bbox_size = min_bbox_size_ * (int)bottom[2]->cpu_data()[2];
    int num_anchors = anchors_blob_.shape(0) * anchors_blob_.shape(1);
    int feature_map_width = bottom[0]->shape(3);
    int feature_map_height = bottom[0]->shape(2);
    int feature_map_size = feature_map_width * feature_map_height;
    cl_mem anchors_mem = (cl_mem) anchors_blob_.gpu_data();

    viennacl::ocl::context &ctx = viennacl::ocl::get_context(
        this->device_->id());
    viennacl::ocl::program &program = this->device_->program();

    int totalProps = feature_map_size * num_anchors;
    int propSize = 5 * sizeof(Dtype);
    Blob<Dtype> out_proposal_blob;
    out_proposal_blob.Reshape(1, 1, totalProps, propSize);
    cl_mem out_proposal_mem = (cl_mem) out_proposal_blob.mutable_gpu_data();

    viennacl::ocl::kernel &proposalForward = program.get_kernel(
        CL_KERNEL_SELECT("proposalForward"));
    proposalForward.global_work_size(0, feature_map_width);
    proposalForward.global_work_size(1, feature_map_height);
    proposalForward.global_work_size(2, num_anchors);

    proposalForward.local_work_size(0, feature_map_width);
    proposalForward.local_work_size(1, 1);
    proposalForward.local_work_size(2, 1);
    viennacl::ocl::enqueue(
        proposalForward(WrapHandle((cl_mem) bottom_deltas_pred, &ctx),
                        WrapHandle(anchors_mem, &ctx),
                        WrapHandle((cl_mem) bottom_cls_forward_prob, &ctx),
                        image_height, image_width, num_anchors,
                        feat_stride_, feature_map_size,
                        feature_map_width, scaled_min_bbox_size,
                        WrapHandle( out_proposal_mem, &ctx)),
                        ctx.get_queue());
    const Dtype *out_proposals = out_proposal_blob.cpu_data();
    for (uint i = 0; i < totalProps; i++)
      if (out_proposals[i * 5 + 1] >= 0) {
        Dtype proposal_confidence = out_proposals[i * 5];
        simpler_nms_roi_t roi { out_proposals[i * 5 + 1],
                                out_proposals[i * 5 + 2],
                                out_proposals[i * 5 + 3],
                                out_proposals[i * 5 + 4] };
        simpler_nms_proposal_t proposal { roi, proposal_confidence, sorted_proposals_confidence.size() };
        sorted_proposals_confidence.push_back(proposal);
      }
  }
  
  sort_and_keep_at_most_top_n(sorted_proposals_confidence, pre_nms_topN_);

  auto res = simpler_nms_perform_nms(sorted_proposals_confidence, iou_threshold_, post_nms_topN_);
  size_t res_num_rois = res.size();

  Dtype* top_data = top[0]->mutable_cpu_data();
  for (size_t i = 0; i < res_num_rois; ++i)
  {
    top_data[5 * i + 0] = 0;    // roi_batch_ind, always zero on test time
    top_data[5 * i + 1] = res[i].x0;
    top_data[5 * i + 2] = res[i].y0;
    top_data[5 * i + 3] = res[i].x1;
    top_data[5 * i + 4] = res[i].y1;
  }

  top[0]->Reshape(vector<int>{ (int)res_num_rois, 5 });
}

template <typename Dtype>
void SimplerNMSLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  // NOT_IMPLEMENTED;
}
INSTANTIATE_LAYER_GPU_FUNCS(SimplerNMSLayer);

}  // namespace caffe
#endif  // USE_GREENTEA
