/*
All modification made by Intel Corporation: © 2017 Intel Corporation

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

#include <math.h>
#include <vector>

#include "caffe/gen_anchors.hpp"

namespace caffe {


static void CalcBasicParams(const anchor& base_anchor,                                       // input
                            float& width, float& height, float& x_center, float& y_center)   // output
{
    width  = base_anchor.end_x - base_anchor.start_x + 1.0f;
    height = base_anchor.end_y - base_anchor.start_y + 1.0f;

    x_center = base_anchor.start_x + 0.5f * (width - 1.0f);
    y_center = base_anchor.start_y + 0.5f * (height - 1.0f);
}


static void MakeAnchors(const vector<float>& ws, const vector<float>& hs, float x_center, float y_center,   // input
                        vector<anchor>& anchors)                                                            // output
{
    int len = ws.size();
    anchors.clear();
    anchors.resize(len);

    for (unsigned int i = 0 ; i < len ; i++) {
        // transpose to create the anchor
        anchors[i].start_x = x_center - 0.5f * (ws[i] - 1.0f);
        anchors[i].start_y = y_center - 0.5f * (hs[i] - 1.0f);
        anchors[i].end_x   = x_center + 0.5f * (ws[i] - 1.0f);
        anchors[i].end_y   = y_center + 0.5f * (hs[i] - 1.0f);
    }
}


static void CalcAnchors(const anchor& base_anchor, const vector<float>& scales,        // input
                        vector<anchor>& anchors)                                       // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    CalcBasicParams(base_anchor, width, height, x_center, y_center);

    int num_scales = scales.size();
    vector<float> ws(num_scales), hs(num_scales);

    for (unsigned int i = 0 ; i < num_scales ; i++) {
        ws[i] = width * scales[i];
        hs[i] = height * scales[i];
    }

    MakeAnchors(ws, hs, x_center, y_center, anchors);
}


static void CalcRatioAnchors(const anchor& base_anchor, const vector<float>& ratios,        // input
                             vector<anchor>& ratio_anchors)                                 // output
{
    float width = 0.0f, height = 0.0f, x_center = 0.0f, y_center = 0.0f;

    CalcBasicParams(base_anchor, width, height, x_center, y_center);

    float size = width * height;

    int num_ratios = ratios.size();

    vector<float> ws(num_ratios), hs(num_ratios);

    for (unsigned int i = 0 ; i < num_ratios ; i++) {
        float new_size = size / ratios[i];
        ws[i] = round(sqrt(new_size));
        hs[i] = round(ws[i] * ratios[i]);
    }

    MakeAnchors(ws, hs, x_center, y_center, ratio_anchors);
}

void GenerateAnchors(unsigned int base_size, const vector<float>& ratios, const vector<float> scales,   // input
                     anchor *anchors)                                                           // output
{
    float end = (float)(base_size - 1);        // because we start at zero

    anchor base_anchor(0.0f, 0.0f, end, end);

    vector<anchor> ratio_anchors;
    CalcRatioAnchors(base_anchor, ratios, ratio_anchors);

    for (int i = 0, index = 0; i < ratio_anchors.size() ; i++) {
        vector<anchor> temp_anchors;
        CalcAnchors(ratio_anchors[i], scales, temp_anchors);

        for (int j = 0 ; j < temp_anchors.size() ; j++) {
            anchors[index++] = temp_anchors[j];
        }
    }
}

}  // namespace caffe
