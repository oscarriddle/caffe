#ifndef CAFFE_REDUCE_MEAN_LAYER_HPP_
#define CAFFE_REDUCE_MEAN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
  /**
   *  reduceMeanLayer reads 4D input blob data and apply operations to calculate the
   *  average of specific axis. The processed axis becomes a scaler value. Developer can
   *  decide whether to retain the specific axis [DEFAULT: keep].
   **/
  template <typename Dtype>
  class reduceMeanLayer : public NeuronLayer<Dtype>{
  public:
    explicit reduceMeanLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
    virtual inline const char* type() const { return "reduceMeanLayer";}
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			     const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			     const vector<Blob<Dtype>*>& top);
    
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  };
   
  

} // namespace caffe

#endif

