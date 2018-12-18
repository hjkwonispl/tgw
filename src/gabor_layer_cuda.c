#include <THC/THC.h>
#include "gabor_layer_cuda_kernel.h"

extern THCState *state;

int gabor_forward(THCudaTensor *sig, THCudaTensor *gamma, THCudaTensor *lambd, 
  THCudaTensor *half_ksize, int rounded_half_ksize, 
  int dim_basis,
  THCudaTensor *gabor_kernels)
{   
  float* sig_flat       = THCudaTensor_data(state, sig);
  float* gamma_flat     = THCudaTensor_data(state, gamma);
  float* lambd_flat     = THCudaTensor_data(state, lambd);
  float* half_ksize_flat  = THCudaTensor_data(state, half_ksize);
  float* gabor_kernels_flat = THCudaTensor_data(state, gabor_kernels);

  cudaStream_t stream = THCState_getCurrentStream(state);
  
  gabor_forward_cuda(sig_flat, gamma_flat, lambd_flat, half_ksize_flat, 
    rounded_half_ksize, dim_basis,  gabor_kernels_flat, stream);
   
  return 1;
}

int gabor_backward(THCudaTensor *sig, THCudaTensor *gamma, THCudaTensor *lambd,    
  THCudaTensor *half_ksize, int dim_basis, int rounded_half_ksize, 
  THCudaTensor *gabor_kernels, THCudaTensor *grad_sig, 
  THCudaTensor *grad_gamma, THCudaTensor *grad_lambd, 
  THCudaTensor *grad_ksize, THCudaTensor *grad_output)
{
  float* sig_flat        = THCudaTensor_data(state, sig);
  float* gamma_flat        = THCudaTensor_data(state, gamma);
  float* lambd_flat        = THCudaTensor_data(state, lambd);
  float* half_ksize_flat     = THCudaTensor_data(state, half_ksize);
  
  float* gabor_kernels_flat = THCudaTensor_data(state, gabor_kernels);
  
  float* grad_sig_flat   = THCudaTensor_data(state, grad_sig);
  float* grad_gamma_flat = THCudaTensor_data(state, grad_gamma);
  float* grad_lambd_flat = THCudaTensor_data(state, grad_lambd);
  float* grad_ksize_flat = THCudaTensor_data(state, grad_ksize);
  float* grad_output_flat = THCudaTensor_data(state, grad_output);
  
  cudaStream_t stream = THCState_getCurrentStream(state);
  
  gabor_backward_cuda(sig_flat, gamma_flat, lambd_flat, half_ksize_flat,    
    dim_basis, rounded_half_ksize, gabor_kernels_flat, grad_sig_flat, 
    grad_gamma_flat, grad_lambd_flat, grad_ksize_flat, grad_output_flat, 
    stream);
   
  return 1;
}