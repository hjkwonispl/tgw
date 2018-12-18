#ifndef _GABOR_LAYER_CUDA_KERNEL
#define _GABOR_LAYER_CUDA_KERNEL

#define BLOCK 512
#define MAX_STREAMS 512

#ifdef __cplusplus
extern "C" {
#endif

// Calculate Gabor wavelets for forward pass
void gabor_forward_cuda(float *sig, float *gamma, float *lambd, 
  float* half_ksize, int rounded_half_ksize, 
  int dim_basis, 
  float *gabor_kernels, 
  cudaStream_t stream);

// Calculate gradients of Gabor wavelets for backward pass
void gabor_backward_cuda(float *sig, float *gamma, float *lambd, 
  float* half_ksize, int dim_basis, int rounded_half_ksize,
  float *gabor_kernels, 
  float *grad_sig, float *grad_gamma, float *grad_lambd, float *grad_ksize, 
  float *grad_output, 
  cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
