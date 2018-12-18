// Calculate Gabor wavelets for forward pass
int gabor_forward(THCudaTensor *sig_tensor, THCudaTensor *gamma_tensor, 
  THCudaTensor *lambd_tensor, THCudaTensor *half_ksize, 
  int rounded_half_ksize, int dim_basis, 
  THCudaTensor *gabor_kernels);

// Calculate gradients of Gabor wavelets for backward pass
int gabor_backward(THCudaTensor *sig, THCudaTensor *gamma, THCudaTensor *lambd,
  THCudaTensor *half_ksize, int dim_basis, int rounded_half_ksize, 
  THCudaTensor *gabor_kernels, THCudaTensor *grad_sig, 
  THCudaTensor *grad_gamma, THCudaTensor *grad_lambd, 
  THCudaTensor *grad_ksize, THCudaTensor *grad_output);