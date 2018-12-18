#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "gabor_layer_cuda_kernel.h"

dim3 cuda_gridsize(int n)
{   
  int k = (n - 1) / BLOCK + 1;
  int x = k;
  int y = 1;
  if(x > 65535) {
    x = ceil(sqrt(k));
    y = (n - 1) / (x * BLOCK) + 1;
  }

  dim3 d(x, y, 1);
  
  return d;
}

__global__ void gabor_forward_kernel(float *sig, float *gamma, float *lambd,   
  float* half_ksize, int ksize,  int rounded_half_ksize, 
  int dim_kernel, int dim_basis, 
  float *gabor_kernels)
{
  int block_length = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + 
    threadIdx.x;
  if(block_length >= dim_kernel * dim_basis) return;
  
  int batch_idx = block_length / dim_kernel; 
  block_length = block_length - dim_kernel * batch_idx;
  
  int col_idx = block_length % ksize; 
  block_length = block_length / ksize;

  int row_idx = block_length % ksize;
     
  // Calculate Gabor Wavelets
  float theta = 180 / float(dim_basis) * 
    float(batch_idx) + 180/(2 * float(dim_basis));   
  float x = (float(col_idx) - float(rounded_half_ksize)) * 
    (*half_ksize) / float(rounded_half_ksize);
  float y = (float(row_idx) - float(rounded_half_ksize)) * 
    (*half_ksize) / float(rounded_half_ksize);;
  float X = x * cos(theta) + y * sin(theta);
  float Y = -x * sin(theta) + y * cos(theta);
  float gabor_element = exp( -(pow(X,2) + pow(*gamma,2) * pow(Y,2)) * 
    0.5 * pow(*sig, -2)) * cos( 2 * M_PI * X * pow(*lambd, -1) );   
  gabor_kernels[batch_idx * dim_kernel + row_idx * ksize + col_idx] = 
    gabor_element;
  
  return;
}

// Forward Gabor Wavelet Wapper
void gabor_forward_cuda(float *sig, float *gamma, float *lambd, 
  float* half_ksize, int rounded_half_ksize, int dim_basis, 
  float *gabor_kernels, 
  cudaStream_t stream)
{   
  int ksize = 2 * rounded_half_ksize + 1;  
  int dim_kernel = ksize * ksize; // Size of sampling grid for Gabor wavelet
  
  cudaError_t err;

  gabor_forward_kernel
    <<<cuda_gridsize(dim_kernel * dim_basis), BLOCK, 0, stream>>> 
    (sig, gamma, lambd, half_ksize, ksize, rounded_half_ksize, dim_kernel, 
    dim_basis, gabor_kernels);
  
  err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  
  return;
}

__global__ void gabor_backward_kernel(float *sig, float *gamma, float *lambd, 
  float *half_ksize, int dim_kernel, int dim_basis, int rounded_half_ksize,
  float *gabor_kernels, float *grad_sig, float *grad_gamma, 
  float *grad_lambd, float *grad_ksize, float *grad_output, 
  cudaStream_t stream)

{
  int block_length = (blockIdx.x + blockIdx.y * gridDim.x) * 
    blockDim.x + threadIdx.x;
  if(block_length >= dim_kernel * dim_basis) return;
 
  int ksize = 2 * rounded_half_ksize + 1; 

  int batch_idx = block_length / dim_kernel; 
  block_length = block_length - dim_kernel * batch_idx;

  int col_idx = block_length % ksize; 
  block_length = block_length / ksize;

  int row_idx = block_length % ksize;

  // Calculate gradients for backprop
  float theta = 180 / float(dim_basis) * 
    float(batch_idx) + 180/(2 * float(dim_basis));   
  float x = (float(col_idx) - float(rounded_half_ksize)) * 
    (*half_ksize) / float(rounded_half_ksize);
  float y = (float(row_idx) - float(rounded_half_ksize)) * 
    (*half_ksize) / float(rounded_half_ksize);;
  
  float X = x * cos(theta) + y * sin(theta);
  float Y = -x * sin(theta) + y * cos(theta);

  float dkern_by_Y = - (pow(*gamma, 2) * pow(*sig, -2) * Y);
  float dkern_by_X = - (X * pow(*sig,-2)) - 
    tan(2 * M_PI * pow(*lambd, -1) * X) * 2 * M_PI * pow(*lambd,-1);
  float grad_ksize_element = (dkern_by_X * cos(theta) + dkern_by_Y * 
    (-sin(theta))) * x / (*half_ksize);
  grad_ksize_element = grad_ksize_element + (dkern_by_X * sin(theta) + 
    dkern_by_Y * cos(theta)) * y / (*half_ksize);
     
  float grad_gamma_element = -pow(Y,2) * (*gamma) * pow((*sig), -2);
  float grad_lambd_element = tan(2 * M_PI * X * pow(*lambd,-1)) * 
    (2 * M_PI * X * pow(*lambd,-2));   
  float grad_sig_element = (pow(X,2) +  (pow(*gamma,2) * pow(Y,2))) * 
    pow(*sig,-3);
   
  grad_sig_element = gabor_kernels[batch_idx * dim_kernel + 
    row_idx * ksize + col_idx] * grad_sig_element;
  grad_ksize_element = gabor_kernels[batch_idx * dim_kernel + 
    row_idx * ksize + col_idx] * grad_ksize_element;
  grad_gamma_element = gabor_kernels[batch_idx * dim_kernel + 
    row_idx * ksize + col_idx] * grad_gamma_element;
  grad_lambd_element = gabor_kernels[batch_idx * dim_kernel + 
    row_idx * ksize + col_idx] * grad_lambd_element;
  
  grad_sig[batch_idx * dim_kernel + row_idx * ksize + col_idx] = 
    grad_output[batch_idx * dim_kernel + row_idx * ksize + col_idx] * 
    grad_sig_element;
  grad_ksize[batch_idx * dim_kernel + row_idx * ksize + col_idx] = 
    grad_output[batch_idx * dim_kernel + row_idx * ksize + col_idx] * 
    grad_ksize_element;  
  grad_lambd[batch_idx * dim_kernel + row_idx * ksize + col_idx] = 
    grad_output[batch_idx * dim_kernel + row_idx * ksize + col_idx] * 
    grad_lambd_element;
  grad_gamma[batch_idx * dim_kernel + row_idx * ksize + col_idx] = 
    grad_output[batch_idx * dim_kernel + row_idx * ksize + col_idx] * 
    grad_gamma_element;
  
  return;
}

// Backward Gabor Wavelet Wapper
void gabor_backward_cuda(float *sig, float *gamma, float *lambd, 
  float *half_ksize, int dim_basis, int rounded_half_ksize, 
  float *gabor_kernels, float *grad_sig, float *grad_gamma, 
  float *grad_lambd, float *grad_ksize, float *grad_output, 
  cudaStream_t stream)
{
  int dim_kernel = (2 * rounded_half_ksize + 1) * 
    (2 * rounded_half_ksize + 1);

  cudaError_t err;
  
  gabor_backward_kernel
    <<<cuda_gridsize(dim_kernel * dim_basis), BLOCK, 0, stream>>>
    (sig, gamma, lambd, half_ksize, 
     dim_kernel, dim_basis, rounded_half_ksize,
     gabor_kernels, grad_sig, grad_gamma, grad_lambd, 
     grad_ksize, grad_output, 
     stream);
  
  err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  
  return;
}