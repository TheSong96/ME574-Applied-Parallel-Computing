#define N 10

__global__ void arradd (float *a, float f, int N)
 {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) a[i] = a[i] + float;
}

int main()
{
  float a = {0,0,0,0,0,0,0,0,0,0}
  float h_a[N];
  float *d_a;
  cudaMalloc ((void **) &a_d, N*sizeof(float));
  int n_blocks = 1
  int block_size = 32

  cudaThreadSynchronize ();
  cudaMemcpy (d_a, h_a, SIZE, cudaMemcpyHostToDevice));

  arradd <<< n_blocks, block_size >>> (d_a, 2.0, N*sizeof(float));

  cudaThreadSynchronize ();
  cudaMemcpy (h_a, d_a, SIZE, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL (cudaFree (a_d));
}
