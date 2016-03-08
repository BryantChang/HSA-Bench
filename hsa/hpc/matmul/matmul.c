/*
 
   matmul.c : Matrix Multiplication Example - Host Code
 
   This example shows the benefit of using tiled Kernels with local data.
   It calls three different kernels, simple_sgemm_tt, tiled_sgemm_tt, and tiled_sgemm_tn.
   See matmulKernels.cl for the implementations of these kernels in c. 
   In most cases, tiled_sgemm_tt will run much faster than simple_stemm_tt.
   
   For comparison, A  CPU version of sgemm_tn is included in this file. 
 
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/param.h> 
#include <time.h>
#include "matmul.h"


typedef struct {
   int width;
   int height;
   int stride;
   int hpad;
   float* elements; 
} Matrix;

u_int32_t arc4random();
void CPU_sgemm_tn(const int M,const int N,const int K,const float alpha, const float* A,const int LDA, 
     const float* B,const int LDB, const float beta,float* C, const int LDC) ;

/*  The following header file is generated by cloc -c matmulKernels.cl */
#include "matmulKernel.h"

int main(int argc, char** argv){

   Matrix A,B,Bt,C;
   int a1,a2,a3,i,j;
   struct timespec start_time, end_time;
   long int nanosecs,total_ops;
   float gflopsSimple,gflopsTiled,gflopsCPU,gflopsTiledtn;

   /*  Read dimensions, allocate storage and generate random matrix */
   a1 = atoi(argv[1]); /* Height of A */
   a2 = atoi(argv[2]); /* Width of A , and Height of B*/
   a3 = atoi(argv[3]); /* Width of B */
   printf("a1:%d;a2:%d;a3:%d",a1,a2,a3);
   total_ops = 2 * (long int) a1 * (long int) a2 * (long int) a3;
   printf("\n\n");

   A.height = a1;
   A.width = a2;
   A.stride = (((A.width-1)/BLOCK_SIZE)+1) * BLOCK_SIZE;
   A.hpad = (((A.height-1)/BLOCK_SIZE)+1) * BLOCK_SIZE;
   A.elements = (float*)malloc(A.stride * A.hpad* sizeof(float));

   B.height = a2;
   B.width = a3;
   B.stride = (((B.width-1)/BLOCK_SIZE)+1) * BLOCK_SIZE;
   B.hpad = (((B.height-1)/BLOCK_SIZE)+1) * BLOCK_SIZE;
   B.elements = (float*)malloc(B.stride * B.hpad * sizeof(float));

   /* Bt is same as B but stored in column-major order */
   Bt.height = B.height; 
   Bt.width = B.width;
   Bt.stride = B.stride;
   Bt.hpad = B.hpad;
   Bt.elements = (float*)malloc(Bt.stride * Bt.hpad * sizeof(float));

   C.height = a1;
   C.width = a3;
   C.stride = (((C.width-1)/BLOCK_SIZE)+1) * BLOCK_SIZE;
   C.hpad = (((C.height-1)/BLOCK_SIZE)+1) * BLOCK_SIZE;
   C.elements = (float*)malloc(C.stride * C.hpad * sizeof(float));

   for(i = 0; i < A.hpad ; i++)
      for(j = 0; j < A.stride; j++) {
         if (( j<A.width ) && (i<A.height)) {
            A.elements[i*A.stride + j] = rand() * 1.0 / RAND_MAX;   
         } else {
            A.elements[i*A.stride + j] = 0.0;
         }
      }

   for(i = 0; i < B.hpad ; i++)
      for(j = 0; j < B.stride; j++) {
         if (( j<B.width ) && (i<B.height)) {
            B.elements[i*B.stride+j] = rand() * 1.0 / RAND_MAX;   
            Bt.elements[j*Bt.stride+i] = B.elements[i*B.stride+j] ;
         } else {
            B.elements[i*B.stride+j] = 0.0;
            Bt.elements[j*Bt.stride+i] = 0.0;
         }
      }

   /* zero C */
   for(i = 0; i < C.hpad; i++)
      for(j = 0; j < C.stride; j++)
         C.elements[i*C.stride+j] = 0.0;

   printf("Matrix A: %d by %d \n",A.height,A.width);
   for(i = 0; i < MIN(10, A.height); i++){
      for(j = 0; j < MIN(10, A.width); j++)
         printf("%.2f ", A.elements[i*A.stride+j]);
      printf("\n");
   }
   printf("\n");

   printf("Matrix B: %d by %d \n",B.height,B.width);
   for(i = 0; i < MIN(10, B.height); i++){
      for(j = 0; j < MIN(10, B.width); j++)
         printf("%.2f ", B.elements[i*B.stride+j]);
      printf("\n");
   }
   printf("\n");
   SNK_INIT_LPARM(lparm,0);
   lparm->ndim=2;
   lparm->gdims[0]=C.hpad;
   lparm->gdims[1]=C.stride;
   lparm->ldims[0]=BLOCK_SIZE;
   lparm->ldims[1]=BLOCK_SIZE;
   printf("Calling Tiled Kernel tn ... \n");
   tiled_sgemm_tn(A.height,Bt.width,Bt.height,1.0,A.elements,A.stride,Bt.elements,Bt.stride,1.0,C.elements,C.stride,lparm); 
   

   printf("Matrix C: %d by %d  after tiled_sgemm_tn kernel\n",C.height,C.width);
   for(i = 0; i < MIN(10, C.height); i++){
      for(j = 0; j < MIN(10, C.width); j++)
         printf("%6.0f ", C.elements[i*C.stride+j]);
      printf("\n");
   }
   printf("\n");
   printf("Dimensions=%d %d %d\n",a1,a2,a3);

}


