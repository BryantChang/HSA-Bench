#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define PLATFORM_TO_USE 0
#include "common.h"
#include "matmul.h"
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

char *kernelPath = "matmulKernel.cl";

int getMin(int a, int b) {
    return (a >= b) ? b : a;
}

typedef struct {
    int width;
    int height;
    int stride;
    int hpad;
    float *elements;
}Matrix;

void check(cl_int status, const char* cmd) {
    if(status != CL_SUCCESS) {
        printf("%s failed (%d)\n", cmd, status);
        exit(-1);
    }
}

int main(int argc, char** argv) {
    Matrix A, B, Bt, C;
    int a1, a2, a3, i, j;
    int m, n, k, lda, ldb, ldc;
    int block_size = 30;
    a1 = atoi(argv[1]);
    a2 = atoi(argv[2]);
    a3 = atoi(argv[3]);
    block_size = atoi(argv[4]);
    printf("a1:%d;a2:%d;a3:%d\n",a1,a2,a3);

    long dataSizeA, dataSizeB, dataSizeBt, dataSizeC;


    A.height = a1;
    A.width = a2;
    A.stride = (((A.width-1) / block_size)+1) * block_size;
    A.hpad = (((A.height-1) / block_size)+1) * block_size;
    dataSizeA = A.stride * A.hpad * sizeof(float);
    A.elements = (float*)malloc(dataSizeA);

    B.height = a2;
    B.width = a3;
    B.stride = (((B.width-1) / block_size)+1) * block_size;
    B.hpad = (((B.height-1) / block_size)+1) * block_size;
    dataSizeB = B.stride * B.hpad * sizeof(float);
    B.elements = (float*)malloc(dataSizeB);

    /* Bt is same as B but stored in column-major order */
    Bt.height = B.height; 
    Bt.width = B.width;
    Bt.stride = B.stride;
    Bt.hpad = B.hpad;
    dataSizeBt = Bt.stride * Bt.hpad * sizeof(float);
    Bt.elements = (float*)malloc(dataSizeBt);

    C.height = a1;
    C.width = a3;
    C.stride = (((C.width-1) / block_size)+1) * block_size;
    C.hpad = (((C.height-1) / block_size)+1) * block_size;
    dataSizeC = C.stride * C.hpad * sizeof(float);
    C.elements = (float*)malloc(dataSizeC);

    srand(time(NULL));
    for(i = 0; i < A.hpad; i++) {
        for(j = 0; j < A.stride; j++) {
            if((j < A.width) && (i < A.height)) {
                A.elements[i * A.stride + j] = rand() * 1.0 / RAND_MAX;
            }
        }
    }

    for(i = 0; i < B.hpad; i++) {
        for(j = 0; j < B.stride; j++) {
            if((j < B.width) && (i < B.height)) {
                B.elements[i * B.stride + j] = rand() * 1.0 / RAND_MAX;
                Bt.elements[i * Bt.stride + j] = B.elements[i * B.stride + j];
            }else {
                B.elements[i * B.stride + j] = 0.0;
                Bt.elements[i * Bt.stride + j] = 0.0;
            }
        }
    }

    //Zero C
    for(i = 0; i < C.hpad; i++) {
        for(j = 0; j < C.stride; j++) {
            C.elements[i * C.stride + j] = 0.0;
        }
    }

    //display the matrix
    printf("Matrix A: %d by %d \n",A.height,A.width);
    for(i = 0; i < getMin(10, A.height); i++){
        for(j = 0; j < getMin(10, A.width); j++)
            printf("%.2f ", A.elements[i*A.stride+j]);
        printf("\n");
    }
    printf("\n");

    printf("Matrix B: %d by %d \n",B.height,B.width);
    for(i = 0; i < getMin(10, B.height); i++){
        for(j = 0; j < getMin(10, B.width); j++)
            printf("%.2f ", B.elements[i*B.stride+j]);
        printf("\n");
    }
    printf("\n");

    //set up the OpenCL environment
    cl_int status;

    //Discovery platform
    cl_platform_id platforms[2];
    cl_platform_id platform;
    status = clGetPlatformIDs(2, platforms, NULL);
    check(status, "clGetPlatformIDs");
    platform = platforms[PLATFORM_TO_USE];

    //Discovery device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    check(status, "clGetDeviceIDs");

    //create context
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platform), 0};
    cl_context context;
    context = clCreateContext(props, 1, &device, NULL, NULL, &status);
    check(status, "clCreateContext");

    //create command queue
    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, 0, &status);
    check(status, "clCreateCommandQueue");

    //create the buffers
    cl_mem d_a, d_bt, d_c;
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSizeA, NULL,
        &status);
    check(status, "clCreateBuffer");

    d_bt = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSizeBt, NULL,
        &status);
    check(status, "clCreateBuffer");
    
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSizeC, NULL,
        &status);
    check(status, "clCreateBuffer");

    status = clEnqueueWriteBuffer(queue, d_a, CL_FALSE, 0, dataSizeA, 
         A.elements, 0, NULL, NULL);
    status |= clEnqueueWriteBuffer(queue, d_bt, CL_FALSE, 0, dataSizeBt, 
         Bt.elements, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    const char* source = readSource(kernelPath);
    


    //create a program object with source and build it
    char macro[256];
    sprintf(macro, "-D BLOCK_SIZE=%d", block_size);
    cl_program program;
    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, &macro, NULL, NULL);

    size_t log_size;
    char *program_log;
    if(status < 0) {
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    check(status, "clBuildProgram");



    //create the kernel object
    cl_kernel kernel;
    kernel = clCreateKernel(program, "tiled_sgemm_tn", &status);
    check(status, "clCreateKernel");

    float alpha, beta;
    alpha = beta = 1.0;
    m = A.height;
    n = Bt.width;
    k = Bt.height;
    lda = A.stride;
    ldb = Bt.stride;
    ldc = C.stride;

    //set arguments
    status = clSetKernelArg(kernel, 0, sizeof(int), &m);
    status |= clSetKernelArg(kernel, 1, sizeof(int), &n);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &k);
    status |= clSetKernelArg(kernel, 3, sizeof(float), &alpha);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_a);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &lda);
    status |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_bt);
    status |= clSetKernelArg(kernel, 7, sizeof(int), &ldb);
    status |= clSetKernelArg(kernel, 8, sizeof(float), &beta);
    status |= clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_c);
    status |= clSetKernelArg(kernel, 10, sizeof(int), &ldc);

    size_t globalSize[2] = {C.hpad, C.stride};
    size_t localSize[2] = {block_size, block_size};

    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    check(status, "clEnqueueNDRangeKernel");

    //read the matrix 
    status = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, dataSizeC, C.elements, 0, NULL, NULL);
    check(status, "clEnqueueReadBuffer");

    printf("Matrix C: %d by %d  after tiled_sgemm_tn kernel\n",C.height,C.width);
    for(i = 0; i < getMin(10, C.height); i++){
        for(j = 0; j < getMin(10, C.width); j++)
            printf("%.2f ", C.elements[i * C.stride + j]);
        printf("\n");
    }
    printf("\n");

    //free opencl resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_bt);
    clReleaseMemObject(d_c);
    clReleaseContext(context);
    free(A.elements);
    free(Bt.elements);
    free(C.elements);
    free(B.elements);


}

