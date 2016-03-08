#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PLATFORM_TO_USE 0
#include "common.h"
#include "imageOps.h"
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
    

#define THETA 3.14159 / 6
char* kernelPath = "blurKernel.cl";

void check(cl_int status, const char* cmd) {
    if(status != CL_SUCCESS) {
        printf("%s failed (%d)\n", cmd, status);
        exit(-1);
    }
}

void usage() {
    printf("----------------------Usage-----------------------\n");
    printf("rotation inputFile outputFile theta\n");
    return;
}

int main(int argc, char **argv) {
    if(argc < 2) {
        usage();
        return -1;
    }
    //init the filter array
    float filter[49] =
      {1,      1,      1,      1,      1,      1,      1,
       1,      1,      1,      1,      1,      1,      1,
       1,      1,      1,      1,      1,      1,      1,
       1,      1,      1,      1,      1,      1,      1,
       1,      1,      1,      1,      1,      1,      1,
       1,      1,      1,      1,      1,      1,      1,
       1,      1,      1,      1,      1,      1,      1};

    //operate the params of cmd
    const char* inputFileName; 
    const char* outputFileName; 
    inputFileName = (argv[1]);
    outputFileName = (argv[2]);

    //the image height and width
    int imageHeight, imageWidth;

    int filterWidth = 7;
    
    //read the bmp image to the memory
    float* inputImage = readBmpImage(inputFileName, &imageWidth, &imageHeight);

    //to check the read is succ
    printf("the width of the image is %d, the height of the image is %d\n", imageWidth, imageHeight);

    //calculate the datasize
    int dataSize = imageHeight * imageWidth * sizeof(float);
    int filterSize = sizeof(float) * filterWidth * filterWidth;

    //output image
    float *outputImage = NULL;
    outputImage = (float*)malloc(dataSize);

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
    
    //create the input and output buffers
    cl_mem d_input, d_output, d_filter;
    d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL,
       &status);
    check(status, "clCreateBuffer");

    d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filterSize, NULL,
       &status);
    check(status, "clCreateBuffer");

    // Copy the input image to the device
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize, NULL,
       &status);
    check(status, "clCreateBuffer");

    printf("create buffer finish\n");

    status = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0, dataSize, 
         inputImage, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0, filterSize,
        filter, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");


    const char* source = readSource(kernelPath);
    //create a program object with source and build it
    cl_program program;
    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
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
    kernel = clCreateKernel(program, "blur", &status);
    check(status, "clCreateKernel");

    //set the kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_output);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_input);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &imageWidth);
    status |= clSetKernelArg(kernel, 3, sizeof(int), &imageHeight);
    status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_filter);
    status |= clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);
    check(status, "clSetKernelArg");

    // Set the work item dimensions
    size_t globalSize[2] = {imageWidth, imageHeight};
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0,
      NULL, NULL);
    check(status, "clEnqueueNDRange");

    // Read the image back to the host
    status = clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, dataSize, 
         outputImage, 0, NULL, NULL); 

    check(status, "clEnqueueReadBuffer");


    // Write the output image to file
    storeBmpImage(outputImage, outputFileName, imageHeight, imageWidth, inputFileName);
    printf("all finish\n");
       //free opencl resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseMemObject(d_filter);
    clReleaseContext(context);

    //free host resources
    free(inputImage);
    free(outputImage);
    

}