#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#define PLATFORM_TO_USE 0
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char* kernelPath = "mergeKernel.cl";

void printArray(float* arr, int size) {
    int i;
    for(i = 0; i < size; i++) {
        printf("%0.2f  ", arr[i]);
        if((i + 1) % 10 == 0) {
            printf("\n");
        }
    }
}

void check(cl_int status, const char* cmd) {
    if(status != CL_SUCCESS) {
        printf("%s failed (%d)\n", cmd, status);
        exit(-1);
    }
}


void usage() {
    printf("usage:./merge -n array_size -w work_group_size\n");
}

int main(int argc, char **argv) {
    //init the array
    float *data_in, *data_out;
    int wg_size = 8;
    int array_size = 64;
    int opt, opt_index = 0;
    char ch;
    int i;
    //operate the params
    while((ch = getopt(argc, argv, "n:w:")) !=EOF) {
        switch(ch) {
            case 'n':
                array_size = atoi(optarg);
                break;
            case 'w':
                wg_size = atoi(optarg);
                break;
            default:
                usage();
                return -1;
        }
    }


    long dataSizeIn, dataSizeOut;
    dataSizeIn = array_size * sizeof(float);
    dataSizeOut = array_size * sizeof(float);
    data_in = (float*)malloc(dataSizeIn);
    data_out = (float*)malloc(dataSizeOut);

    printf("the size of array is %d\n", array_size);
    printf("the size of work group is %d\n", wg_size);

    for(i = 0; i < array_size; i++) {
        data_in[i] = (array_size - i) * 1.0;
        data_out[i] = 0.0;
    }

    printf("array init completed\n");
    //printArray(data_in, array_size);

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

    //cl_mem
    cl_mem d_data_in, d_data_out;
    d_data_in = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSizeIn, NULL,
        &status);
    check(status, "clCreateBuffer");

    d_data_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSizeOut, NULL,
        &status);
    check(status, "clCreateBuffer");

    status = clEnqueueWriteBuffer(queue, d_data_in, CL_TRUE, 0, dataSizeIn,
        data_in, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    const char* source = readSource(kernelPath);

    char opts[256];
    sprintf(opts, "-D WG_SIZE=%d", wg_size);

    //create a program object with source and build it
    cl_program program;
    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, opts, NULL, NULL);

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

    cl_kernel merge_kernel;
    merge_kernel = clCreateKernel(program, "mergeSort", &status);
    check(status, "clCreateKernel");

    size_t globalSize[1] = {array_size};
    size_t localSize[1] = {wg_size};

    status = clSetKernelArg(merge_kernel, 0, sizeof(cl_mem), &d_data_in);
    status |= clSetKernelArg(merge_kernel, 1, sizeof(cl_mem), &d_data_out);

    status = clEnqueueNDRangeKernel(queue, merge_kernel, 1, NULL, globalSize, localSize, 0, 
        NULL, NULL);
    check(status, "clEnqueueNDRangeKernel");

    status = clEnqueueReadBuffer(queue, d_data_out, CL_TRUE, 0, dataSizeOut, data_out, 0, 
        NULL, NULL);
    check(status, "clEnqueueReadBuffer");

    //check the result
    printf("sort complete\n");
   // printArray(data_out, array_size);
    printf("\n");

    clReleaseKernel(merge_kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseMemObject(d_data_out);
    clReleaseMemObject(d_data_in);
    clReleaseContext(context);
    free(data_in);
    free(data_out);
    return 0;
}