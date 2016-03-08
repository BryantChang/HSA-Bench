#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#define PLATFORM_TO_USE 0
#include "common.h"
#include "common_rand.h"
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "common_rand.h"
#define D_FACTOR 0.85
#define RANDOM 1
#define TXT 2
const char *kernelPath = "pagerankKernel.cl";
const int max_iter = 1000;
const float threshold = 0.00001;

void check(cl_int status, const char* cmd) {
    if(status != CL_SUCCESS) {
        printf("%s failed (%d)\n", cmd, status);
        exit(-1);
    }
}

//read graph from bigdatabench
int *init_pages(int n, char *filename, int *noutlinks) {
    printf("%s\n", filename);
    int *pages = malloc(sizeof(*pages) * n * n);
    int i, j, k, t;
    int from, to;
    char line[256];
    FILE *fp = fopen(filename, "r");
    if(fp == NULL) {
        printf("cannot open the file\n");
        exit(-1);
    }
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            pages[i * n + j] = 0;
        }
    }
    while(fgets(line, 256, fp)) {
        //skip the statment
        if(line[0] == '#') {
            continue;
        }else {
            sscanf(line, "%d\t%d", &from, &to);
            pages[from * n + to] = 1;
        }
    }

    //calculate the outlinks
    for(i = 0; i < n; i++) {
        noutlinks[i] = 0;
        for(j = 0; j < n; j++) {
            if(pages[i * n + j]) {
                noutlinks[i] += 1;
            }
        }
    }
    //avoid the zero outlink
    if(noutlinks[i] == 0){
        do { k = abs(common_rand()) % n; } while ( k == i);
        pages[i * n + k] = 1;
        noutlinks[i] = 1;
    }
    return pages;

}


//init pages by random
int *random_pages(int n, int *noutlinks, int divisor){
    int i, j, k, t;
    int *pages = malloc(sizeof(*pages) * n * n); // matrix 1 means link from j->i

    if (divisor <= 0) {
        fprintf(stderr, "ERROR: Invalid divisor '%d' for random initialization, divisor should be greater or equal to 1\n", divisor);
        exit(1);
    }

    for(i = 0; i < n; i++){
        noutlinks[i] = 0;
        for(j = 0; j < n; j++){
            if(i != j && (abs(common_rand()) % divisor == 0)){
                pages[i * n + j] = 1;
                noutlinks[i] += 1;
            }
        }

        // the case with no outlinks is avoided
        if(noutlinks[i] == 0){
            do { k = abs(common_rand()) % n; } while ( k == i);
            pages[i * n + k] = 1;
            noutlinks[i] = 1;
        }
    }
    return pages;
}

void init_array(float *a, int n, float val) {
    int i;
    for(i = 0; i < n; i++) {
        a[i] = val;
    }
}

void printI(int *aa, int m, int n) {
    int i = 0;
    int j = 0;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%d\t", aa[i * n + j]);
        }
        printf("\n");
    }
}


void printM(float *aa, int m, int n){
    int i = 0;
    int j = 0;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            fprintf(stderr, "%lf,", aa[i * n + j]);
        }
        fprintf(stderr, "\n");
    }
}


void usage() {
    printf("./pagerank -k kind(1:random,2:from txt) -t thresh -n matrix_size -i iter -f filename\n");
}


int main(int argc, char** argv) {
    int *pages;
    float *maps;
    float *page_ranks;
    int *noutlinks;
    int t;
    int max_diff;

    char ch;
    int kind = 0;
    int opt, opt_index = 0;
    char filename[256];

    int i = 0;
    int j;
    int n = 1000;
    int iter = max_iter;
    float thresh = threshold;
    int divisor = 2;
    int nb_links = 0;


    //operate the params
    while((ch = getopt(argc, argv, "k:t:n:i:f:")) != EOF) {
        switch(ch) {
            case 'k':
                kind = atoi(optarg);
                break;
            case 'i':
                iter = atoi(optarg);
                break;
            case 't':
                thresh = atof(optarg);
                break;
            case 'f':
                strcpy(filename, optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            default:
                usage();
                return -1;

        }
    }
    //check the params
    if(kind == 0 || iter == 0 || !strcmp(filename, "") 
        || thresh == 0.0f || n == 0) {
        usage();
        exit(1);
    }

    //*pagerank present to the element size
    long dataSizePagesRanks = sizeof(float) * n;
    long dataSizePages = sizeof(float) * n * n;
    long dataSizeMaps = sizeof(float) * n * n;
    long dataSizeNoutlinks = sizeof(int) * n;
    long dataSizeDif = sizeof(float) * n;
    long dataSizeNzeros = sizeof(float) * n;

    page_ranks = (float*)malloc(sizeof(*page_ranks) * n);
    maps = (float*)malloc(sizeof(*maps) * n * n);
    noutlinks = (int*)malloc(sizeof(*noutlinks) * n);
    max_diff = 99.0f;
    for(i = 0;i < n;i++) {
        noutlinks[i] = 0;
    }
    if(kind == RANDOM) {
        printf("start init the pages\n");
        pages = random_pages(n, noutlinks, divisor);
    }else if(kind == TXT) {
        printf("start init the pages\n");
        pages = init_pages(n, filename, noutlinks);
    }
    printf("init complete\n");
    printf("\n");
    //printI(pages, n, n);
   // exit(1);
    init_array(page_ranks, n, 1.0 / (float)n);
    nb_links = 0;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            nb_links += pages[i * n + j];
        }
    }

    //printf("nb_links=%d\n", nb_links);
    float *diffs, *nzeros;
    diffs = (float*)malloc(sizeof(float) * n);
    nzeros = (float*)malloc(sizeof(float) * n);
    for(i = 0; i < n; i++) {
        diffs[i] = 0.0f;
        nzeros[i]= 0.0f;
    }


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

    //create the buffer
    cl_mem d_pages, d_page_ranks, d_outlinks, d_maps, d_diff;


    d_pages = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizePages, NULL,
        &status);
    check(status, "clCreateBuffer");

    d_maps = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizeMaps, NULL,
        &status);
    check(status, "clCreateBuffer");

    d_page_ranks = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizePagesRanks, NULL,
        &status);
    check(status, "clCreateBuffer");

    d_outlinks = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizeNoutlinks, NULL,
        &status);
    check(status, "clCreateBuffer");

    d_diff = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSizeDif, NULL,
        &status);
    check(status, "clCreateBuffer");


    //write buffers
    status = clEnqueueWriteBuffer(queue, d_pages, CL_TRUE, 0, dataSizePages,
        pages, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(queue, d_page_ranks, CL_TRUE, 0, dataSizePagesRanks,
        page_ranks, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(queue, d_maps, CL_TRUE, 0, dataSizeMaps, 
        maps, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(queue, d_diff, CL_TRUE, 0, dataSizeDif,
        diffs, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    status = clEnqueueWriteBuffer(queue, d_outlinks, CL_TRUE, 0, dataSizeNoutlinks,
        noutlinks, 0, NULL, NULL);
    check(status, "clEnqueueWriteBuffer");

    const char* source = readSource(kernelPath);
    


    //create a program object with source and build it
    cl_program program;
    program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, "-D D_FACTOR=0.85 ", NULL, NULL);

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
    cl_kernel map_kernel, reduce_kernel;
    map_kernel = clCreateKernel(program, "map_page_rank", &status);
    check(status, "clCreateKernel");

    reduce_kernel = clCreateKernel(program, "reduce_page_rank", &status);
    check(status, "clCreateKernel");

    size_t globalSize[1] = {n};
    status = clSetKernelArg(map_kernel, 0, sizeof(cl_mem), &d_pages);
    status |= clSetKernelArg(map_kernel, 1, sizeof(cl_mem), &d_page_ranks);
    status |= clSetKernelArg(map_kernel, 2, sizeof(cl_mem), &d_maps);
    status |= clSetKernelArg(map_kernel, 3, sizeof(cl_mem), &d_outlinks);
    status |= clSetKernelArg(map_kernel, 4, sizeof(int), &n);
    check(status, "clSetKernelArg");    

    status = clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), &d_page_ranks);
    status |= clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &d_maps);
    status |= clSetKernelArg(reduce_kernel, 2, sizeof(int), &n);
    status |= clSetKernelArg(reduce_kernel, 3, sizeof(cl_mem), &d_diff);
    check(status, "clSetKernelArg");

    for(t=1; t<=iter;  t++){
        // // MAP PAGE RANKS
        status = clEnqueueNDRangeKernel(queue, map_kernel, 1, NULL, globalSize, NULL, 0, NULL, NULL);
        // REDUCE PAGE RANKS
        status = clEnqueueNDRangeKernel(queue, reduce_kernel, 1, NULL, globalSize, NULL, 0, NULL, NULL);

        status = clEnqueueReadBuffer(queue, d_diff, CL_TRUE, 0, dataSizeDif, diffs, 0, NULL, NULL);
        clFinish(queue);
        //get the max
        float max = 0;
        for(i = 0; i < n; i++) {
            if(diffs[i] > max) {
                max = diffs[i];
            }
        }
        if(max < thresh) break;
        status = clEnqueueWriteBuffer(queue, d_diff, CL_TRUE, 0, dataSizeNzeros, nzeros, 0, NULL, NULL);
    }

    status = clEnqueueReadBuffer(queue, d_maps, CL_TRUE, 0,  dataSizeMaps, maps, 0, NULL, NULL);
    status = clEnqueueReadBuffer(queue, d_page_ranks, CL_TRUE, 0, dataSizePagesRanks, page_ranks, 0, NULL, NULL);
    clFinish(queue);

    // for(i = 0; i < n; i++) {
    //     printf("%f    ", page_ranks[i]);
    // }
    printf("\n");
    printf("All complete\n");
    printf("total iter times is %d\n", t);




    return 0;
}


