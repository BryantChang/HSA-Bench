#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fftnum.h"
#include "fftKernel.h"
int main(int argc, char **argv) {
    int err, i;
    size_t global_size, local_size;
    unsigned long local_mem_size;
   
    float gflopsCPU,gflopsHSA;
    long int nanosecs,total_ops;
    int numgroup;
    /* Data and buffer */
    int direction;
    unsigned int num_points, points_per_group, stage;
    float data[NUM_POINTS*2],out_data[NUM_POINTS*2];
   
    struct timespec start_time, end_time;
    double error, check_input[NUM_POINTS][2], check_output[NUM_POINTS][2];
   

    /* Initialize data */
    srand(time(NULL));
    for(i = 0; i < NUM_POINTS; i++) {
        data[2 * i] = random();
        data[2 * i + 1] = random();
    }
	num_points = NUM_POINTS;
	direction = DIRECTION;
	points_per_group = SUMS_PER_WKG;
	if(points_per_group > num_points)
        points_per_group = num_points;
   
    local_size = 1024;
    SNK_INIT_LPARM(lparm,0);
    lparm->ndim = 1;
    lparm->ldims[0] = local_size;
    lparm->gdims[0] = (num_points/points_per_group)*local_size;
    numgroup=num_points / SUMS_PER_WKG;
    fft_init(data,out_data,points_per_group, numgroup, lparm);
    /* Enqueue further stages of the FFT */
    if(num_points > points_per_group) {
        for(stage = 2; stage <= num_points/points_per_group; stage <<= 1) {
            fft_stage(out_data,stage,points_per_group,lparm);
        }
    }
    /* Scale values if performing the inverse FFT */
   if(direction < 0) {
       fft_scale(out_data,num_points,lparm);
   }
   error = 0.0;
   printf("%u-point \n", num_points);
   printf("completed with %lf average relative error.\n", error);
   printf("GFLOPS for HSA             =  %6.6f*10000\n",gflopsHSA);
   return 0;
}
