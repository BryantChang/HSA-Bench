#include <stdio.h>
#include <stdlib.h>
#include "mergeKernel.h"
#define WG_SIZE 8
void printArray(float *arr, int size) {
	int i;
	for(i = 0; i < size; i++) {
		printf("%0.2f  ",arr[i]);
		if((i+1) % 10 == 0) {
			printf("\n");
		}
	}
}

void usage() {
	printf("usage:./merge array_size\n");
}

int main(int argc, char **argv) {
	//init the array
	if(argc < 2) {
		usage();
		exit(1);
	}
	int array_size = atoi(argv[1]);
	float *in, *out;
	int i;
	in = (float*)malloc(array_size * sizeof(float));
	out = (float*)malloc(array_size * sizeof(float));

	printf("the size of array is %d\n", array_size);

	for(i = 0; i < array_size; i++) {
		in[i] = (array_size - i) * 1.0;
		out[i] = 0.0;
	}

	printf("the array is complete\n");
	//printArray(in, array_size);

	//init lparm
	SNK_INIT_LPARM(lparm, array_size);
    lparm->ndim = 1;
    lparm->gdims[0] = array_size;
    lparm->ldims[0] = WG_SIZE;

    //call the kernel function
    mergeSort(in, out, lparm);

    //check the result
    printf("sort complete\n");
    //printArray(out, array_size);
    printf("\n");
	return 0;
}