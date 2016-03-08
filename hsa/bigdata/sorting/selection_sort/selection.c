#include <stdio.h>
#include <stdlib.h>
#include "selectionKernel.h"
#define ARR_SIZE 50


void printArray(float *arr, int size) {
	int i;
	for(i = 0; i < size; i++) {
		printf("%0.2f  ",arr[i]);
		if((i+1) % 10 == 0) {
			printf("\n");
		}
	}
}

int main(int argc, char **argv) {
	//init the array
	float *in, *out;
	int i;
	in = (float*)malloc(ARR_SIZE * sizeof(float));
	out = (float*)malloc(ARR_SIZE * sizeof(float));

	printf("the size of array is %d\n", ARR_SIZE);

	for(i = 0; i < ARR_SIZE; i++) {
		in[i] = (ARR_SIZE - i) * 1.0;
		out[i] = 0.0;
	}

	printf("the array is complete\n");
	printArray(in, ARR_SIZE);

	//init lparm
	SNK_INIT_LPARM(lparm, ARR_SIZE);
    lparm->ndim = 1;
    lparm->gdims[0] = ARR_SIZE;
    lparm->ldims[0] = 1;

    //call the kernel function
    selectionSort(in, out, lparm);

    //check the result
    printf("sort complete\n");
    printArray(out, ARR_SIZE);
	return 0;
}