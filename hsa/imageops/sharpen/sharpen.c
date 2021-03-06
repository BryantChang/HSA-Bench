#include <stdio.h>
#include <stdlib.h>
#include "imageOps.h"
#include "sharpenKernel.h"
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
      {-1,      -1,      -1,      -1,      -1,      -1,      -1,
       -1,      -1,      -1,      -1,      -1,      -1,      -1,
       -1,      -1,      -1,      -1,      -1,      -1,      -1,
       -1,      -1,      -1,      49,      -1,      -1,      -1,
       -1,      -1,      -1,      -1,      -1,      -1,      -1,
       -1,      -1,      -1,      -1,      -1,      -1,      -1,
       -1,      -1,      -1,      -1,      -1,      -1,      -1};

	//operate the params of cmd
	const char* inputFileName; 
    const char* outputFileName; 
    inputFileName = (argv[1]);
    outputFileName = (argv[2]);

    //the image height and width
    int imageHeight, imageWidth;
    
    //read the bmp image to the memory
    float* inputImage = readBmpImage(inputFileName, &imageWidth, &imageHeight);

    //to check the read is succ
    printf("the width of the image is %d, the height of the image is %d\n", imageWidth, imageHeight);

    //calculate the datasize
    int dataSize = imageHeight * imageWidth * sizeof(float);

    //output image
    float *outputImage = NULL;
    outputImage = (float*)malloc(dataSize);

    //init lparm
    SNK_INIT_LPARM(lparm, (imageWidth * imageHeight));
    lparm->ndim=2; 
    lparm->gdims[0]=imageWidth;
    lparm->gdims[1]=imageHeight;
    lparm->ldims[0]=1;
    lparm->ldims[1]=1;
    
    //call the blur function
    int filterWidth = 7;
    sharpen(outputImage, inputImage, imageWidth, imageHeight, filter, filterWidth, lparm);

    //store the image 
    storeBmpImage(outputImage, outputFileName, imageHeight, imageWidth, inputFileName);
    free(outputImage);
	return 0;
}
