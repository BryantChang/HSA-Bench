#ifndef __IMAGE_OPS_H__
#define __IMAGE_OPS_H__
typedef unsigned char uchar;
float* readBmpImage(const char *filename, int* widthOut, int* heightOut);
void storeBmpImage(float *imageOut, const char *filename, int rows, int cols, 
                const char* refFilename);
#endif