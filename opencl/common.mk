AMDAPPSDKROOT=/opt/AMDAPPSDK-3.0
CC=gcc
LIB_OCL=OpenCL
INC_OCL_DIRS=$(AMDAPPSDKROOT)/include
LIB_OCL_DIRS=$(AMDAPPSDKROOT)/lib/x86_64
LIB_MATH=m
CFLAGS=-std=c99 -w