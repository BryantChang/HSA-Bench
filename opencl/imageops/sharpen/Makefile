#include
include ../../common.mk
#macros
PROJ=sharpen
INC_COMMON_DIR=../../include
INC_IMGOPS_DIR=../include
all:$(PROJ)
$(PROJ): $(PROJ).c $(INC_IMGOPS_DIR)/imageOps.o $(INC_COMMON_DIR)/common.o
	$(CC) $(CFLAGS) -o $@ $^ -I$(INC_OCL_DIRS) -I$(INC_COMMON_DIR) -I$(INC_IMGOPS_DIR) -L$(LIB_OCL_DIRS) -l$(LIB_OCL) -l$(LIB_MATH)
imageOps.o:$(INC_DIR_IMGOPS)/imageOps.c
	$(CC) -o $(INC_DIR_IMGOPS)/$@  $^ -I$(INC_DIR_IMGOPS)
imageOps:$(INC_DIR_IMGOPS)/imageOps.c
	$(CC) -o $(INC_DIR_IMGOPS)/imageOps.o $^ -I$(INC_DIR_IMGOPS) 
common.o:$(INC_COMMON_DIR)/common.c
	$(CC) $(CFLAGS) -o $(INC_COMMON_DIR)/$@  $^ -I$(INC_COMMON_DIR) -I$(INC_OCL_DIRS) -L$(LIB_OCL_DIRS) -l$(LIB_OCL) 
common:$(INC_COMMON_DIR)/common.c
	$(CC) $(CFLAGS) -o $(INC_COMMON_DIR)/common.o  $^ -I$(INC_COMMON_DIR) -I$(INC_OCL_DIRS) -L$(LIB_OCL_DIRS) -l$(LIB_OCL) 
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf $(INC_IMGOPS_DIR)/imageOps.o
	rm -rf sharpenKernel.h
