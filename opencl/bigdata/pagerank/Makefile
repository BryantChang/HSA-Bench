#include
include ../../common.mk
#macros
PROJ=pagerank
CFLAGS_COMPILER=-c
INC_COMMON_DIR=../../include
all:$(PROJ)
$(PROJ): $(PROJ).c $(INC_COMMON_DIR)/common.o common_rand.o
	$(CC) $(CFLAGS) -o $@ $^ -I$(INC_OCL_DIRS) -I$(INC_COMMON_DIR)  -L$(LIB_OCL_DIRS) -l$(LIB_OCL) -l$(LIB_MATH)
common.o:$(INC_COMMON_DIR)/common.c
	$(CC) $(CFLAGS) -o $(INC_COMMON_DIR)/$@  $^ -I$(INC_COMMON_DIR) -I$(INC_OCL_DIRS) -L$(LIB_OCL_DIRS) -l$(LIB_OCL) 
common:$(INC_COMMON_DIR)/common.c
	$(CC) $(CFLAGS) -o $(INC_COMMON_DIR)/common.o  $^ -I$(INC_COMMON_DIR) -I$(INC_OCL_DIRS) -L$(LIB_OCL_DIRS) -l$(LIB_OCL) 
common_rand.o:common_rand.c
	$(CC) $(CFLAGS_COMPILER) $^ -l$(LIB_MATH) -o $@
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf $(INC_COMMON_DIR)/common.o
	rm -rf pagerankKernel.h