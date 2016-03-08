PROJ=selection
CC=gcc
CFLAGS_COMPILER=-c
LINK_HSA_DIR=/opt/hsa/lib
LIB_MATH=m
LIB_HSA=hsa-runtime64
LIB_BSD=bsd
all:$(PROJ)
$(PROJ):$(PROJ).c selectionKernel.o
	$(CC) $^ -L$(LINK_HSA_DIR) -l$(LIB_HSA) -l$(LIB_MATH) -l$(LIB_BSD)  -o $@
selectionKernel.o:
	@snack.sh -c -opt 3 -vv selectionKernel.cl
selectionKernel:
	@snack.sh -c -opt 3 -vv selectionKernel.cl
.PHONY:clean
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf selectionKernel.h