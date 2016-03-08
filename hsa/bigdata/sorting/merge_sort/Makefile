#include
include ../../../common_hsa.mk
PROJ=merge
all:$(PROJ)
$(PROJ):$(PROJ).c mergeKernel.o
	$(CC) $^ -L$(LINK_HSA_DIR) -l$(LIB_HSA) -l$(LIB_MATH) -l$(LIB_BSD)  -o $@
mergeKernel.o:
	@snack.sh -c -opt 3 -vv mergeKernel.cl
mergeKernel:
	@snack.sh -c -opt 3 -vv mergeKernel.cl
.PHONY:clean
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf mergeKernel.h