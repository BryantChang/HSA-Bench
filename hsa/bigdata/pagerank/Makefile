#include
include ../../common_hsa.mk
#macros
PROJ=kmeans
all:$(PROJ)
$(PROJ):$(PROJ).cc kmeansKernel.o
	g++ -std=c++11 -o kmeans kmeans_kernel.o kmeans.cc -L$(LINK_HSA_DIR) -l$(LIB_HSA) -l$(LIB_MATH)
kmeansKernel.o:
	@snack.sh -c -opt 3 -vv kmeansKernel.cl
kmeansKenel:
	@snack.sh -c -opt 3 -vv kmeansKernel.cl
.PHONY:clean
clean:
	rm -rf *.o
	rm -rf $(PROJ)
	rm -rf kmeansKernel.h



