#include
include ../../common_hsa.mk
#macros
PROJ=matmul
all:$(PROJ)
$(PROJ):$(PROJ).c matmulKernel.o
	$(CC)  $^ -L$(LINK_HSA_DIR) -l$(LIB_HSA) -l$(LIB_MATH) -l$(LIB_BSD)  -o $@
matmulKernel.o:
	@snack.sh -c -opt 3 -vv matmulKernel.cl
matmulKernel:
	@snack.sh -c -opt 3 -vv matmulKernel.cl
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf matmulKernel.h