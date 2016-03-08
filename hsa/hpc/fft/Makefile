#include
include ../../common_hsa.mk
#macros
PROJ=fft
all:$(PROJ)
$(PROJ):$(PROJ).c fftKernel.o
	$(CC) $^ -L$(LINK_HSA_DIR) -l$(LIB_HSA) -l$(LIB_MATH)  -o $@
fftKernel.o:
	@snack.sh -c -opt 3 -vv fftKernel.cl
fftKernel:
	@snack.sh -c -opt 3 -vv fftKernel.cl
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf fftKernel.h