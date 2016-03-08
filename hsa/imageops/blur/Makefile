#include
include ../../common_hsa.mk
#macros
PROJ=blur
all:$(PROJ)
$(PROJ):$(PROJ).c blurKernel.o $(INC_DIR_IMGOPS)/imageOps.o
	$(CC) -I$(INC_DIR_IMGOPS) $^  -L$(LINK_HSA_DIR) -l$(LIB_HSA) -l$(LIB_MATH)  -o $@
blurKernel.o:
	@snack.sh -c -opt 3 -vv blurKernel.cl
blurKernel:
	@snack.sh -c -opt 3 -vv blurKernel.cl
imageOps.o:$(INC_DIR_IMGOPS)/imageOps.c
	$(CC) -I$(INC_DIR_IMGOPS) $^ -o $(INC_DIR_IMGOPS)$@
imageOps:$(INC_DIR_IMGOPS)/imageOps.c
	$(CC) -I$(INC_DIR_IMGOPS) $^ -o $(INC_DIR_IMGOPS)/imageOps.o
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf blurKernel.h