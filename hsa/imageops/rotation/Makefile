#include
include ../../common_hsa.mk
#macros
PROJ=rotation
all:$(PROJ)
$(PROJ):$(PROJ).c rotationKernel.o $(INC_DIR_IMGOPS)/imageOps.o
	$(CC) -I$(INC_DIR_IMGOPS) $^  -L$(LINK_HSA_DIR) -l$(LIB_HSA) -l$(LIB_MATH)  -o $@
rotationKernel.o:
	@snack.sh -c -opt 3 -vv rotationKernel.cl
rotationKernel:
	@snack.sh -c -opt 3 -vv rotationKernel.cl
imageOps.o:$(INC_DIR_IMGOPS)/imageOps.c
	$(CC) -I$(INC_DIR_IMGOPS) $^ -o $(INC_DIR_IMGOPS)/$@
imageOps:$(INC_DIR_IMGOPS)/imageOps.c
	$(CC) -I$(INC_DIR_IMGOPS) $^ -o $(INC_DIR_IMGOPS)/imageOps.o
clean:
	rm -rf $(PROJ)
	rm -rf *.o
	rm -rf rotationKernel.h