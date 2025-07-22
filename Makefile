SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 
CXX = g++
CXXFLAGS = -std=c++11
TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  CPUVecAdd GPUVecAdd VecAddUM
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

CPUVecAdd: CPUVecAdd.cpp
	$(CXX) -o CPUVecAdd CPUVecAdd.cpp $(CXXFLAGS)

#######################################################################

GPUVecAdd: GPUVecAdd.cu
	${NVCC} -o GPUVecAdd GPUVecAdd.cu $(LIB) $(OPTIONS)

#######################################################################

VecAddUM: VecAddUM.cu
	${NVCC} -o VecAddUM VecAddUM.cu $(LIB) $(OPTIONS)