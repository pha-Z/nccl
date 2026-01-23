#
# Makefile for collectives overhead experiment
#

include ../../makefiles/common.mk
MPI = 1
include ../../makefiles/examples.mk

TARGET = collectives_overhead
TARGET_MPI = collectives_overhead_mpi
SOURCES = main.cc
SOURCES_MPI = main_mpi.cc
OBJECTS = $(SOURCES:.cc=.o)
OBJECTS_MPI = $(SOURCES_MPI:.cc=.o)

all: $(TARGET) $(TARGET_MPI)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

$(TARGET_MPI): $(OBJECTS_MPI)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS_MPI) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

main_mpi.o: main_mpi.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(OBJECTS_MPI) $(TARGET) $(TARGET_MPI)

.PHONY: all clean
