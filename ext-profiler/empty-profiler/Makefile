#
# Makefile for empty NCCL profiler plugin (C version)
#

# Use CUDA_HOME if set, otherwise try CUDA_INC or common locations
CUDA_HOME ?= $(shell echo $$CUDA_HOME)
CUDA_INC  ?= $(if $(CUDA_HOME),$(CUDA_HOME)/include,$(shell echo $$CUDA_INC))

# Source and output
SRC = empty_profiler.cc
OUT = libnccl-profiler-empty.so
NCCL_HEADERS_DIR = nccl

# Compiler flags (C)
CC = gcc
CFLAGS = -fPIC -shared -Wall -std=c11
INCLUDES = -I$(NCCL_HEADERS_DIR) $(if $(CUDA_INC),-I$(CUDA_INC),)
LDFLAGS = -ldl

.PHONY: all clean setup-headers

all: setup-headers $(OUT)

# Copy NCCL headers if they don't exist
setup-headers:
	@if [ ! -d "$(NCCL_HEADERS_DIR)" ]; then \
		echo "Copying NCCL headers..."; \
		mkdir -p $(NCCL_HEADERS_DIR); \
		cp ../example/nccl/*.h $(NCCL_HEADERS_DIR)/; \
	fi

# Build the shared library
$(OUT): $(SRC) setup-headers
	@echo "Compiling $(SRC) -> $(OUT)"
	@echo "Using CUDA_INC: $(CUDA_INC)"
	$(CC) $(CFLAGS) $(INCLUDES) -x c -o $(OUT) $(SRC) $(LDFLAGS)

clean:
	rm -f $(OUT)
	@echo "Note: Keeping $(NCCL_HEADERS_DIR)/ directory. Remove manually if needed."

# Create symlink (optional, for convenience)
symlink: $(OUT)
	ln -sf $(OUT) libnccl-profiler.so

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the profiler plugin (default)"
	@echo "  clean        - Remove built files"
	@echo "  symlink      - Create libnccl-profiler.so symlink"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUDA_HOME    - Path to CUDA installation (default: \$$CUDA_HOME)"
	@echo "  CUDA_INC     - Path to CUDA headers (default: \$$CUDA_INC or \$$CUDA_HOME/include)"
