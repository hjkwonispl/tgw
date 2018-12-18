# Unix commands.
PYTHON := python
NVCC_COMPILE := /usr/local/cuda/bin/nvcc -c -o
RM_RF := rm -rf

# Library compilation rules.
NVCC_FLAGS := -x cu -Xcompiler -fPIC -shared

# File structure.
BUILD_DIR := build
INCLUDE_DIRS := include
TORCH_FFI_BUILD := build_ffi.py
MATHUTIL_KERNEL := $(BUILD_DIR)/gabor_layer_cuda_kernel.so
TORCH_FFI_TARGET := $(BUILD_DIR)/gabor_layer/_gabor_layer.so

INCLUDE_FLAGS := $(foreach d, $(INCLUDE_DIRS), -I$d)

all: $(TORCH_FFI_TARGET)

$(TORCH_FFI_TARGET): $(MATHUTIL_KERNEL) $(TORCH_FFI_BUILD)
	$(PYTHON) $(TORCH_FFI_BUILD)

$(BUILD_DIR)/%.so: src/%.cu
	@ mkdir -p $(BUILD_DIR)
	# Separate cpp shared library that will be loaded to the extern C ffi
	$(NVCC_COMPILE) $@ $? $(NVCC_FLAGS) $(INCLUDE_FLAGS)

clean:
	$(RM_RF) $(BUILD_DIR) $(MATHUTIL_KERNEL)
