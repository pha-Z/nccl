# NCCL Advanced Features Examples

SUBDIRS = 01_cuda_graphs 02_interleaved_computation_communication 03_multiple_collectives_group 04_multi_stream 05_device_api_simple

.PHONY: all clean install $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

install:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir install; \
	done

