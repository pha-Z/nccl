#
# Makefile for NCCL Profiler Overhead Experiments
#

# Define all experiment directories
EXPERIMENTS := 01_collectives_overhead 02_p2p_overhead 03_comm_ops_overhead

# Default target
all: $(EXPERIMENTS)

# Build all experiments
$(EXPERIMENTS):
	@echo "Building $@..."
	@$(MAKE) -C $@

# Clean all experiments
clean:
	@echo "Cleaning all experiments..."
	@for exp in $(EXPERIMENTS); do \
		$(MAKE) -C $$exp clean; \
	done

# Help target
help:
	@echo "NCCL Profiler Overhead Experiments"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Build all experiments"
	@echo "  clean    - Clean all build artifacts"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Experiments:"
	@echo "  01_collectives_overhead - Measure overhead for collective operations"
	@echo "  02_p2p_overhead        - Measure overhead for P2P operations"
	@echo "  03_comm_ops_overhead   - Measure overhead for communicator operations"

.PHONY: all clean help $(EXPERIMENTS)
