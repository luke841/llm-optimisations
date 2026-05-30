NVCC = nvcc
NVCC_FLAGS = -O2
NVCC_DEBUG_FLAGS = -O2 -lineinfo

all: gpu_info vector_add matmul_naive matmul_tiled memory_bandwidth profiling_guide

gpu_info: 00_gpu_info.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vector_add: 01_vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

matmul_naive: 02_matmul_naive.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

matmul_tiled: 03_matmul_tiled.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

memory_bandwidth: 04_memory_bandwidth.cu cuda_utils.h
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

profiling_guide: 05_profiling_guide.cu cuda_utils.h
	$(NVCC) $(NVCC_DEBUG_FLAGS) -o $@ $<

# --- Profiling targets ---
# Nsight Compute: deep kernel analysis (occupancy, memory throughput, stalls)
profile-ncu: profiling_guide
	ncu --set full --target-processes all ./profiling_guide

# Nsight Systems: system-level timeline (kernel launches, memory copies, gaps)
profile-nsys: profiling_guide
	nsys profile --stats=true -o profiling_report ./profiling_guide
	@echo ""
	@echo "Open profiling_report.nsys-rep in Nsight Systems GUI for visual timeline"

# Profile the tiled matmul specifically
profile-matmul: matmul_tiled
	ncu --set full ./matmul_tiled

# Compare naive vs tiled in Nsight
profile-compare: matmul_naive matmul_tiled
	@echo "=== Naive Matmul ==="
	ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,dram__throughput.avg_pct_of_peak_sustained_elapsed ./matmul_naive
	@echo ""
	@echo "=== Tiled Matmul ==="
	ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,dram__throughput.avg_pct_of_peak_sustained_elapsed ./matmul_tiled

clean:
	rm -f gpu_info vector_add matmul_naive matmul_tiled memory_bandwidth profiling_guide
	rm -f *.nsys-rep *.ncu-rep *.sqlite

.PHONY: all clean profile-ncu profile-nsys profile-matmul profile-compare
