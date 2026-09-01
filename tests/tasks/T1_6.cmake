# T1.6 — adjacency table builder / mmap reader acceptance test.
# Needs the GPU only to (re)build data/adj.u32 when absent and for the
# to_device upload check; everything else is a CPU pass over the mmap.
# Working directory is the source root (kiss_add_test), so `data` resolves.
kiss_add_test(NAME test_adjacency SOURCES test_adjacency.cpp GPU LIBS kiss_cuda TIMEOUT 900 ARGS data)
