/* Minimal OpenMP shim.
 *
 * Apple clang ships without libomp, so a discovered kernel that includes
 * <omp.h> and uses the runtime query functions cannot be compiled there.  This
 * header lets such a program build single-threaded: the `#pragma omp`
 * directives are simply ignored by the compiler, and the two runtime functions
 * report a one-thread team.  Results are unchanged; only throughput differs.
 */
#ifndef DREAMRSI_OMP_SHIM_H
#define DREAMRSI_OMP_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
static inline void omp_set_num_threads(int n) { (void)n; }
static inline double omp_get_wtime(void) { return 0.0; }

#ifdef __cplusplus
}
#endif

#endif /* DREAMRSI_OMP_SHIM_H */
