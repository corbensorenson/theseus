#ifndef THESEUS_ANE_METAL_SURFACE_CONTRACT_H
#define THESEUS_ANE_METAL_SURFACE_CONTRACT_H

#include <stdint.h>

/* ABI-only custody metadata. Tensor storage remains owned by IOSurface. */
#define THESEUS_ANE_METAL_SURFACE_ABI_VERSION 1u
#define THESEUS_READER_METAL (1u << 0)
#define THESEUS_READER_ANE (1u << 1)

typedef enum {
    THESEUS_WRITER_NONE = 0,
    THESEUS_WRITER_HOST = 1,
    THESEUS_WRITER_METAL = 2,
    THESEUS_WRITER_ANE = 3
} theseus_surface_writer;

typedef enum {
    THESEUS_DTYPE_FLOAT16 = 1,
    THESEUS_DTYPE_FLOAT32 = 2
} theseus_surface_dtype;

typedef struct {
    uint32_t abi_version;
    uint32_t iosurface_id;
    uint32_t dtype;
    uint32_t rank;
    uint64_t shape[4];
    uint64_t strides_bytes[4];
    uint64_t generation;
    uint32_t active_readers;
    uint32_t active_writer;
    uint64_t checkpoint_generation;
} theseus_ane_metal_surface;

/*
 * Contract:
 * - active_writer != NONE requires active_readers == 0.
 * - METAL and ANE reader bits may coexist only for an immutable generation.
 * - generation increments after a successful exclusive write.
 * - checkpoint_generation changes only after join, parity, and replay gates.
 * - a process/OS/private-API failure invalidates the candidate generation and
 *   falls back before checkpoint publication.
 */

#endif
