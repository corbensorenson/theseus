/*
 * Production-shape CPU/Accelerate weight-gradient control for Theseus.
 *
 * Computes dW = X^T dY for a logical batch 4 x sequence 512 projection.
 * This is an operator mechanics probe only: no checkpoint, corpus row, model
 * claim, or production route is touched.
 */

#define ACCELERATE_NEW_LAPACK
#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>

#include <mach/mach_time.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS (4 * 512)
#define INPUT_CHANNELS 512
#define OUTPUT_CHANNELS 768
#define DEFAULT_WARMUP 8
#define DEFAULT_REPETITIONS 64

static double Seconds(uint64_t ticks) {
    static mach_timebase_info_data_t info;
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)ticks * (double)info.numer /
           (double)info.denom / 1.0e9;
}

static int CompareDouble(const void *left, const void *right) {
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static const char *ThermalStateLabel(NSProcessInfoThermalState state) {
    switch (state) {
        case NSProcessInfoThermalStateNominal: return "nominal";
        case NSProcessInfoThermalStateFair: return "fair";
        case NSProcessInfoThermalStateSerious: return "serious";
        case NSProcessInfoThermalStateCritical: return "critical";
    }
    return "unknown";
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        bool singleThreaded = false;
        int warmup = DEFAULT_WARMUP;
        int repetitions = DEFAULT_REPETITIONS;
        for (int index = 1; index < argc; ++index) {
            if (strcmp(argv[index], "--single-threaded") == 0) {
                singleThreaded = true;
            } else if (
                strcmp(argv[index], "--warmup") == 0 &&
                index + 1 < argc
            ) {
                warmup = atoi(argv[++index]);
            } else if (
                strcmp(argv[index], "--repetitions") == 0 &&
                index + 1 < argc
            ) {
                repetitions = atoi(argv[++index]);
            } else {
                fprintf(stderr, "invalid_argument:%s\n", argv[index]);
                return 2;
            }
        }
        if (warmup < 1 || repetitions < 2) {
            fprintf(stderr, "invalid_repetition_contract\n");
            return 2;
        }

        float *x = NULL;
        float *dy = NULL;
        float *dw = NULL;
        if (
            posix_memalign(
                (void **)&x, 64,
                (size_t)ROWS * INPUT_CHANNELS * sizeof(float)
            ) ||
            posix_memalign(
                (void **)&dy, 64,
                (size_t)ROWS * OUTPUT_CHANNELS * sizeof(float)
            ) ||
            posix_memalign(
                (void **)&dw, 64,
                (size_t)INPUT_CHANNELS * OUTPUT_CHANNELS * sizeof(float)
            )
        ) {
            fprintf(stderr, "allocation_failed\n");
            free(x);
            free(dy);
            free(dw);
            return 3;
        }
        for (size_t index = 0;
             index < (size_t)ROWS * INPUT_CHANNELS; ++index) {
            x[index] = sinf((float)(index % 4093) * 0.0031f) * 0.125f;
        }
        for (size_t index = 0;
             index < (size_t)ROWS * OUTPUT_CHANNELS; ++index) {
            dy[index] = cosf((float)(index % 4099) * 0.0029f) * 0.0625f;
        }

        BLASSetThreading(
            singleThreaded
                ? BLAS_THREADING_SINGLE_THREADED
                : BLAS_THREADING_MULTI_THREADED
        );
        const enum BLAS_THREADING observedThreading = BLASGetThreading();
        for (int iteration = 0; iteration < warmup; ++iteration) {
            cblas_sgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                INPUT_CHANNELS,
                OUTPUT_CHANNELS,
                ROWS,
                1.0f,
                x,
                INPUT_CHANNELS,
                dy,
                OUTPUT_CHANNELS,
                0.0f,
                dw,
                OUTPUT_CHANNELS
            );
        }

        double *durations = calloc((size_t)repetitions, sizeof(double));
        if (!durations) {
            fprintf(stderr, "duration_allocation_failed\n");
            free(x);
            free(dy);
            free(dw);
            return 3;
        }
        for (int iteration = 0; iteration < repetitions; ++iteration) {
            const uint64_t started = mach_continuous_time();
            cblas_sgemm(
                CblasRowMajor,
                CblasTrans,
                CblasNoTrans,
                INPUT_CHANNELS,
                OUTPUT_CHANNELS,
                ROWS,
                1.0f,
                x,
                INPUT_CHANNELS,
                dy,
                OUTPUT_CHANNELS,
                0.0f,
                dw,
                OUTPUT_CHANNELS
            );
            durations[iteration] = Seconds(
                mach_continuous_time() - started
            );
        }

        size_t mismatchCount = 0;
        float maximumAbsoluteDelta = 0.0f;
        for (int sample = 0; sample < 64; ++sample) {
            const int inputChannel =
                (sample * 193) % INPUT_CHANNELS;
            const int outputChannel =
                (sample * 389) % OUTPUT_CHANNELS;
            float reference = 0.0f;
            for (int row = 0; row < ROWS; ++row) {
                reference = fmaf(
                    x[(size_t)row * INPUT_CHANNELS + inputChannel],
                    dy[(size_t)row * OUTPUT_CHANNELS + outputChannel],
                    reference
                );
            }
            const float observed =
                dw[(size_t)inputChannel * OUTPUT_CHANNELS + outputChannel];
            const float delta = fabsf(observed - reference);
            if (delta > 0.001f) ++mismatchCount;
            maximumAbsoluteDelta = fmaxf(maximumAbsoluteDelta, delta);
        }

        double mean = 0.0;
        for (int iteration = 0; iteration < repetitions; ++iteration) {
            mean += durations[iteration];
        }
        mean /= repetitions;
        qsort(
            durations,
            (size_t)repetitions,
            sizeof(double),
            CompareDouble
        );
        const double median = (
            repetitions % 2
                ? durations[repetitions / 2]
                : (
                    durations[repetitions / 2 - 1] +
                    durations[repetitions / 2]
                ) / 2.0
        );
        int p95Index = (int)ceil(repetitions * 0.95) - 1;
        if (p95Index < 0) p95Index = 0;
        if (p95Index >= repetitions) p95Index = repetitions - 1;
        const double operations =
            2.0 * ROWS * INPUT_CHANNELS * OUTPUT_CHANNELS;
        NSProcessInfo *processInfo = [NSProcessInfo processInfo];
        printf(
            "{"
            "\"policy\":\"project_theseus_cpu_accelerate_dw_probe_v1\","
            "\"trigger_state\":\"%s\","
            "\"shape\":{\"rows\":%d,\"input_channels\":%d,"
            "\"output_channels\":%d},"
            "\"threading\":\"%s\","
            "\"blas_threading_value\":%u,"
            "\"warmup\":%d,\"repetitions\":%d,"
            "\"mean_milliseconds\":%.9g,"
            "\"median_milliseconds\":%.9g,"
            "\"p95_milliseconds\":%.9g,"
            "\"gflops\":%.9g,"
            "\"sample_count\":64,"
            "\"mismatch_count\":%zu,"
            "\"maximum_absolute_delta\":%.9g,"
            "\"thermal_state\":\"%s\","
            "\"low_power_mode\":%s,"
            "\"public_benchmark_rows_read\":0,"
            "\"external_inference_calls\":0,"
            "\"canonical_checkpoint_mutated\":false,"
            "\"claim_scope\":\"FP32 Accelerate dW operator mechanics only; "
            "no joined training or speedup claim.\""
            "}\n",
            mismatchCount == 0 ? "GREEN" : "RED",
            ROWS,
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
            singleThreaded ? "single" : "accelerate_managed_multi",
            (unsigned int)observedThreading,
            warmup,
            repetitions,
            mean * 1000.0,
            median * 1000.0,
            durations[p95Index] * 1000.0,
            operations / mean / 1.0e9,
            mismatchCount,
            maximumAbsoluteDelta,
            ThermalStateLabel(processInfo.thermalState),
            processInfo.lowPowerModeEnabled ? "true" : "false"
        );

        free(durations);
        free(x);
        free(dy);
        free(dw);
        return mismatchCount == 0 ? 0 : 1;
    }
}
