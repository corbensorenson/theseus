/*
 * Exact generation-tagged Project Theseus decoder-block transaction.
 *
 * One process joins:
 *   - compile-once ANE attention forward,
 *   - native Metal/Accelerate block remainder and scalar objective,
 *   - compile-once ANE attention backward,
 *   - single-thread Accelerate FP32 attention parameter gradients,
 *   - one combined global norm, clip, and FP32 AdamW publication.
 *
 * The embedded sources remain independently executable qualification owners.
 */

#define THESEUS_ATTENTION_BACKWARD_ENTRY \
    theseus_embedded_attention_backward_main
#include "ane_exact_attention_backward.m"
#undef THESEUS_ATTENTION_BACKWARD_ENTRY

/* Isolate the remainder source's translation-unit-local names. */
#undef DIM
#undef OUTPUT_ELEMENTS
#define Timebase RemainderTimebase
#define Milliseconds RemainderMilliseconds
#define Comparison RemainderComparison
#define AllFinite RemainderAllFinite
#define main theseus_embedded_remainder_main
#include "exact_decoder_block_remainder.m"
#undef main
#undef AllFinite
#undef Comparison
#undef Milliseconds
#undef Timebase

#include "ane_metal_surface_contract.h"

#define ATTENTION_PARAMETER_ELEMENTS \
    ((size_t)DIM + (size_t)DIM * QUERY_DIM + \
     (size_t)DIM * KV_DIM + (size_t)DIM * KV_DIM)
#define TOTAL_BLOCK_PARAMETERS \
    (ATTENTION_PARAMETER_ELEMENTS + PARAMETER_ELEMENTS)
#define ATTENTION_NORM_OFFSET 0
#define ATTENTION_Q_OFFSET ((size_t)DIM)
#define ATTENTION_K_OFFSET (ATTENTION_Q_OFFSET + (size_t)DIM * QUERY_DIM)
#define ATTENTION_V_OFFSET (ATTENTION_K_OFFSET + (size_t)DIM * KV_DIM)
#define COMBINED_PARTIALS \
    ((TOTAL_BLOCK_PARAMETERS + THREADS - 1) / THREADS)

typedef struct {
    id<MTLComputePipelineState> squareCombined;
    id<MTLComputePipelineState> reduceCombined;
    id<MTLComputePipelineState> updateBuffer;
    id<MTLBuffer> attentionWeight;
    id<MTLBuffer> attentionFirst;
    id<MTLBuffer> attentionSecond;
    id<MTLBuffer> attentionGradient;
    id<MTLBuffer> combinedPartials;
    id<MTLBuffer> combinedNorm;
} JoinedUpdate;

typedef struct {
    id forwardModel;
    NSString *forwardDirectory;
    id backwardModel;
    NSString *backwardDirectory;
    IOSurfaceRef forwardInput;
    IOSurfaceRef forwardOutput;
    IOSurfaceRef backwardInput;
    IOSurfaceRef backwardOutput;
} AttentionRuntime;

typedef struct {
    double forwardANE;
    double remainder;
    double backwardANE;
    double attentionGradient;
    double unifiedUpdate;
    double joined;
    float loss;
    float gradientNorm;
} JoinedReceipt;

typedef struct {
    theseus_ane_metal_surface forwardInput;
    theseus_ane_metal_surface forwardOutput;
    theseus_ane_metal_surface backwardInput;
    theseus_ane_metal_surface backwardOutput;
} JoinedCustody;

static theseus_ane_metal_surface JoinCustodyForSurface(
    IOSurfaceRef surface, uint64_t channels, uint64_t spatial
) {
    theseus_ane_metal_surface custody = {0};
    custody.abi_version = THESEUS_ANE_METAL_SURFACE_ABI_VERSION;
    custody.iosurface_id = IOSurfaceGetID(surface);
    custody.dtype = THESEUS_DTYPE_FLOAT16;
    custody.rank = 4;
    custody.shape[0] = 1;
    custody.shape[1] = channels;
    custody.shape[2] = 1;
    custody.shape[3] = spatial;
    custody.strides_bytes[3] = sizeof(_Float16);
    custody.strides_bytes[2] = spatial * sizeof(_Float16);
    custody.strides_bytes[1] = spatial * sizeof(_Float16);
    custody.strides_bytes[0] = channels * spatial * sizeof(_Float16);
    return custody;
}

static NSString *JoinedUpdateSource(void) {
    return [NSString stringWithFormat:
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "constant uint ATTENTION_COUNT=%zu;\n"
         "constant uint REMAINDER_COUNT=%zu;\n"
         "constant uint TOTAL_COUNT=%zu;\n"
         "kernel void square_combined("
         " device const float* attention [[buffer(0)]],"
         " device const float* remainder [[buffer(1)]],"
         " device float* partials [[buffer(2)]],"
         " uint tid [[thread_position_in_grid]],"
         " uint lane [[thread_index_in_threadgroup]],"
         " uint group [[threadgroup_position_in_grid]]) {"
         " threadgroup float scratch[256]; float v=0.0f;"
         " if(tid<TOTAL_COUNT)v=tid<ATTENTION_COUNT?attention[tid]"
         " :remainder[tid-ATTENTION_COUNT];"
         " scratch[lane]=v*v;"
         " threadgroup_barrier(mem_flags::mem_threadgroup);"
         " for(uint s=128;s>0;s>>=1){if(lane<s)scratch[lane]+=scratch[lane+s];"
         " threadgroup_barrier(mem_flags::mem_threadgroup);}"
         " if(lane==0)partials[group]=scratch[0]; }\n"
         "kernel void reduce_combined("
         " device const float* partials [[buffer(0)]],"
         " device float* norm [[buffer(1)]],"
         " uint lane [[thread_index_in_threadgroup]]) {"
         " threadgroup float scratch[256]; float sum=0.0f;"
         " for(uint i=lane;i<%zu;i+=256)sum+=partials[i];"
         " scratch[lane]=sum;"
         " threadgroup_barrier(mem_flags::mem_threadgroup);"
         " for(uint s=128;s>0;s>>=1){if(lane<s)scratch[lane]+=scratch[lane+s];"
         " threadgroup_barrier(mem_flags::mem_threadgroup);}"
         " if(lane==0)norm[0]=sqrt(scratch[0]); }\n"
         "kernel void update_buffer("
         " device float* weight [[buffer(0)]],"
         " device float* first [[buffer(1)]],"
         " device float* second [[buffer(2)]],"
         " device const float* gradient [[buffer(3)]],"
         " device const float* norm [[buffer(4)]],"
         " constant uint& count [[buffer(5)]],"
         " constant float& lr [[buffer(6)]],"
         " constant float& b1 [[buffer(7)]],"
         " constant float& b2 [[buffer(8)]],"
         " constant float& eps [[buffer(9)]],"
         " constant float& decay [[buffer(10)]],"
         " constant float& clip [[buffer(11)]],"
         " uint tid [[thread_position_in_grid]]) {"
         " if(tid>=count)return;"
         " float scale=min(1.0f,clip/max(norm[0],1.0e-12f));"
         " float g=gradient[tid]*scale;"
         " float m=b1*first[tid]+(1.0f-b1)*g;"
         " float v=b2*second[tid]+(1.0f-b2)*g*g;"
         " weight[tid]=weight[tid]*(1.0f-lr*decay)"
         " -lr*m/(sqrt(v)+eps); first[tid]=m; second[tid]=v; }\n",
        (size_t)ATTENTION_PARAMETER_ELEMENTS,
        (size_t)PARAMETER_ELEMENTS,
        (size_t)TOTAL_BLOCK_PARAMETERS,
        (size_t)COMBINED_PARTIALS];
}

static BOOL BuildJoinedUpdate(
    MetalRemainder *metal, JoinedUpdate *joined, NSError **error
) {
    id<MTLLibrary> library = [metal->device
        newLibraryWithSource:JoinedUpdateSource() options:nil error:error];
    if (!library) return NO;
    joined->squareCombined =
        Pipeline(metal->device, library, @"square_combined", error);
    joined->reduceCombined =
        Pipeline(metal->device, library, @"reduce_combined", error);
    joined->updateBuffer =
        Pipeline(metal->device, library, @"update_buffer", error);
    joined->attentionWeight =
        Buffer(metal->device, ATTENTION_PARAMETER_ELEMENTS, NO);
    joined->attentionFirst =
        Buffer(metal->device, ATTENTION_PARAMETER_ELEMENTS, YES);
    joined->attentionSecond =
        Buffer(metal->device, ATTENTION_PARAMETER_ELEMENTS, YES);
    joined->attentionGradient =
        Buffer(metal->device, ATTENTION_PARAMETER_ELEMENTS, YES);
    joined->combinedPartials =
        Buffer(metal->device, COMBINED_PARTIALS, YES);
    joined->combinedNorm = Buffer(metal->device, 1, YES);
    return joined->squareCombined && joined->reduceCombined &&
        joined->updateBuffer && joined->attentionWeight &&
        joined->attentionFirst && joined->attentionSecond &&
        joined->attentionGradient && joined->combinedPartials &&
        joined->combinedNorm;
}

static BOOL EncodeJoinedUpdate(
    MetalRemainder *metal, MetalBuffers *remainder, JoinedUpdate *joined,
    NSError **error
) {
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> square = [command computeCommandEncoder];
    [square setComputePipelineState:joined->squareCombined];
    [square setBuffer:joined->attentionGradient offset:0 atIndex:0];
    [square setBuffer:remainder->gradient offset:0 atIndex:1];
    [square setBuffer:joined->combinedPartials offset:0 atIndex:2];
    [square dispatchThreads:MTLSizeMake(COMBINED_PARTIALS * THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [square endEncoding];
    id<MTLComputeCommandEncoder> reduce = [command computeCommandEncoder];
    [reduce setComputePipelineState:joined->reduceCombined];
    [reduce setBuffer:joined->combinedPartials offset:0 atIndex:0];
    [reduce setBuffer:joined->combinedNorm offset:0 atIndex:1];
    [reduce dispatchThreads:MTLSizeMake(THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [reduce endEncoding];
    float lr = 3.0e-5f, b1 = 0.9f, b2 = 0.999f;
    float epsilon = 1.0e-8f, decay = 0.01f, clip = 1.0f;
    uint32_t counts[2] = {
        (uint32_t)ATTENTION_PARAMETER_ELEMENTS,
        (uint32_t)PARAMETER_ELEMENTS,
    };
    id<MTLBuffer> weights[2] = {
        joined->attentionWeight, remainder->weight,
    };
    id<MTLBuffer> firsts[2] = {
        joined->attentionFirst, remainder->first,
    };
    id<MTLBuffer> seconds[2] = {
        joined->attentionSecond, remainder->second,
    };
    id<MTLBuffer> gradients[2] = {
        joined->attentionGradient, remainder->gradient,
    };
    for (int owner = 0; owner < 2; ++owner) {
        id<MTLComputeCommandEncoder> update = [command computeCommandEncoder];
        [update setComputePipelineState:joined->updateBuffer];
        [update setBuffer:weights[owner] offset:0 atIndex:0];
        [update setBuffer:firsts[owner] offset:0 atIndex:1];
        [update setBuffer:seconds[owner] offset:0 atIndex:2];
        [update setBuffer:gradients[owner] offset:0 atIndex:3];
        [update setBuffer:joined->combinedNorm offset:0 atIndex:4];
        [update setBytes:&counts[owner] length:sizeof(uint32_t) atIndex:5];
        [update setBytes:&lr length:sizeof(float) atIndex:6];
        [update setBytes:&b1 length:sizeof(float) atIndex:7];
        [update setBytes:&b2 length:sizeof(float) atIndex:8];
        [update setBytes:&epsilon length:sizeof(float) atIndex:9];
        [update setBytes:&decay length:sizeof(float) atIndex:10];
        [update setBytes:&clip length:sizeof(float) atIndex:11];
        [update dispatchThreads:MTLSizeMake(counts[owner], 1, 1)
            threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
        [update endEncoding];
    }
    return Complete(command, error);
}

static BOOL BuildAttentionRuntime(
    AttentionRuntime *runtime, NSError **error
) {
    float *cosine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
    float *sine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
    float *mask = malloc(SEQUENCE * SEQUENCE * sizeof(float));
    BuildRoPE(cosine, sine);
    for (int row = 0; row < SEQUENCE; ++row)
        for (int column = 0; column < SEQUENCE; ++column)
            mask[row * SEQUENCE + column] =
                column <= row ? 0.0f : -10000.0f;
    NSString *forwardDirectory = nil;
    NSString *backwardDirectory = nil;
    runtime->forwardModel = CompileModel(
        ExactAttentionMIL(),
        @{
            @"@model_path/weights/cos.bin" :
                @{@"offset": @0, @"data":
                    BuildWeightBlob(cosine, SEQUENCE * HEAD_DIM)},
            @"@model_path/weights/sin.bin" :
                @{@"offset": @0, @"data":
                    BuildWeightBlob(sine, SEQUENCE * HEAD_DIM)},
            @"@model_path/weights/mask.bin" :
                @{@"offset": @0, @"data":
                    BuildWeightBlob(mask, SEQUENCE * SEQUENCE)},
        },
        &forwardDirectory, error);
    runtime->backwardModel = CompileModel(
        ExactAttentionBackwardMIL(),
        @{
            @"@model_path/weights/mask.bin" :
                @{@"offset": @0, @"data":
                    BuildWeightBlob(mask, SEQUENCE * SEQUENCE)},
        },
        &backwardDirectory, error);
    runtime->forwardDirectory = forwardDirectory;
    runtime->backwardDirectory = backwardDirectory;
    free(cosine); free(sine); free(mask);
    runtime->forwardInput = CreateHalfSurface(INPUT_ELEMENTS);
    runtime->forwardOutput = CreateHalfSurface(
        (size_t)OUTPUT_CHANNELS * SEQUENCE);
    runtime->backwardInput = CreateHalfSurface(BACKWARD_INPUT_ELEMENTS);
    runtime->backwardOutput = CreateHalfSurface(BACKWARD_OUTPUT_ELEMENTS);
    return runtime->forwardModel && runtime->backwardModel &&
        runtime->forwardInput && runtime->forwardOutput &&
        runtime->backwardInput && runtime->backwardOutput;
}

static void PackForwardInput(
    AttentionRuntime *runtime, const float *hidden,
    const float *normScale, const float *wq,
    const float *wk, const float *wv
) {
    IOSurfaceLock(runtime->forwardInput, 0, NULL);
    _Float16 *packed = IOSurfaceGetBaseAddress(runtime->forwardInput);
    memset(packed, 0, INPUT_ELEMENTS * sizeof(_Float16));
    int qOffset = SEQUENCE + NORM_SCALE_SPAN;
    int kOffset = qOffset + QUERY_DIM;
    int vOffset = kOffset + KV_DIM;
    for (int channel = 0; channel < DIM; ++channel) {
        for (int position = 0; position < SEQUENCE; ++position)
            packed[(size_t)channel * PACKED_SPATIAL + position] =
                (_Float16)hidden[(size_t)position * DIM + channel];
        packed[(size_t)channel * PACKED_SPATIAL + SEQUENCE] =
            (_Float16)normScale[channel];
        for (int out = 0; out < QUERY_DIM; ++out)
            packed[(size_t)channel * PACKED_SPATIAL + qOffset + out] =
                (_Float16)wq[(size_t)channel * QUERY_DIM + out];
        for (int out = 0; out < KV_DIM; ++out) {
            packed[(size_t)channel * PACKED_SPATIAL + kOffset + out] =
                (_Float16)wk[(size_t)channel * KV_DIM + out];
            packed[(size_t)channel * PACKED_SPATIAL + vOffset + out] =
                (_Float16)wv[(size_t)channel * KV_DIM + out];
        }
    }
    IOSurfaceUnlock(runtime->forwardInput, 0, NULL);
}

static void ReadForwardTaps(
    AttentionRuntime *runtime, float *attended, float *query,
    float *compactKey, float *compactValue, float *normalized
) {
    IOSurfaceLock(runtime->forwardOutput, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *output = IOSurfaceGetBaseAddress(runtime->forwardOutput);
    size_t offset = 0;
    ReadChannelMajorToRows(attended, output + offset, QUERY_DIM);
    offset += (size_t)QUERY_DIM * SEQUENCE;
    ReadChannelMajorToRows(query, output + offset, QUERY_DIM);
    offset += (size_t)QUERY_DIM * SEQUENCE;
    ReadChannelMajorToRows(compactKey, output + offset, KV_DIM);
    offset += (size_t)KV_DIM * SEQUENCE;
    ReadChannelMajorToRows(compactValue, output + offset, KV_DIM);
    offset += (size_t)KV_DIM * SEQUENCE;
    ReadChannelMajorToRows(normalized, output + offset, DIM);
    IOSurfaceUnlock(runtime->forwardOutput, kIOSurfaceLockReadOnly, NULL);
}

static void PackBackwardInput(
    AttentionRuntime *runtime, const float *query,
    const float *compactKey, const float *compactValue,
    const float *dAttended
) {
    float *key = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    float *value = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    TileContiguousKV(key, compactKey);
    TileContiguousKV(value, compactValue);
    IOSurfaceLock(runtime->backwardInput, 0, NULL);
    _Float16 *packed = IOSurfaceGetBaseAddress(runtime->backwardInput);
    PackChannelMajor(packed, query, QUERY_DIM);
    PackChannelMajor(
        packed + (size_t)QUERY_DIM * SEQUENCE, key, QUERY_DIM);
    PackChannelMajor(
        packed + (size_t)2 * QUERY_DIM * SEQUENCE, value, QUERY_DIM);
    PackChannelMajor(
        packed + (size_t)3 * QUERY_DIM * SEQUENCE, dAttended, QUERY_DIM);
    IOSurfaceUnlock(runtime->backwardInput, 0, NULL);
    free(key); free(value);
}

static void CloseAttentionGradients(
    AttentionRuntime *runtime, const float *hidden,
    const float *normScale, const float *normalized,
    const float *wq, const float *wk, const float *wv,
    float *attentionGradient, float *dHidden
) {
    float *dq = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    float *dkTiled = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    float *dvTiled = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    IOSurfaceLock(runtime->backwardOutput, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *output = IOSurfaceGetBaseAddress(runtime->backwardOutput);
    ReadChannelMajorToRows(dq, output, QUERY_DIM);
    ReadChannelMajorToRows(
        dkTiled, output + (size_t)QUERY_DIM * SEQUENCE, QUERY_DIM);
    ReadChannelMajorToRows(
        dvTiled, output + (size_t)2 * QUERY_DIM * SEQUENCE, QUERY_DIM);
    IOSurfaceUnlock(runtime->backwardOutput, kIOSurfaceLockReadOnly, NULL);
    float *dk = malloc((size_t)SEQUENCE * KV_DIM * sizeof(float));
    float *dv = malloc((size_t)SEQUENCE * KV_DIM * sizeof(float));
    ReduceContiguousKV(dk, dkTiled);
    ReduceContiguousKV(dv, dvTiled);
    float *cosine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
    float *sine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
    BuildRoPE(cosine, sine);
    InverseSplitHalfRoPEGradient(dq, QUERY_HEADS, cosine, sine);
    InverseSplitHalfRoPEGradient(dk, KV_HEADS, cosine, sine);
    float *dNormalized =
        malloc((size_t)SEQUENCE * DIM * sizeof(float));
    ProjectionGradientsAccelerate(
        normalized, wq, wk, wv, dq, dk, dv,
        attentionGradient + ATTENTION_Q_OFFSET,
        attentionGradient + ATTENTION_K_OFFSET,
        attentionGradient + ATTENTION_V_OFFSET,
        dNormalized);
    RMSNormBackwardReference(
        hidden, normScale, dNormalized, dHidden,
        attentionGradient + ATTENTION_NORM_OFFSET);
    free(dq); free(dkTiled); free(dvTiled); free(dk); free(dv);
    free(cosine); free(sine); free(dNormalized);
}

static BOOL RunJoinedStep(
    AttentionRuntime *attention, MetalRemainder *metal,
    MetalBuffers *remainder, JoinedUpdate *joined,
    JoinedCustody *custody, const float *hidden, float *dHiddenTotal,
    uint64_t generation, JoinedReceipt *receipt, NSError **error
) {
    uint64_t joinedStarted = mach_absolute_time();
    float *attentionWeight = joined->attentionWeight.contents;
    float *normScale = attentionWeight + ATTENTION_NORM_OFFSET;
    float *wq = attentionWeight + ATTENTION_Q_OFFSET;
    float *wk = attentionWeight + ATTENTION_K_OFFSET;
    float *wv = attentionWeight + ATTENTION_V_OFFSET;
    PackForwardInput(attention, hidden, normScale, wq, wk, wv);
    custody->forwardInput.active_writer = THESEUS_WRITER_NONE;
    custody->forwardInput.active_readers = THESEUS_READER_ANE;
    custody->forwardInput.generation = generation;
    custody->forwardOutput.active_writer = THESEUS_WRITER_ANE;
    uint64_t started = mach_absolute_time();
    if (!Evaluate(
            attention->forwardModel, attention->forwardInput,
            attention->forwardOutput, error)) return NO;
    custody->forwardInput.active_readers = 0;
    custody->forwardOutput.active_writer = THESEUS_WRITER_NONE;
    custody->forwardOutput.generation = generation;
    receipt->forwardANE = Milliseconds(mach_absolute_time() - started);
    float *attended = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    float *query = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    float *key = malloc((size_t)SEQUENCE * KV_DIM * sizeof(float));
    float *value = malloc((size_t)SEQUENCE * KV_DIM * sizeof(float));
    float *attentionNormalized =
        malloc((size_t)SEQUENCE * DIM * sizeof(float));
    ReadForwardTaps(
        attention, attended, query, key, value, attentionNormalized);
    float *remainderNormalized =
        malloc((size_t)SEQUENCE * DIM * sizeof(float));
    float *remainderDNormalized =
        malloc((size_t)SEQUENCE * DIM * sizeof(float));
    float *dAfter = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    float *dAttended = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    float *dHiddenDirect = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    StepReceipt remainderReceipt = {0};
    started = mach_absolute_time();
    if (!RunStep(
            metal, remainder, hidden, attended, remainderNormalized,
            remainderDNormalized, dAfter, dAttended, dHiddenDirect,
            NO, &remainderReceipt, error)) return NO;
    receipt->remainder = Milliseconds(mach_absolute_time() - started);
    PackBackwardInput(attention, query, key, value, dAttended);
    custody->backwardInput.active_writer = THESEUS_WRITER_NONE;
    custody->backwardInput.active_readers = THESEUS_READER_ANE;
    custody->backwardInput.generation = generation;
    custody->backwardOutput.active_writer = THESEUS_WRITER_ANE;
    started = mach_absolute_time();
    if (!Evaluate(
            attention->backwardModel, attention->backwardInput,
            attention->backwardOutput, error)) return NO;
    custody->backwardInput.active_readers = 0;
    custody->backwardOutput.active_writer = THESEUS_WRITER_NONE;
    custody->backwardOutput.generation = generation;
    receipt->backwardANE = Milliseconds(mach_absolute_time() - started);
    float *dHiddenAttention =
        malloc((size_t)SEQUENCE * DIM * sizeof(float));
    started = mach_absolute_time();
    CloseAttentionGradients(
        attention, hidden, normScale, attentionNormalized, wq, wk, wv,
        joined->attentionGradient.contents, dHiddenAttention);
    receipt->attentionGradient =
        Milliseconds(mach_absolute_time() - started);
    for (size_t index = 0; index < (size_t)SEQUENCE * DIM; ++index)
        dHiddenTotal[index] = dHiddenDirect[index] + dHiddenAttention[index];
    started = mach_absolute_time();
    if (!EncodeJoinedUpdate(metal, remainder, joined, error)) return NO;
    receipt->unifiedUpdate = Milliseconds(mach_absolute_time() - started);
    receipt->loss = ((float *)remainder->scalarLoss.contents)[0];
    receipt->gradientNorm = ((float *)joined->combinedNorm.contents)[0];
    receipt->joined = Milliseconds(mach_absolute_time() - joinedStarted);
    if (custody->forwardInput.generation != generation ||
        custody->forwardOutput.generation != generation ||
        custody->backwardInput.generation != generation ||
        custody->backwardOutput.generation != generation) return NO;
    free(attended); free(query); free(key); free(value);
    free(attentionNormalized); free(remainderNormalized);
    free(remainderDNormalized); free(dAfter); free(dAttended);
    free(dHiddenDirect); free(dHiddenAttention);
    return YES;
}

static void InitializeJoinedValues(
    MetalBuffers *remainder, JoinedUpdate *joined,
    float *hidden
) {
    float *attentionWeight = joined->attentionWeight.contents;
    float *normScale = attentionWeight + ATTENTION_NORM_OFFSET;
    float *wq = attentionWeight + ATTENTION_Q_OFFSET;
    float *wk = attentionWeight + ATTENTION_K_OFFSET;
    float *wv = attentionWeight + ATTENTION_V_OFFSET;
    float *dummyNormalized = malloc((size_t)SEQUENCE * DIM * sizeof(float));
    float *dummyQ = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    float *dummyK = malloc((size_t)SEQUENCE * KV_DIM * sizeof(float));
    float *dummyV = malloc((size_t)SEQUENCE * KV_DIM * sizeof(float));
    float *dummyUpstream = malloc((size_t)SEQUENCE * QUERY_DIM * sizeof(float));
    BuildCoherentInputs(
        hidden, normScale, dummyNormalized, wq, wk, wv,
        dummyQ, dummyK, dummyV, dummyUpstream);
    float *remainderWeight = remainder->weight.contents;
    float *target = remainder->target.contents;
    float *mask = remainder->mask.contents;
    for (size_t i = 0; i < PARAMETER_ELEMENTS; ++i)
        remainderWeight[i] = sinf((float)(i % 887) * 0.005f) * 0.02f;
    for (int i = 0; i < DIM; ++i)
        remainderWeight[NORM_OFFSET + i] =
            1.0f + sinf(i * 0.019f) * 0.01f;
    for (size_t i = 0; i < (size_t)SEQUENCE * DIM; ++i)
        target[i] = sinf((float)(i % 953) * 0.007f) * 0.125f;
    for (int row = 0; row < SEQUENCE; ++row)
        mask[row] = row < SEQUENCE - 3 ? 1.0f : 0.0f;
    free(dummyNormalized); free(dummyQ); free(dummyK); free(dummyV);
    free(dummyUpstream);
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&Timebase);
        mach_timebase_info(&RemainderTimebase);
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        if (!LoadPrivateANE()) return 2;
        NSError *error = nil;
        MetalRemainder metal = {0};
        MetalBuffers remainder = {0};
        JoinedUpdate joined = {0};
        AttentionRuntime attention = {0};
        if (!BuildMetal(&metal, &error) ||
            !BuildBuffers(&metal, &remainder) ||
            !BuildJoinedUpdate(&metal, &joined, &error) ||
            !BuildAttentionRuntime(&attention, &error)) {
            fprintf(stderr, "joined_initialization_failed:%s\n",
                    error.description.UTF8String);
            return 3;
        }
        float *hidden = malloc((size_t)SEQUENCE * DIM * sizeof(float));
        float *dHidden = malloc((size_t)SEQUENCE * DIM * sizeof(float));
        JoinedCustody custody = {0};
        custody.forwardInput = JoinCustodyForSurface(
            attention.forwardInput, DIM, PACKED_SPATIAL);
        custody.forwardOutput = JoinCustodyForSurface(
            attention.forwardOutput, OUTPUT_CHANNELS, SEQUENCE);
        custody.backwardInput = JoinCustodyForSurface(
            attention.backwardInput, BACKWARD_INPUT_CHANNELS, SEQUENCE);
        custody.backwardOutput = JoinCustodyForSurface(
            attention.backwardOutput, BACKWARD_OUTPUT_CHANNELS, SEQUENCE);
        InitializeJoinedValues(&remainder, &joined, hidden);
        size_t attentionBytes =
            ATTENTION_PARAMETER_ELEMENTS * sizeof(float);
        size_t remainderBytes = PARAMETER_ELEMENTS * sizeof(float);
        float *initialAttention = malloc(attentionBytes);
        float *initialRemainder = malloc(remainderBytes);
        memcpy(initialAttention, joined.attentionWeight.contents, attentionBytes);
        memcpy(initialRemainder, remainder.weight.contents, remainderBytes);
        float *firstAttention = malloc(attentionBytes);
        float *firstRemainder = malloc(remainderBytes);
        float *firstAttentionM = malloc(attentionBytes);
        float *firstAttentionV = malloc(attentionBytes);
        float *firstRemainderM = malloc(remainderBytes);
        float *firstRemainderV = malloc(remainderBytes);
        float *firstAttentionGradient = malloc(attentionBytes);
        float *firstRemainderGradient = malloc(remainderBytes);
        JoinedReceipt first = {0};
        if (!RunJoinedStep(
                &attention, &metal, &remainder, &joined, &custody,
                hidden, dHidden,
                0, &first, &error)) {
            fprintf(stderr, "joined_first_step_failed:%s\n",
                    error.description.UTF8String);
            return 4;
        }
        memcpy(firstAttention, joined.attentionWeight.contents, attentionBytes);
        memcpy(firstRemainder, remainder.weight.contents, remainderBytes);
        memcpy(firstAttentionM, joined.attentionFirst.contents, attentionBytes);
        memcpy(firstAttentionV, joined.attentionSecond.contents, attentionBytes);
        memcpy(firstRemainderM, remainder.first.contents, remainderBytes);
        memcpy(firstRemainderV, remainder.second.contents, remainderBytes);
        memcpy(
            firstAttentionGradient,
            joined.attentionGradient.contents, attentionBytes);
        memcpy(
            firstRemainderGradient,
            remainder.gradient.contents, remainderBytes);
        memcpy(joined.attentionWeight.contents, initialAttention, attentionBytes);
        memcpy(remainder.weight.contents, initialRemainder, remainderBytes);
        memset(joined.attentionFirst.contents, 0, attentionBytes);
        memset(joined.attentionSecond.contents, 0, attentionBytes);
        memset(remainder.first.contents, 0, remainderBytes);
        memset(remainder.second.contents, 0, remainderBytes);
        JoinedReceipt replay = {0};
        if (!RunJoinedStep(
                &attention, &metal, &remainder, &joined, &custody,
                hidden, dHidden,
                0, &replay, &error)) return 5;
        BOOL replayExact =
            !memcmp(joined.attentionWeight.contents, firstAttention, attentionBytes)
            && !memcmp(remainder.weight.contents, firstRemainder, remainderBytes)
            && !memcmp(joined.attentionFirst.contents, firstAttentionM, attentionBytes)
            && !memcmp(joined.attentionSecond.contents, firstAttentionV, attentionBytes)
            && !memcmp(remainder.first.contents, firstRemainderM, remainderBytes)
            && !memcmp(remainder.second.contents, firstRemainderV, remainderBytes)
            && replay.loss == first.loss
            && replay.gradientNorm == first.gradientNorm;
        BOOL finite = RemainderAllFinite(
            joined.attentionGradient.contents, ATTENTION_PARAMETER_ELEMENTS)
            && RemainderAllFinite(
                remainder.gradient.contents, PARAMETER_ELEMENTS)
            && RemainderAllFinite(dHidden, (size_t)SEQUENCE * DIM);
        memcpy(joined.attentionWeight.contents, initialAttention, attentionBytes);
        memcpy(remainder.weight.contents, initialRemainder, remainderBytes);
        memset(joined.attentionFirst.contents, 0, attentionBytes);
        memset(joined.attentionSecond.contents, 0, attentionBytes);
        memset(remainder.first.contents, 0, remainderBytes);
        memset(remainder.second.contents, 0, remainderBytes);
        double joined64Total = 0.0;
        BOOL finite64 = YES;
        for (uint64_t step = 0; step < 64; ++step) {
            JoinedReceipt receipt = {0};
            if (!RunJoinedStep(
                    &attention, &metal, &remainder, &joined, &custody,
                    hidden, dHidden, step, &receipt, &error)) return 6;
            joined64Total += receipt.joined;
            finite64 = finite64 && isfinite(receipt.loss) &&
                isfinite(receipt.gradientNorm) &&
                RemainderAllFinite(
                    joined.attentionWeight.contents,
                    ATTENTION_PARAMETER_ELEMENTS) &&
                RemainderAllFinite(
                    remainder.weight.contents, PARAMETER_ELEMENTS);
        }
        size_t attentionNonzero = 0, remainderNonzero = 0;
        float *attentionGradient = firstAttentionGradient;
        float *remainderGradient = firstRemainderGradient;
        for (size_t i = 0; i < ATTENTION_PARAMETER_ELEMENTS; ++i)
            if (attentionGradient[i] != 0.0f) ++attentionNonzero;
        for (size_t i = 0; i < PARAMETER_ELEMENTS; ++i)
            if (remainderGradient[i] != 0.0f) ++remainderNonzero;
        size_t leafOffsets[9] = {
            ATTENTION_NORM_OFFSET, ATTENTION_Q_OFFSET,
            ATTENTION_K_OFFSET, ATTENTION_V_OFFSET,
            ATTENTION_PARAMETER_ELEMENTS + OUT_OFFSET,
            ATTENTION_PARAMETER_ELEMENTS + NORM_OFFSET,
            ATTENTION_PARAMETER_ELEMENTS + GATE_OFFSET,
            ATTENTION_PARAMETER_ELEMENTS + UP_OFFSET,
            ATTENTION_PARAMETER_ELEMENTS + DOWN_OFFSET,
        };
        size_t leafCounts[9] = {
            DIM, (size_t)DIM * QUERY_DIM,
            (size_t)DIM * KV_DIM, (size_t)DIM * KV_DIM,
            OUT_WEIGHT_ELEMENTS, NORM_ELEMENTS,
            FF_WEIGHT_ELEMENTS, FF_WEIGHT_ELEMENTS,
            DOWN_WEIGHT_ELEMENTS,
        };
        double minimumLeafNonzeroFraction = 1.0;
        for (int leaf = 0; leaf < 9; ++leaf) {
            size_t nonzero = 0;
            for (size_t index = 0; index < leafCounts[leaf]; ++index) {
                size_t joinedIndex = leafOffsets[leaf] + index;
                float value = joinedIndex < ATTENTION_PARAMETER_ELEMENTS
                    ? attentionGradient[joinedIndex]
                    : remainderGradient[
                        joinedIndex - ATTENTION_PARAMETER_ELEMENTS];
                if (value != 0.0f) ++nonzero;
            }
            minimumLeafNonzeroFraction = fmin(
                minimumLeafNonzeroFraction,
                (double)nonzero / leafCounts[leaf]);
        }
        size_t mismatches =
            (replayExact ? 0 : 1) + (finite ? 0 : 1) +
            (finite64 ? 0 : 1) +
            (minimumLeafNonzeroFraction >= 0.95 ? 0 : 1);
        printf(
            "{\"policy\":\"project_theseus_exact_decoder_block_join_v1\","
            "\"shape\":{\"batch\":1,\"sequence\":%d,\"d_model\":%d,"
            "\"ff_dim\":%d,\"query_heads\":%d,\"kv_heads\":%d},"
            "\"parameter_generation\":0,\"parameter_elements\":%zu,"
            "\"parameter_leaf_count\":9,"
            "\"objective_authority_mass\":%.1f,"
            "\"timing\":{\"joined_milliseconds\":%.6f,"
            "\"ane_forward_milliseconds\":%.6f,"
            "\"remainder_milliseconds\":%.6f,"
            "\"ane_backward_milliseconds\":%.6f,"
            "\"attention_gradient_milliseconds\":%.6f,"
            "\"unified_update_milliseconds\":%.6f,"
            "\"mean_joined_64_milliseconds\":%.6f},"
            "\"loss\":%.9g,\"global_gradient_norm\":%.9g,"
            "\"attention_nonzero_gradient_fraction\":%.9g,"
            "\"remainder_nonzero_gradient_fraction\":%.9g,"
            "\"minimum_leaf_nonzero_gradient_fraction\":%.9g,"
            "\"gates\":{\"one_process\":true,"
            "\"compile_once_ane_forward_backward\":true,"
            "\"generation_tagged_iosurface_forward_backward\":true,"
            "\"single_thread_fp32_accelerate_dw\":true,"
            "\"native_metal_remainder\":true,"
            "\"all_nine_parameter_leaves\":true,"
            "\"combined_hidden_gradient\":true,"
            "\"one_objective_mass_normalization\":true,"
            "\"one_global_norm_and_clip\":true,"
            "\"one_fp32_adamw_publication\":true,"
            "\"replay_exact\":%s,\"all_finite\":%s,"
            "\"sixty_four_step_finite\":%s,"
            "\"matched_mlx_wall_control\":false,"
            "\"production_eligible\":false},"
            "\"mismatch_count\":%zu,\"trigger_state\":\"%s\","
            "\"capability_claim\":\"NONE_ENGINEERING_EXACT_BLOCK_JOIN_ONLY\"}\n",
            SEQUENCE, DIM, FF_DIM, QUERY_HEADS, KV_HEADS,
            (size_t)TOTAL_BLOCK_PARAMETERS,
            (double)(SEQUENCE - 3) * DIM,
            first.joined, first.forwardANE, first.remainder,
            first.backwardANE, first.attentionGradient,
            first.unifiedUpdate, joined64Total / 64.0,
            first.loss, first.gradientNorm,
            (double)attentionNonzero / ATTENTION_PARAMETER_ELEMENTS,
            (double)remainderNonzero / PARAMETER_ELEMENTS,
            minimumLeafNonzeroFraction,
            replayExact ? "true" : "false",
            finite ? "true" : "false",
            finite64 ? "true" : "false",
            mismatches, mismatches == 0 ? "GREEN" : "RED");
        return mismatches == 0 ? 0 : 6;
    }
}
