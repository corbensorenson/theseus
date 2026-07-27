/*
 * Exact Project Theseus decoder-block remainder qualification.
 *
 * This native transaction starts at the frozen hidden/attended boundary and
 * closes out_proj + residual, RMSNorm, SwiGLU + down projection, the second
 * residual, masked half-MSE, every remainder parameter/input gradient, one
 * global clip, and one FP32 AdamW publication. Accelerate owns the FP32 GEMMs;
 * Metal owns SwiGLU, objective/reduction, global norm, and the sole update.
 *
 * It is paired with ane_exact_attention_{forward,backward}.m but deliberately
 * makes no full-block or speedup claim until the IOSurface join is measured.
 */

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <math.h>

#define ROWS 128
#define DIM 512
#define FF_DIM 1536
#define THREADS 256
#define OUTPUT_ELEMENTS ((size_t)ROWS * DIM)
#define FF_ELEMENTS ((size_t)ROWS * FF_DIM)
#define OUT_WEIGHT_ELEMENTS ((size_t)DIM * DIM)
#define NORM_ELEMENTS ((size_t)DIM)
#define FF_WEIGHT_ELEMENTS ((size_t)DIM * FF_DIM)
#define DOWN_WEIGHT_ELEMENTS ((size_t)FF_DIM * DIM)
#define OUT_OFFSET 0
#define NORM_OFFSET (OUT_OFFSET + OUT_WEIGHT_ELEMENTS)
#define GATE_OFFSET (NORM_OFFSET + NORM_ELEMENTS)
#define UP_OFFSET (GATE_OFFSET + FF_WEIGHT_ELEMENTS)
#define DOWN_OFFSET (UP_OFFSET + FF_WEIGHT_ELEMENTS)
#define PARAMETER_ELEMENTS (DOWN_OFFSET + DOWN_WEIGHT_ELEMENTS)
#define LOSS_PARTIALS ((OUTPUT_ELEMENTS + THREADS - 1) / THREADS)
#define GRADIENT_PARTIALS ((PARAMETER_ELEMENTS + THREADS - 1) / THREADS)

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> swigluForward;
    id<MTLComputePipelineState> loss;
    id<MTLComputePipelineState> reduceLoss;
    id<MTLComputePipelineState> swigluBackward;
    id<MTLComputePipelineState> squarePartials;
    id<MTLComputePipelineState> reduceNorm;
    id<MTLComputePipelineState> adamw;
} MetalRemainder;

typedef struct {
    id<MTLBuffer> weight;
    id<MTLBuffer> first;
    id<MTLBuffer> second;
    id<MTLBuffer> gradient;
    id<MTLBuffer> gate;
    id<MTLBuffer> up;
    id<MTLBuffer> activated;
    id<MTLBuffer> afterAttention;
    id<MTLBuffer> down;
    id<MTLBuffer> target;
    id<MTLBuffer> mask;
    id<MTLBuffer> output;
    id<MTLBuffer> dOutput;
    id<MTLBuffer> dActivated;
    id<MTLBuffer> dGate;
    id<MTLBuffer> dUp;
    id<MTLBuffer> lossPartials;
    id<MTLBuffer> scalarLoss;
    id<MTLBuffer> gradientPartials;
    id<MTLBuffer> gradientNorm;
} MetalBuffers;

typedef struct {
    double forwardGemmMilliseconds;
    double metalElementwiseMilliseconds;
    double backwardGemmMilliseconds;
    double metalUpdateMilliseconds;
    double joinedMilliseconds;
    float loss;
    float gradientNorm;
} StepReceipt;

typedef struct {
    size_t mismatches;
    float maximum;
} Comparison;

static mach_timebase_info_data_t Timebase;

static double Milliseconds(uint64_t ticks) {
    return (double)ticks * Timebase.numer / Timebase.denom / 1e6;
}

static id<MTLBuffer> Buffer(id<MTLDevice> device, size_t count, BOOL zero) {
    id<MTLBuffer> result =
        [device newBufferWithLength:count * sizeof(float)
                            options:MTLResourceStorageModeShared];
    if (zero && result) memset(result.contents, 0, count * sizeof(float));
    return result;
}

static NSString *MetalSource(void) {
    return [NSString stringWithFormat:
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "constant uint OUTPUT_ELEMENTS=%zu;\n"
         "constant uint FF_ELEMENTS=%zu;\n"
         "constant uint PARAMETER_ELEMENTS=%zu;\n"
         "kernel void swiglu_forward("
         " device const float* gate [[buffer(0)]],"
         " device const float* up [[buffer(1)]],"
         " device float* out [[buffer(2)]],"
         " uint tid [[thread_position_in_grid]]) {"
         " if(tid>=FF_ELEMENTS)return;"
         " float g=gate[tid]; float s=1.0f/(1.0f+exp(-g));"
         " out[tid]=(g*s)*up[tid]; }\n"
         "kernel void block_loss("
         " device const float* after [[buffer(0)]],"
         " device const float* down [[buffer(1)]],"
         " device const float* target [[buffer(2)]],"
         " device const float* mask [[buffer(3)]],"
         " device float* output [[buffer(4)]],"
         " device float* doutput [[buffer(5)]],"
         " device float* partials [[buffer(6)]],"
         " constant float& authority [[buffer(7)]],"
         " uint tid [[thread_position_in_grid]],"
         " uint lane [[thread_index_in_threadgroup]],"
         " uint group [[threadgroup_position_in_grid]]) {"
         " threadgroup float scratch[256]; float square=0.0f;"
         " if(tid<OUTPUT_ELEMENTS){uint row=tid/%d;"
         " float y=after[tid]+down[tid]; float d=y-target[tid];"
         " float active=mask[row]; output[tid]=y;"
         " doutput[tid]=active*d/authority;"
         " square=active*d*d;}"
         " scratch[lane]=square;"
         " threadgroup_barrier(mem_flags::mem_threadgroup);"
         " for(uint s=128;s>0;s>>=1){if(lane<s)scratch[lane]+=scratch[lane+s];"
         " threadgroup_barrier(mem_flags::mem_threadgroup);}"
         " if(lane==0)partials[group]=scratch[0]; }\n"
         "kernel void reduce_loss("
         " device const float* partials [[buffer(0)]],"
         " device float* loss [[buffer(1)]],"
         " constant float& authority [[buffer(2)]],"
         " uint lane [[thread_index_in_threadgroup]]) {"
         " threadgroup float scratch[256]; float sum=0.0f;"
         " for(uint i=lane;i<%zu;i+=256)sum+=partials[i];"
         " scratch[lane]=sum;"
         " threadgroup_barrier(mem_flags::mem_threadgroup);"
         " for(uint s=128;s>0;s>>=1){if(lane<s)scratch[lane]+=scratch[lane+s];"
         " threadgroup_barrier(mem_flags::mem_threadgroup);}"
         " if(lane==0)loss[0]=0.5f*scratch[0]/authority; }\n"
         "kernel void swiglu_backward("
         " device const float* gate [[buffer(0)]],"
         " device const float* up [[buffer(1)]],"
         " device const float* dout [[buffer(2)]],"
         " device float* dgate [[buffer(3)]],"
         " device float* dup [[buffer(4)]],"
         " uint tid [[thread_position_in_grid]]) {"
         " if(tid>=FF_ELEMENTS)return;"
         " float g=gate[tid]; float s=1.0f/(1.0f+exp(-g));"
         " float silu=g*s; float ds=s+g*s*(1.0f-s);"
         " dgate[tid]=dout[tid]*up[tid]*ds;"
         " dup[tid]=dout[tid]*silu; }\n"
         "kernel void square_partials("
         " device const float* gradient [[buffer(0)]],"
         " device float* partials [[buffer(1)]],"
         " uint tid [[thread_position_in_grid]],"
         " uint lane [[thread_index_in_threadgroup]],"
         " uint group [[threadgroup_position_in_grid]]) {"
         " threadgroup float scratch[256];"
         " float v=tid<PARAMETER_ELEMENTS?gradient[tid]:0.0f;"
         " scratch[lane]=v*v;"
         " threadgroup_barrier(mem_flags::mem_threadgroup);"
         " for(uint s=128;s>0;s>>=1){if(lane<s)scratch[lane]+=scratch[lane+s];"
         " threadgroup_barrier(mem_flags::mem_threadgroup);}"
         " if(lane==0)partials[group]=scratch[0]; }\n"
         "kernel void reduce_norm("
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
         "kernel void adamw("
         " device float* weight [[buffer(0)]],"
         " device float* first [[buffer(1)]],"
         " device float* second [[buffer(2)]],"
         " device const float* gradient [[buffer(3)]],"
         " device const float* norm [[buffer(4)]],"
         " constant float& lr [[buffer(5)]],"
         " constant float& b1 [[buffer(6)]],"
         " constant float& b2 [[buffer(7)]],"
         " constant float& eps [[buffer(8)]],"
         " constant float& decay [[buffer(9)]],"
         " constant float& clip [[buffer(10)]],"
         " uint tid [[thread_position_in_grid]]) {"
         " if(tid>=PARAMETER_ELEMENTS)return;"
         " float scale=min(1.0f,clip/max(norm[0],1.0e-12f));"
         " float g=gradient[tid]*scale;"
         " float m=b1*first[tid]+(1.0f-b1)*g;"
         " float v=b2*second[tid]+(1.0f-b2)*g*g;"
         " weight[tid]=weight[tid]*(1.0f-lr*decay)"
         " -lr*m/(sqrt(v)+eps); first[tid]=m; second[tid]=v; }\n",
        OUTPUT_ELEMENTS, FF_ELEMENTS, PARAMETER_ELEMENTS, DIM,
        (size_t)LOSS_PARTIALS, (size_t)GRADIENT_PARTIALS];
}

static id<MTLComputePipelineState> Pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *name,
    NSError **error
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function
        ? [device newComputePipelineStateWithFunction:function error:error]
        : nil;
}

static BOOL BuildMetal(MetalRemainder *metal, NSError **error) {
    metal->device = MTLCreateSystemDefaultDevice();
    metal->queue = [metal->device newCommandQueue];
    id<MTLLibrary> library = [metal->device
        newLibraryWithSource:MetalSource() options:nil error:error];
    if (!library) return NO;
    metal->swigluForward =
        Pipeline(metal->device, library, @"swiglu_forward", error);
    metal->loss = Pipeline(metal->device, library, @"block_loss", error);
    metal->reduceLoss =
        Pipeline(metal->device, library, @"reduce_loss", error);
    metal->swigluBackward =
        Pipeline(metal->device, library, @"swiglu_backward", error);
    metal->squarePartials =
        Pipeline(metal->device, library, @"square_partials", error);
    metal->reduceNorm =
        Pipeline(metal->device, library, @"reduce_norm", error);
    metal->adamw = Pipeline(metal->device, library, @"adamw", error);
    return metal->queue && metal->swigluForward && metal->loss &&
        metal->reduceLoss && metal->swigluBackward &&
        metal->squarePartials && metal->reduceNorm && metal->adamw;
}

static BOOL BuildBuffers(
    MetalRemainder *metal, MetalBuffers *state
) {
    state->weight = Buffer(metal->device, PARAMETER_ELEMENTS, NO);
    state->first = Buffer(metal->device, PARAMETER_ELEMENTS, YES);
    state->second = Buffer(metal->device, PARAMETER_ELEMENTS, YES);
    state->gradient = Buffer(metal->device, PARAMETER_ELEMENTS, YES);
    state->gate = Buffer(metal->device, FF_ELEMENTS, NO);
    state->up = Buffer(metal->device, FF_ELEMENTS, NO);
    state->activated = Buffer(metal->device, FF_ELEMENTS, NO);
    state->afterAttention = Buffer(metal->device, OUTPUT_ELEMENTS, NO);
    state->down = Buffer(metal->device, OUTPUT_ELEMENTS, NO);
    state->target = Buffer(metal->device, OUTPUT_ELEMENTS, NO);
    state->mask = Buffer(metal->device, ROWS, NO);
    state->output = Buffer(metal->device, OUTPUT_ELEMENTS, NO);
    state->dOutput = Buffer(metal->device, OUTPUT_ELEMENTS, NO);
    state->dActivated = Buffer(metal->device, FF_ELEMENTS, NO);
    state->dGate = Buffer(metal->device, FF_ELEMENTS, NO);
    state->dUp = Buffer(metal->device, FF_ELEMENTS, NO);
    state->lossPartials = Buffer(metal->device, LOSS_PARTIALS, YES);
    state->scalarLoss = Buffer(metal->device, 1, YES);
    state->gradientPartials =
        Buffer(metal->device, GRADIENT_PARTIALS, YES);
    state->gradientNorm = Buffer(metal->device, 1, YES);
    return state->weight && state->first && state->second &&
        state->gradient && state->gate && state->up && state->activated &&
        state->afterAttention && state->down && state->target &&
        state->mask && state->output && state->dOutput &&
        state->dActivated && state->dGate && state->dUp &&
        state->lossPartials && state->scalarLoss &&
        state->gradientPartials && state->gradientNorm;
}

static BOOL Complete(id<MTLCommandBuffer> command, NSError **error) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusCompleted) return YES;
    if (error) *error = command.error;
    return NO;
}

static void RMSForward(
    float *output, const float *input, const float *scale
) {
    for (int row = 0; row < ROWS; ++row) {
        double sum = 0.0;
        for (int channel = 0; channel < DIM; ++channel) {
            float value = input[(size_t)row * DIM + channel];
            sum += (double)value * value;
        }
        float inverse = 1.0f / sqrtf((float)(sum / DIM) + 1.0e-5f);
        for (int channel = 0; channel < DIM; ++channel) {
            size_t index = (size_t)row * DIM + channel;
            output[index] = input[index] * inverse * scale[channel];
        }
    }
}

static void RMSBackward(
    float *dInput, float *dScale, const float *input, const float *scale,
    const float *dOutput
) {
    memset(dScale, 0, DIM * sizeof(float));
    for (int row = 0; row < ROWS; ++row) {
        double square = 0.0;
        double dot = 0.0;
        for (int channel = 0; channel < DIM; ++channel) {
            size_t index = (size_t)row * DIM + channel;
            square += (double)input[index] * input[index];
            dot += (double)dOutput[index] * scale[channel] * input[index];
        }
        float inverse = 1.0f / sqrtf((float)(square / DIM) + 1.0e-5f);
        float correction =
            inverse * inverse * inverse * (float)(dot / DIM);
        for (int channel = 0; channel < DIM; ++channel) {
            size_t index = (size_t)row * DIM + channel;
            dScale[channel] += dOutput[index] * input[index] * inverse;
            dInput[index] = inverse * dOutput[index] * scale[channel]
                - input[index] * correction;
        }
    }
}

static BOOL EncodeElementwiseForward(
    MetalRemainder *metal, MetalBuffers *state, float authority,
    NSError **error
) {
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> swiglu = [command computeCommandEncoder];
    [swiglu setComputePipelineState:metal->swigluForward];
    [swiglu setBuffer:state->gate offset:0 atIndex:0];
    [swiglu setBuffer:state->up offset:0 atIndex:1];
    [swiglu setBuffer:state->activated offset:0 atIndex:2];
    [swiglu dispatchThreads:MTLSizeMake(FF_ELEMENTS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [swiglu endEncoding];
    return Complete(command, error);
}

static BOOL EncodeLossAndBackward(
    MetalRemainder *metal, MetalBuffers *state, float authority,
    NSError **error
) {
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> loss = [command computeCommandEncoder];
    [loss setComputePipelineState:metal->loss];
    [loss setBuffer:state->afterAttention offset:0 atIndex:0];
    [loss setBuffer:state->down offset:0 atIndex:1];
    [loss setBuffer:state->target offset:0 atIndex:2];
    [loss setBuffer:state->mask offset:0 atIndex:3];
    [loss setBuffer:state->output offset:0 atIndex:4];
    [loss setBuffer:state->dOutput offset:0 atIndex:5];
    [loss setBuffer:state->lossPartials offset:0 atIndex:6];
    [loss setBytes:&authority length:sizeof(float) atIndex:7];
    [loss dispatchThreads:MTLSizeMake(LOSS_PARTIALS * THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [loss endEncoding];
    id<MTLComputeCommandEncoder> reduce = [command computeCommandEncoder];
    [reduce setComputePipelineState:metal->reduceLoss];
    [reduce setBuffer:state->lossPartials offset:0 atIndex:0];
    [reduce setBuffer:state->scalarLoss offset:0 atIndex:1];
    [reduce setBytes:&authority length:sizeof(float) atIndex:2];
    [reduce dispatchThreads:MTLSizeMake(THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [reduce endEncoding];
    return Complete(command, error);
}

static BOOL EncodeSwiGLUBackward(
    MetalRemainder *metal, MetalBuffers *state, NSError **error
) {
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:metal->swigluBackward];
    [encoder setBuffer:state->gate offset:0 atIndex:0];
    [encoder setBuffer:state->up offset:0 atIndex:1];
    [encoder setBuffer:state->dActivated offset:0 atIndex:2];
    [encoder setBuffer:state->dGate offset:0 atIndex:3];
    [encoder setBuffer:state->dUp offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(FF_ELEMENTS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [encoder endEncoding];
    return Complete(command, error);
}

static BOOL EncodeUpdate(
    MetalRemainder *metal, MetalBuffers *state, NSError **error
) {
    id<MTLCommandBuffer> command = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> square = [command computeCommandEncoder];
    [square setComputePipelineState:metal->squarePartials];
    [square setBuffer:state->gradient offset:0 atIndex:0];
    [square setBuffer:state->gradientPartials offset:0 atIndex:1];
    [square dispatchThreads:MTLSizeMake(GRADIENT_PARTIALS * THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [square endEncoding];
    id<MTLComputeCommandEncoder> reduce = [command computeCommandEncoder];
    [reduce setComputePipelineState:metal->reduceNorm];
    [reduce setBuffer:state->gradientPartials offset:0 atIndex:0];
    [reduce setBuffer:state->gradientNorm offset:0 atIndex:1];
    [reduce dispatchThreads:MTLSizeMake(THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [reduce endEncoding];
    float lr = 3.0e-5f, b1 = 0.9f, b2 = 0.999f;
    float epsilon = 1.0e-8f, decay = 0.01f, clip = 1.0f;
    id<MTLComputeCommandEncoder> update = [command computeCommandEncoder];
    [update setComputePipelineState:metal->adamw];
    [update setBuffer:state->weight offset:0 atIndex:0];
    [update setBuffer:state->first offset:0 atIndex:1];
    [update setBuffer:state->second offset:0 atIndex:2];
    [update setBuffer:state->gradient offset:0 atIndex:3];
    [update setBuffer:state->gradientNorm offset:0 atIndex:4];
    [update setBytes:&lr length:sizeof(float) atIndex:5];
    [update setBytes:&b1 length:sizeof(float) atIndex:6];
    [update setBytes:&b2 length:sizeof(float) atIndex:7];
    [update setBytes:&epsilon length:sizeof(float) atIndex:8];
    [update setBytes:&decay length:sizeof(float) atIndex:9];
    [update setBytes:&clip length:sizeof(float) atIndex:10];
    [update dispatchThreads:MTLSizeMake(PARAMETER_ELEMENTS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [update endEncoding];
    return Complete(command, error);
}

static BOOL AllFinite(const float *values, size_t count) {
    for (size_t index = 0; index < count; ++index)
        if (!isfinite(values[index])) return NO;
    return YES;
}

static Comparison Compare(
    const float *actual, const float *expected, size_t count, float tolerance
) {
    Comparison result = {0};
    for (size_t index = 0; index < count; ++index) {
        float delta = fabsf(actual[index] - expected[index]);
        result.maximum = fmaxf(result.maximum, delta);
        if (delta > tolerance) ++result.mismatches;
    }
    return result;
}

static void BuildReferenceSwiGLU(
    float *activated, const float *gate, const float *up
) {
    for (size_t index = 0; index < FF_ELEMENTS; ++index) {
        float sigmoid = 1.0f / (1.0f + expf(-gate[index]));
        activated[index] = gate[index] * sigmoid * up[index];
    }
}

static void BuildReferenceSwiGLUBackward(
    float *dGate, float *dUp, const float *gate, const float *up,
    const float *dActivated
) {
    for (size_t index = 0; index < FF_ELEMENTS; ++index) {
        float sigmoid = 1.0f / (1.0f + expf(-gate[index]));
        float silu = gate[index] * sigmoid;
        float derivative =
            sigmoid + gate[index] * sigmoid * (1.0f - sigmoid);
        dGate[index] = dActivated[index] * up[index] * derivative;
        dUp[index] = dActivated[index] * silu;
    }
}

static BOOL RunStep(
    MetalRemainder *metal, MetalBuffers *state,
    const float *hidden, const float *attended,
    float *normalized, float *dNormalized, float *dAfter,
    float *dAttended, float *dHiddenDirect,
    StepReceipt *receipt, NSError **error
) {
    uint64_t joinedStarted = mach_absolute_time();
    float *weight = state->weight.contents;
    float *gradient = state->gradient.contents;
    float *outWeight = weight + OUT_OFFSET;
    float *normScale = weight + NORM_OFFSET;
    float *gateWeight = weight + GATE_OFFSET;
    float *upWeight = weight + UP_OFFSET;
    float *downWeight = weight + DOWN_OFFSET;
    float *after = state->afterAttention.contents;
    uint64_t started = mach_absolute_time();
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ROWS, DIM, DIM, 1.0f, attended, DIM,
        outWeight, DIM, 0.0f, after, DIM);
    for (size_t index = 0; index < OUTPUT_ELEMENTS; ++index)
        after[index] += hidden[index];
    RMSForward(normalized, after, normScale);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ROWS, FF_DIM, DIM, 1.0f, normalized, DIM,
        gateWeight, FF_DIM, 0.0f, state->gate.contents, FF_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ROWS, FF_DIM, DIM, 1.0f, normalized, DIM,
        upWeight, FF_DIM, 0.0f, state->up.contents, FF_DIM);
    receipt->forwardGemmMilliseconds =
        Milliseconds(mach_absolute_time() - started);
    started = mach_absolute_time();
    if (!EncodeElementwiseForward(metal, state, 0.0f, error)) return NO;
    receipt->metalElementwiseMilliseconds =
        Milliseconds(mach_absolute_time() - started);
    started = mach_absolute_time();
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ROWS, DIM, FF_DIM, 1.0f, state->activated.contents, FF_DIM,
        downWeight, DIM, 0.0f, state->down.contents, DIM);
    receipt->forwardGemmMilliseconds +=
        Milliseconds(mach_absolute_time() - started);
    float authority = (ROWS - 3) * DIM;
    started = mach_absolute_time();
    if (!EncodeLossAndBackward(metal, state, authority, error)) return NO;
    receipt->metalElementwiseMilliseconds +=
        Milliseconds(mach_absolute_time() - started);
    float *dOutput = state->dOutput.contents;
    started = mach_absolute_time();
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        FF_DIM, DIM, ROWS, 1.0f, state->activated.contents, FF_DIM,
        dOutput, DIM, 0.0f, gradient + DOWN_OFFSET, DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ROWS, FF_DIM, DIM, 1.0f, dOutput, DIM,
        downWeight, DIM, 0.0f, state->dActivated.contents, FF_DIM);
    receipt->backwardGemmMilliseconds =
        Milliseconds(mach_absolute_time() - started);
    started = mach_absolute_time();
    if (!EncodeSwiGLUBackward(metal, state, error)) return NO;
    receipt->metalElementwiseMilliseconds +=
        Milliseconds(mach_absolute_time() - started);
    started = mach_absolute_time();
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        DIM, FF_DIM, ROWS, 1.0f, normalized, DIM,
        state->dGate.contents, FF_DIM, 0.0f,
        gradient + GATE_OFFSET, FF_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        DIM, FF_DIM, ROWS, 1.0f, normalized, DIM,
        state->dUp.contents, FF_DIM, 0.0f,
        gradient + UP_OFFSET, FF_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ROWS, DIM, FF_DIM, 1.0f, state->dGate.contents, FF_DIM,
        gateWeight, FF_DIM, 0.0f, dNormalized, DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ROWS, DIM, FF_DIM, 1.0f, state->dUp.contents, FF_DIM,
        upWeight, FF_DIM, 1.0f, dNormalized, DIM);
    RMSBackward(
        dAfter, gradient + NORM_OFFSET, after, normScale, dNormalized);
    for (size_t index = 0; index < OUTPUT_ELEMENTS; ++index)
        dAfter[index] += dOutput[index];
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        DIM, DIM, ROWS, 1.0f, attended, DIM,
        dAfter, DIM, 0.0f, gradient + OUT_OFFSET, DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        ROWS, DIM, DIM, 1.0f, dAfter, DIM,
        outWeight, DIM, 0.0f, dAttended, DIM);
    memcpy(dHiddenDirect, dAfter, OUTPUT_ELEMENTS * sizeof(float));
    receipt->backwardGemmMilliseconds +=
        Milliseconds(mach_absolute_time() - started);
    started = mach_absolute_time();
    if (!EncodeUpdate(metal, state, error)) return NO;
    receipt->metalUpdateMilliseconds =
        Milliseconds(mach_absolute_time() - started);
    receipt->loss = ((float *)state->scalarLoss.contents)[0];
    receipt->gradientNorm = ((float *)state->gradientNorm.contents)[0];
    receipt->joinedMilliseconds =
        Milliseconds(mach_absolute_time() - joinedStarted);
    return YES;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&Timebase);
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
        NSError *error = nil;
        MetalRemainder metal = {0};
        MetalBuffers state = {0};
        if (!BuildMetal(&metal, &error) || !BuildBuffers(&metal, &state)) {
            fprintf(stderr, "metal_initialization_failed:%s\n",
                    error.description.UTF8String);
            return 2;
        }
        float *hidden = malloc(OUTPUT_ELEMENTS * sizeof(float));
        float *attended = malloc(OUTPUT_ELEMENTS * sizeof(float));
        float *normalized = malloc(OUTPUT_ELEMENTS * sizeof(float));
        float *dNormalized = malloc(OUTPUT_ELEMENTS * sizeof(float));
        float *dAfter = malloc(OUTPUT_ELEMENTS * sizeof(float));
        float *dAttended = malloc(OUTPUT_ELEMENTS * sizeof(float));
        float *dHiddenDirect = malloc(OUTPUT_ELEMENTS * sizeof(float));
        float *weight = state.weight.contents;
        float *target = state.target.contents;
        float *mask = state.mask.contents;
        for (size_t i = 0; i < OUTPUT_ELEMENTS; ++i) {
            hidden[i] = sinf((float)(i % 1009) * 0.011f) * 0.125f;
            attended[i] = cosf((float)(i % 977) * 0.013f) * 0.0625f;
            target[i] = sinf((float)(i % 953) * 0.007f) * 0.125f;
        }
        for (int row = 0; row < ROWS; ++row)
            mask[row] = row < ROWS - 3 ? 1.0f : 0.0f;
        for (size_t i = 0; i < PARAMETER_ELEMENTS; ++i)
            weight[i] = sinf((float)(i % 887) * 0.005f) * 0.02f;
        for (int i = 0; i < DIM; ++i)
            weight[NORM_OFFSET + i] = 1.0f + sinf(i * 0.019f) * 0.01f;
        size_t stateBytes = PARAMETER_ELEMENTS * sizeof(float);
        float *initialWeight = malloc(stateBytes);
        float *initialFirst = calloc(PARAMETER_ELEMENTS, sizeof(float));
        float *initialSecond = calloc(PARAMETER_ELEMENTS, sizeof(float));
        float *firstWeight = malloc(stateBytes);
        float *firstFirst = malloc(stateBytes);
        float *firstSecond = malloc(stateBytes);
        memcpy(initialWeight, weight, stateBytes);
        StepReceipt first = {0};
        if (!RunStep(
                &metal, &state, hidden, attended, normalized, dNormalized,
                dAfter, dAttended, dHiddenDirect, &first, &error)) {
            fprintf(stderr, "first_step_failed:%s\n", error.description.UTF8String);
            return 3;
        }
        memcpy(firstWeight, state.weight.contents, stateBytes);
        memcpy(firstFirst, state.first.contents, stateBytes);
        memcpy(firstSecond, state.second.contents, stateBytes);
        float *referenceActivated = malloc(FF_ELEMENTS * sizeof(float));
        float *referenceDGate = malloc(FF_ELEMENTS * sizeof(float));
        float *referenceDUp = malloc(FF_ELEMENTS * sizeof(float));
        BuildReferenceSwiGLU(
            referenceActivated, state.gate.contents, state.up.contents);
        BuildReferenceSwiGLUBackward(
            referenceDGate, referenceDUp, state.gate.contents,
            state.up.contents, state.dActivated.contents);
        Comparison activatedComparison = Compare(
            state.activated.contents, referenceActivated, FF_ELEMENTS, 1.0e-5f);
        Comparison dGateComparison = Compare(
            state.dGate.contents, referenceDGate, FF_ELEMENTS, 1.0e-5f);
        Comparison dUpComparison = Compare(
            state.dUp.contents, referenceDUp, FF_ELEMENTS, 1.0e-5f);
        memcpy(state.weight.contents, initialWeight, stateBytes);
        memcpy(state.first.contents, initialFirst, stateBytes);
        memcpy(state.second.contents, initialSecond, stateBytes);
        StepReceipt replay = {0};
        if (!RunStep(
                &metal, &state, hidden, attended, normalized, dNormalized,
                dAfter, dAttended, dHiddenDirect, &replay, &error)) return 4;
        BOOL replayExact =
            !memcmp(state.weight.contents, firstWeight, stateBytes) &&
            !memcmp(state.first.contents, firstFirst, stateBytes) &&
            !memcmp(state.second.contents, firstSecond, stateBytes) &&
            replay.loss == first.loss &&
            replay.gradientNorm == first.gradientNorm;
        memcpy(state.weight.contents, initialWeight, stateBytes);
        memset(state.first.contents, 0, stateBytes);
        memset(state.second.contents, 0, stateBytes);
        double joinedTotal = 0.0;
        BOOL finite64 = YES;
        for (int step = 0; step < 64; ++step) {
            StepReceipt receipt = {0};
            if (!RunStep(
                    &metal, &state, hidden, attended, normalized, dNormalized,
                    dAfter, dAttended, dHiddenDirect, &receipt, &error)) return 5;
            joinedTotal += receipt.joinedMilliseconds;
            finite64 = finite64 && isfinite(receipt.loss) &&
                isfinite(receipt.gradientNorm) &&
                AllFinite(state.weight.contents, PARAMETER_ELEMENTS) &&
                AllFinite(state.first.contents, PARAMETER_ELEMENTS) &&
                AllFinite(state.second.contents, PARAMETER_ELEMENTS);
        }
        BOOL gradientsFinite =
            AllFinite(state.gradient.contents, PARAMETER_ELEMENTS) &&
            AllFinite(dAttended, OUTPUT_ELEMENTS) &&
            AllFinite(dHiddenDirect, OUTPUT_ELEMENTS);
        size_t nonzeroGradients = 0;
        float *gradient = state.gradient.contents;
        for (size_t i = 0; i < PARAMETER_ELEMENTS; ++i)
            if (gradient[i] != 0.0f) ++nonzeroGradients;
        size_t mismatches =
            activatedComparison.mismatches + dGateComparison.mismatches +
            dUpComparison.mismatches + (replayExact ? 0 : 1) +
            (finite64 ? 0 : 1) + (gradientsFinite ? 0 : 1);
        printf(
            "{\"policy\":\"project_theseus_exact_decoder_block_remainder_v1\","
            "\"shape\":{\"rows\":%d,\"d_model\":%d,\"ff_dim\":%d},"
            "\"parameter_generation\":0,\"parameter_elements\":%zu,"
            "\"objective_authority_mass\":%.1f,"
            "\"timing\":{\"first_joined_milliseconds\":%.6f,"
            "\"forward_gemm_milliseconds\":%.6f,"
            "\"metal_elementwise_milliseconds\":%.6f,"
            "\"backward_gemm_milliseconds\":%.6f,"
            "\"metal_update_milliseconds\":%.6f,"
            "\"mean_joined_64_milliseconds\":%.6f},"
            "\"comparisons\":{\"swiglu_activation\":{\"tolerance\":1e-5,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"swiglu_gate_gradient\":{\"tolerance\":1e-5,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"swiglu_up_gradient\":{\"tolerance\":1e-5,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu}},"
            "\"loss\":%.9g,\"gradient_norm\":%.9g,"
            "\"nonzero_gradient_fraction\":%.9g,"
            "\"gates\":{\"out_projection_and_unscaled_residual\":true,"
            "\"second_rmsnorm_forward_backward\":true,"
            "\"swiglu_forward_backward\":true,\"down_projection\":true,"
            "\"masked_scalar_loss\":true,\"all_five_parameter_leaves\":true,"
            "\"attended_and_direct_hidden_gradients\":true,"
            "\"one_global_clip\":true,\"one_fp32_adamw_update\":true,"
            "\"replay_exact\":%s,\"sixty_four_step_finite\":%s,"
            "\"complete_attention_join\":false,\"production_eligible\":false},"
            "\"mismatch_count\":%zu,\"trigger_state\":\"%s\","
            "\"capability_claim\":\"NONE_ENGINEERING_BLOCK_REMAINDER_ONLY\"}\n",
            ROWS, DIM, FF_DIM, (size_t)PARAMETER_ELEMENTS,
            (double)(ROWS - 3) * DIM,
            first.joinedMilliseconds, first.forwardGemmMilliseconds,
            first.metalElementwiseMilliseconds,
            first.backwardGemmMilliseconds, first.metalUpdateMilliseconds,
            joinedTotal / 64.0,
            activatedComparison.maximum, activatedComparison.mismatches,
            dGateComparison.maximum, dGateComparison.mismatches,
            dUpComparison.maximum, dUpComparison.mismatches,
            first.loss, first.gradientNorm,
            (double)nonzeroGradients / PARAMETER_ELEMENTS,
            replayExact ? "true" : "false",
            finite64 ? "true" : "false",
            mismatches, mismatches == 0 ? "GREEN" : "RED");
        free(hidden); free(attended); free(normalized); free(dNormalized);
        free(dAfter); free(dAttended); free(dHiddenDirect);
        free(initialWeight); free(initialFirst); free(initialSecond);
        free(firstWeight); free(firstFirst); free(firstSecond);
        free(referenceActivated); free(referenceDGate); free(referenceDUp);
        return mismatches == 0 ? 0 : 6;
    }
}
