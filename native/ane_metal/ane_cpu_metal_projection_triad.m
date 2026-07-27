/*
 * Native zero-copy ANE + Accelerate + Metal q_proj training transaction.
 *
 * ANE: forward and input gradient over generation-tagged IOSurfaces.
 * Accelerate: single-thread FP32 dW directly from shared X/dY buffers.
 * Metal: FP32->FP16 surface packing, station loss/dY, reductions, clipping,
 *        and the sole FP32 AdamW update.
 *
 * The private ANE runtime and dynamic-matmul MIL technique are independently
 * reimplemented from maderix/ANE (MIT); see THIRD_PARTY_NOTICES.md.
 */

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <dlfcn.h>
#include <mach/mach_time.h>
#include <math.h>
#include <unistd.h>

#include "ane_metal_surface_contract.h"

#define ROWS 2048
#define DIM 512
#define ACTIVATION_ELEMENTS ((size_t)ROWS * DIM)
#define WEIGHT_ELEMENTS ((size_t)DIM * DIM)
#define THREADS 256
#define LOSS_PARTIALS ((ACTIVATION_ELEMENTS + THREADS - 1) / THREADS)
#define GRADIENT_PARTIALS ((WEIGHT_ELEMENTS + THREADS - 1) / THREADS)

static Class DescriptorClass;
static Class InMemoryModelClass;
static Class RequestClass;
static Class IOSurfaceObjectClass;
static mach_timebase_info_data_t Timebase;

typedef struct {
    id model;
    NSString *temporaryDirectory;
} ANEModel;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> packForward;
    id<MTLComputePipelineState> lossAndDY;
    id<MTLComputePipelineState> reduceLoss;
    id<MTLComputePipelineState> packDX;
    id<MTLComputePipelineState> squarePartials;
    id<MTLComputePipelineState> reducePartials;
    id<MTLComputePipelineState> adamw;
} MetalStation;

typedef struct {
    id<MTLBuffer> x;
    id<MTLBuffer> target;
    id<MTLBuffer> weight;
    id<MTLBuffer> first;
    id<MTLBuffer> second;
    id<MTLBuffer> dy;
    id<MTLBuffer> dw;
    id<MTLBuffer> lossPartials;
    id<MTLBuffer> loss;
    id<MTLBuffer> gradientPartials;
    id<MTLBuffer> gradientNorm;
    IOSurfaceRef forwardInput;
    IOSurfaceRef forwardOutput;
    IOSurfaceRef dxInput;
    IOSurfaceRef dxOutput;
    id<MTLTexture> forwardInputTexture;
    id<MTLTexture> forwardOutputTexture;
    id<MTLTexture> dxInputTexture;
    id<MTLTexture> dxOutputTexture;
    theseus_ane_metal_surface forwardCustody;
    theseus_ane_metal_surface outputCustody;
    theseus_ane_metal_surface dxInputCustody;
    theseus_ane_metal_surface dxOutputCustody;
} TransactionState;

typedef struct {
    double packForwardMilliseconds;
    double aneForwardMilliseconds;
    double lossAndPackDXMilliseconds;
    double concurrentDXDWMilliseconds;
    double aneDXMilliseconds;
    double cpuDWMilliseconds;
    double metalUpdateMilliseconds;
    double joinedMilliseconds;
    float loss;
    float gradientNorm;
} StepReceipt;

static double Milliseconds(uint64_t ticks) {
    return (double)ticks * Timebase.numer / Timebase.denom / 1e6;
}

static BOOL LoadPrivateANE(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    DescriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    InMemoryModelClass = NSClassFromString(@"_ANEInMemoryModel");
    RequestClass = NSClassFromString(@"_ANERequest");
    IOSurfaceObjectClass = NSClassFromString(@"_ANEIOSurfaceObject");
    return DescriptorClass && InMemoryModelClass && RequestClass &&
           IOSurfaceObjectClass;
}

static IOSurfaceRef CreateHalfSurface(size_t width, size_t height) {
    size_t bytesPerRow = width * sizeof(_Float16);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (__bridge NSString *)kIOSurfaceWidth : @(width),
        (__bridge NSString *)kIOSurfaceHeight : @(height),
        (__bridge NSString *)kIOSurfaceBytesPerElement : @2,
        (__bridge NSString *)kIOSurfaceBytesPerRow : @(bytesPerRow),
        (__bridge NSString *)kIOSurfaceAllocSize : @(bytesPerRow * height),
        (__bridge NSString *)kIOSurfacePixelFormat :
            @(kCVPixelFormatType_OneComponent16Half),
    });
}

static theseus_ane_metal_surface CustodyForSurface(
    IOSurfaceRef surface, uint64_t spatial, uint64_t channels
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
    custody.active_writer = THESEUS_WRITER_NONE;
    return custody;
}

static NSString *DynamicProjectionMIL(void) {
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, "
         "{\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [mil appendFormat:
        @"    func main<ios18>(tensor<fp16, [1,%d,1,%d]> packed) {\n",
        DIM, ROWS + DIM];
    [mil appendString:
        @"        tensor<int32,[4]> ba = const()[name=string(\"ba\"),"
         "val=tensor<int32,[4]>([0,0,0,0])];\n"];
    [mil appendFormat:
        @"        tensor<int32,[4]> sa = const()[name=string(\"sa\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n", DIM, ROWS];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> act = "
         "slice_by_size(x=packed,begin=ba,size=sa)[name=string(\"act\")];\n",
        DIM, ROWS];
    [mil appendFormat:
        @"        tensor<int32,[4]> bw = const()[name=string(\"bw\"),"
         "val=tensor<int32,[4]>([0,0,0,%d])];\n", ROWS];
    [mil appendFormat:
        @"        tensor<int32,[4]> sw = const()[name=string(\"sw\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n", DIM, DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> wt = "
         "slice_by_size(x=packed,begin=bw,size=sw)[name=string(\"wt\")];\n",
        DIM, DIM];
    [mil appendFormat:
        @"        tensor<int32,[4]> ra = const()[name=string(\"ra\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n", DIM, ROWS];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> a2 = "
         "reshape(shape=ra,x=act)[name=string(\"a2\")];\n", DIM, ROWS];
    [mil appendString:
        @"        tensor<int32,[4]> pm = const()[name=string(\"pm\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> a3 = "
         "transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", ROWS, DIM];
    [mil appendFormat:
        @"        tensor<int32,[4]> rw = const()[name=string(\"rw\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n", DIM, DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> w = "
         "reshape(shape=rw,x=wt)[name=string(\"w\")];\n", DIM, DIM];
    [mil appendString:
        @"        bool bf = const()[name=string(\"bf\"),val=bool(false)];\n"];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> yh = "
         "matmul(transpose_x=bf,transpose_y=bf,x=a3,y=w)"
         "[name=string(\"projection\")];\n", ROWS, DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> yt = "
         "transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", DIM, ROWS];
    [mil appendFormat:
        @"        tensor<int32,[4]> ro = const()[name=string(\"ro\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n", DIM, ROWS];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> out = "
         "reshape(shape=ro,x=yt)[name=string(\"out\")];\n", DIM, ROWS];
    [mil appendString:@"    } -> (out);\n}\n"];
    return mil;
}

static ANEModel *CompileANEModel(NSError **error) {
    NSData *milData =
        [DynamicProjectionMIL() dataUsingEncoding:NSUTF8StringEncoding];
    id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        DescriptorClass,
        @selector(modelWithMILText:weights:optionsPlist:),
        milData, @{}, nil);
    if (!descriptor) {
        return NULL;
    }
    id model = ((id(*)(Class, SEL, id))objc_msgSend)(
        InMemoryModelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
    NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
        model, @selector(hexStringIdentifier));
    NSString *directory =
        [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
    [[NSFileManager defaultManager]
        createDirectoryAtPath:[directory stringByAppendingPathComponent:@"weights"]
  withIntermediateDirectories:YES
                   attributes:nil
                        error:nil];
    [milData writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
              atomically:YES];
    BOOL compiled = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, error);
    if (!compiled) {
        [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
        return NULL;
    }
    BOOL loaded = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, error);
    if (!loaded) {
        [[NSFileManager defaultManager] removeItemAtPath:directory error:nil];
        return NULL;
    }
    ANEModel *result = calloc(1, sizeof(ANEModel));
    result->model = model;
    result->temporaryDirectory = directory;
    return result;
}

static BOOL EvaluateANE(
    ANEModel *model, IOSurfaceRef input, IOSurfaceRef output, NSError **error
) {
    id wrappedInput = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        IOSurfaceObjectClass, @selector(objectWithIOSurface:), input);
    id wrappedOutput = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        IOSurfaceObjectClass, @selector(objectWithIOSurface:), output);
    id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
        RequestClass,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                                 weightsBuffer:perfStats:procedureIndex:),
        @[ wrappedInput ], @[ @0 ], @[ wrappedOutput ], @[ @0 ],
        nil, nil, @0);
    return ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        model->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, request, error);
}

static void FreeANEModel(ANEModel *model) {
    if (!model) {
        return;
    }
    NSError *error = nil;
    ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        model->model, @selector(unloadWithQoS:error:), 21, &error);
    [[NSFileManager defaultManager]
        removeItemAtPath:model->temporaryDirectory error:nil];
    free(model);
}

static NSString *MetalSource(void) {
    return [NSString stringWithFormat:
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "constant uint ROWS = %d;\n"
         "constant uint DIM = %d;\n"
         "constant uint ACTIVATION_ELEMENTS = %zu;\n"
         "constant uint WEIGHT_ELEMENTS = %zu;\n"
         "kernel void pack_forward("
         " device const float *x [[buffer(0)]],"
         " device const float *weight [[buffer(1)]],"
         " texture2d<half,access::write> packed [[texture(0)]],"
         " uint2 gid [[thread_position_in_grid]]) {"
         "  if (gid.y >= DIM || gid.x >= ROWS + DIM) return;"
         "  float v = gid.x < ROWS"
         "    ? x[gid.x * DIM + gid.y]"
         "    : weight[gid.y * DIM + (gid.x - ROWS)];"
         "  packed.write(half4(half(v)), gid);"
         "}\n"
         "kernel void loss_and_dy("
         " texture2d<half,access::read> y [[texture(0)]],"
         " texture2d<half,access::write> dy_half [[texture(1)]],"
         " device const float *target [[buffer(0)]],"
         " device float *dy [[buffer(1)]],"
         " device float *partials [[buffer(2)]],"
         " uint tid [[thread_position_in_grid]],"
         " uint lane [[thread_index_in_threadgroup]],"
         " uint group [[threadgroup_position_in_grid]]) {"
         "  threadgroup float scratch[256];"
         "  float square = 0.0f;"
         "  if (tid < ACTIVATION_ELEMENTS) {"
         "    uint row = tid / DIM;"
         "    uint channel = tid - row * DIM;"
         "    float diff = float(y.read(uint2(row, channel)).x) - target[tid];"
         "    float grad = diff / float(ACTIVATION_ELEMENTS);"
         "    dy[tid] = grad;"
         "    dy_half.write(half4(half(grad)), uint2(row, channel));"
         "    square = diff * diff;"
         "  }"
         "  scratch[lane] = square;"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  for (uint stride = 128; stride > 0; stride >>= 1) {"
         "    if (lane < stride) scratch[lane] += scratch[lane + stride];"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  }"
         "  if (lane == 0) partials[group] = scratch[0];"
         "}\n"
         "kernel void reduce_loss("
         " device const float *partials [[buffer(0)]],"
         " device float *loss [[buffer(1)]],"
         " uint lane [[thread_index_in_threadgroup]]) {"
         "  threadgroup float scratch[256];"
         "  float sum = 0.0f;"
         "  for (uint index = lane; index < %zu; index += 256)"
         "    sum += partials[index];"
         "  scratch[lane] = sum;"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  for (uint stride = 128; stride > 0; stride >>= 1) {"
         "    if (lane < stride) scratch[lane] += scratch[lane + stride];"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  }"
         "  if (lane == 0) loss[0] = 0.5f * scratch[0] / "
         "float(ACTIVATION_ELEMENTS);"
         "}\n"
         "kernel void pack_dx("
         " texture2d<half,access::read> dy_half [[texture(0)]],"
         " texture2d<half,access::write> packed [[texture(1)]],"
         " device const float *weight [[buffer(0)]],"
         " uint2 gid [[thread_position_in_grid]]) {"
         "  if (gid.y >= DIM || gid.x >= ROWS + DIM) return;"
         "  half v = gid.x < ROWS"
         "    ? dy_half.read(gid).x"
         "    : half(weight[(gid.x - ROWS) * DIM + gid.y]);"
         "  packed.write(half4(v), gid);"
         "}\n"
         "kernel void square_partials("
         " device const float *gradient [[buffer(0)]],"
         " device float *partials [[buffer(1)]],"
         " uint tid [[thread_position_in_grid]],"
         " uint lane [[thread_index_in_threadgroup]],"
         " uint group [[threadgroup_position_in_grid]]) {"
         "  threadgroup float scratch[256];"
         "  float value = tid < WEIGHT_ELEMENTS ? gradient[tid] : 0.0f;"
         "  scratch[lane] = value * value;"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  for (uint stride = 128; stride > 0; stride >>= 1) {"
         "    if (lane < stride) scratch[lane] += scratch[lane + stride];"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  }"
         "  if (lane == 0) partials[group] = scratch[0];"
         "}\n"
         "kernel void reduce_norm("
         " device const float *partials [[buffer(0)]],"
         " device float *norm [[buffer(1)]],"
         " uint lane [[thread_index_in_threadgroup]]) {"
         "  threadgroup float scratch[256];"
         "  float sum = 0.0f;"
         "  for (uint index = lane; index < %zu; index += 256)"
         "    sum += partials[index];"
         "  scratch[lane] = sum;"
         "  threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  for (uint stride = 128; stride > 0; stride >>= 1) {"
         "    if (lane < stride) scratch[lane] += scratch[lane + stride];"
         "    threadgroup_barrier(mem_flags::mem_threadgroup);"
         "  }"
         "  if (lane == 0) norm[0] = sqrt(scratch[0]);"
         "}\n"
         "kernel void adamw("
         " device float *weight [[buffer(0)]],"
         " device float *first [[buffer(1)]],"
         " device float *second [[buffer(2)]],"
         " device const float *gradient [[buffer(3)]],"
         " device const float *norm [[buffer(4)]],"
         " constant float &lr [[buffer(5)]],"
         " constant float &beta1 [[buffer(6)]],"
         " constant float &beta2 [[buffer(7)]],"
         " constant float &epsilon [[buffer(8)]],"
         " constant float &weight_decay [[buffer(9)]],"
         " constant float &clip_norm [[buffer(10)]],"
         " uint tid [[thread_position_in_grid]]) {"
         "  if (tid >= WEIGHT_ELEMENTS) return;"
         "  float scale = min(1.0f, clip_norm / max(norm[0], 1.0e-6f));"
         "  float g = gradient[tid] * scale;"
         "  float m = beta1 * first[tid] + (1.0f - beta1) * g;"
         "  float v = beta2 * second[tid] + (1.0f - beta2) * g * g;"
         "  float w = weight[tid] * (1.0f - lr * weight_decay);"
         "  weight[tid] = w - lr * m / (sqrt(v) + epsilon);"
         "  first[tid] = m;"
         "  second[tid] = v;"
         "}\n",
        ROWS, DIM, ACTIVATION_ELEMENTS, WEIGHT_ELEMENTS,
        (size_t)LOSS_PARTIALS, (size_t)GRADIENT_PARTIALS];
}

static id<MTLComputePipelineState> Pipeline(
    id<MTLDevice> device, id<MTLLibrary> library, NSString *name, NSError **error
) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

static BOOL BuildMetalStation(MetalStation *station, NSError **error) {
    station->device = MTLCreateSystemDefaultDevice();
    if (!station->device) {
        return NO;
    }
    station->queue = [station->device newCommandQueue];
    id<MTLLibrary> library =
        [station->device newLibraryWithSource:MetalSource()
                                      options:nil
                                        error:error];
    if (!library) {
        return NO;
    }
    station->packForward = Pipeline(
        station->device, library, @"pack_forward", error);
    station->lossAndDY = Pipeline(
        station->device, library, @"loss_and_dy", error);
    station->reduceLoss = Pipeline(
        station->device, library, @"reduce_loss", error);
    station->packDX = Pipeline(station->device, library, @"pack_dx", error);
    station->squarePartials = Pipeline(
        station->device, library, @"square_partials", error);
    station->reducePartials = Pipeline(
        station->device, library, @"reduce_norm", error);
    station->adamw = Pipeline(station->device, library, @"adamw", error);
    return station->queue && station->packForward && station->lossAndDY &&
           station->reduceLoss && station->packDX && station->squarePartials &&
           station->reducePartials && station->adamw;
}

static id<MTLTexture> TextureForSurface(
    id<MTLDevice> device, IOSurfaceRef surface, size_t width, size_t height,
    MTLTextureUsage usage
) {
    MTLTextureDescriptor *descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:
                                  MTLPixelFormatR16Float
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = usage;
    return [device newTextureWithDescriptor:descriptor
                                  iosurface:surface
                                      plane:0];
}

static id<MTLBuffer> SharedBuffer(
    id<MTLDevice> device, size_t count, BOOL zero
) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:count * sizeof(float)
                            options:MTLResourceStorageModeShared];
    if (zero && buffer) {
        memset([buffer contents], 0, count * sizeof(float));
    }
    return buffer;
}

static BOOL InitializeTransaction(
    MetalStation *metal, TransactionState *state
) {
    state->x = SharedBuffer(metal->device, ACTIVATION_ELEMENTS, NO);
    state->target = SharedBuffer(metal->device, ACTIVATION_ELEMENTS, NO);
    state->weight = SharedBuffer(metal->device, WEIGHT_ELEMENTS, NO);
    state->first = SharedBuffer(metal->device, WEIGHT_ELEMENTS, YES);
    state->second = SharedBuffer(metal->device, WEIGHT_ELEMENTS, YES);
    state->dy = SharedBuffer(metal->device, ACTIVATION_ELEMENTS, YES);
    state->dw = SharedBuffer(metal->device, WEIGHT_ELEMENTS, YES);
    state->lossPartials =
        SharedBuffer(metal->device, LOSS_PARTIALS, YES);
    state->loss = SharedBuffer(metal->device, 1, YES);
    state->gradientPartials =
        SharedBuffer(metal->device, GRADIENT_PARTIALS, YES);
    state->gradientNorm = SharedBuffer(metal->device, 1, YES);
    state->forwardInput = CreateHalfSurface(ROWS + DIM, DIM);
    state->forwardOutput = CreateHalfSurface(ROWS, DIM);
    state->dxInput = CreateHalfSurface(ROWS + DIM, DIM);
    state->dxOutput = CreateHalfSurface(ROWS, DIM);
    state->forwardInputTexture = TextureForSurface(
        metal->device, state->forwardInput, ROWS + DIM, DIM,
        MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead);
    state->forwardOutputTexture = TextureForSurface(
        metal->device, state->forwardOutput, ROWS, DIM,
        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    state->dxInputTexture = TextureForSurface(
        metal->device, state->dxInput, ROWS + DIM, DIM,
        MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead);
    state->dxOutputTexture = TextureForSurface(
        metal->device, state->dxOutput, ROWS, DIM,
        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    if (!state->x || !state->target || !state->weight || !state->first ||
        !state->second || !state->dy || !state->dw ||
        !state->lossPartials || !state->loss || !state->gradientPartials ||
        !state->gradientNorm || !state->forwardInput ||
        !state->forwardOutput || !state->dxInput || !state->dxOutput ||
        !state->forwardInputTexture || !state->forwardOutputTexture ||
        !state->dxInputTexture || !state->dxOutputTexture) {
        return NO;
    }
    state->forwardCustody =
        CustodyForSurface(state->forwardInput, ROWS + DIM, DIM);
    state->outputCustody =
        CustodyForSurface(state->forwardOutput, ROWS, DIM);
    state->dxInputCustody =
        CustodyForSurface(state->dxInput, ROWS + DIM, DIM);
    state->dxOutputCustody =
        CustodyForSurface(state->dxOutput, ROWS, DIM);
    return YES;
}

static uint32_t NextRandom(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float RandomValue(uint32_t *state, float scale) {
    uint32_t value = NextRandom(state) >> 8;
    return (((float)value / 16777216.0f) - 0.5f) * scale;
}

static void InitializeValues(TransactionState *state) {
    uint32_t randomState = 20260727u;
    float *x = [state->x contents];
    float *target = [state->target contents];
    float *weight = [state->weight contents];
    for (size_t index = 0; index < ACTIVATION_ELEMENTS; ++index) {
        x[index] = RandomValue(&randomState, 0.25f);
    }
    for (size_t index = 0; index < WEIGHT_ELEMENTS; ++index) {
        weight[index] = RandomValue(&randomState, 0.0625f);
    }
    for (size_t index = 0; index < ACTIVATION_ELEMENTS; ++index) {
        target[index] = RandomValue(&randomState, 0.125f);
    }
}

static BOOL LoadValues(
    TransactionState *state, NSString *directory, NSError **error
) {
    NSDictionary<NSString *, id<MTLBuffer>> *buffers = @{
        @"x_f32.bin" : state->x,
        @"weight_f32.bin" : state->weight,
        @"target_f32.bin" : state->target,
    };
    NSDictionary<NSString *, NSNumber *> *sizes = @{
        @"x_f32.bin" : @(ACTIVATION_ELEMENTS * sizeof(float)),
        @"weight_f32.bin" : @(WEIGHT_ELEMENTS * sizeof(float)),
        @"target_f32.bin" : @(ACTIVATION_ELEMENTS * sizeof(float)),
    };
    for (NSString *name in buffers) {
        NSString *path = [directory stringByAppendingPathComponent:name];
        NSData *data = [NSData dataWithContentsOfFile:path
                                              options:0
                                                error:error];
        if (!data || data.length != sizes[name].unsignedLongLongValue) {
            return NO;
        }
        memcpy([buffers[name] contents], data.bytes, data.length);
    }
    return YES;
}

static BOOL CompleteCommandBuffer(
    id<MTLCommandBuffer> commandBuffer, NSError **error
) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        if (error) {
            *error = commandBuffer.error;
        }
        return NO;
    }
    return YES;
}

static BOOL EncodePackForward(
    MetalStation *metal, TransactionState *state, NSError **error,
    double *milliseconds
) {
    id<MTLCommandBuffer> commandBuffer = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:metal->packForward];
    [encoder setBuffer:state->x offset:0 atIndex:0];
    [encoder setBuffer:state->weight offset:0 atIndex:1];
    [encoder setTexture:state->forwardInputTexture atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(ROWS + DIM, DIM, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [encoder endEncoding];
    uint64_t started = mach_absolute_time();
    BOOL ok = CompleteCommandBuffer(commandBuffer, error);
    *milliseconds = Milliseconds(mach_absolute_time() - started);
    return ok;
}

static BOOL EncodeLossAndPackDX(
    MetalStation *metal, TransactionState *state, NSError **error,
    double *milliseconds
) {
    id<MTLCommandBuffer> commandBuffer = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> lossEncoder =
        [commandBuffer computeCommandEncoder];
    [lossEncoder setComputePipelineState:metal->lossAndDY];
    [lossEncoder setTexture:state->forwardOutputTexture atIndex:0];
    [lossEncoder setTexture:state->dxOutputTexture atIndex:1];
    [lossEncoder setBuffer:state->target offset:0 atIndex:0];
    [lossEncoder setBuffer:state->dy offset:0 atIndex:1];
    [lossEncoder setBuffer:state->lossPartials offset:0 atIndex:2];
    [lossEncoder dispatchThreads:MTLSizeMake(LOSS_PARTIALS * THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [lossEncoder endEncoding];

    id<MTLComputeCommandEncoder> reduceEncoder =
        [commandBuffer computeCommandEncoder];
    [reduceEncoder setComputePipelineState:metal->reduceLoss];
    [reduceEncoder setBuffer:state->lossPartials offset:0 atIndex:0];
    [reduceEncoder setBuffer:state->loss offset:0 atIndex:1];
    [reduceEncoder dispatchThreads:MTLSizeMake(THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [reduceEncoder endEncoding];

    id<MTLComputeCommandEncoder> packEncoder =
        [commandBuffer computeCommandEncoder];
    [packEncoder setComputePipelineState:metal->packDX];
    [packEncoder setTexture:state->dxOutputTexture atIndex:0];
    [packEncoder setTexture:state->dxInputTexture atIndex:1];
    [packEncoder setBuffer:state->weight offset:0 atIndex:0];
    [packEncoder dispatchThreads:MTLSizeMake(ROWS + DIM, DIM, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [packEncoder endEncoding];
    uint64_t started = mach_absolute_time();
    BOOL ok = CompleteCommandBuffer(commandBuffer, error);
    *milliseconds = Milliseconds(mach_absolute_time() - started);
    return ok;
}

static BOOL EncodeUpdate(
    MetalStation *metal, TransactionState *state, NSError **error,
    double *milliseconds
) {
    id<MTLCommandBuffer> commandBuffer = [metal->queue commandBuffer];
    id<MTLComputeCommandEncoder> squareEncoder =
        [commandBuffer computeCommandEncoder];
    [squareEncoder setComputePipelineState:metal->squarePartials];
    [squareEncoder setBuffer:state->dw offset:0 atIndex:0];
    [squareEncoder setBuffer:state->gradientPartials offset:0 atIndex:1];
    [squareEncoder
        dispatchThreads:MTLSizeMake(GRADIENT_PARTIALS * THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [squareEncoder endEncoding];

    id<MTLComputeCommandEncoder> reduceEncoder =
        [commandBuffer computeCommandEncoder];
    [reduceEncoder setComputePipelineState:metal->reducePartials];
    [reduceEncoder setBuffer:state->gradientPartials offset:0 atIndex:0];
    [reduceEncoder setBuffer:state->gradientNorm offset:0 atIndex:1];
    [reduceEncoder dispatchThreads:MTLSizeMake(THREADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [reduceEncoder endEncoding];

    const float learningRate = 3.0e-5f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1.0e-8f;
    const float weightDecay = 0.01f;
    const float clipNorm = 1.0f;
    id<MTLComputeCommandEncoder> updateEncoder =
        [commandBuffer computeCommandEncoder];
    [updateEncoder setComputePipelineState:metal->adamw];
    [updateEncoder setBuffer:state->weight offset:0 atIndex:0];
    [updateEncoder setBuffer:state->first offset:0 atIndex:1];
    [updateEncoder setBuffer:state->second offset:0 atIndex:2];
    [updateEncoder setBuffer:state->dw offset:0 atIndex:3];
    [updateEncoder setBuffer:state->gradientNorm offset:0 atIndex:4];
    [updateEncoder setBytes:&learningRate length:sizeof(float) atIndex:5];
    [updateEncoder setBytes:&beta1 length:sizeof(float) atIndex:6];
    [updateEncoder setBytes:&beta2 length:sizeof(float) atIndex:7];
    [updateEncoder setBytes:&epsilon length:sizeof(float) atIndex:8];
    [updateEncoder setBytes:&weightDecay length:sizeof(float) atIndex:9];
    [updateEncoder setBytes:&clipNorm length:sizeof(float) atIndex:10];
    [updateEncoder dispatchThreads:MTLSizeMake(WEIGHT_ELEMENTS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(THREADS, 1, 1)];
    [updateEncoder endEncoding];
    uint64_t started = mach_absolute_time();
    BOOL ok = CompleteCommandBuffer(commandBuffer, error);
    *milliseconds = Milliseconds(mach_absolute_time() - started);
    return ok;
}

static BOOL RunStep(
    ANEModel *ane, MetalStation *metal, TransactionState *state,
    uint64_t generation, BOOL applyUpdate, StepReceipt *receipt, NSError **error
) {
    uint64_t joinedStarted = mach_absolute_time();
    state->forwardCustody.active_writer = THESEUS_WRITER_METAL;
    if (!EncodePackForward(
            metal, state, error, &receipt->packForwardMilliseconds)) {
        return NO;
    }
    state->forwardCustody.active_writer = THESEUS_WRITER_NONE;
    state->forwardCustody.generation = generation;
    state->forwardCustody.active_readers = THESEUS_READER_ANE;
    state->outputCustody.active_writer = THESEUS_WRITER_ANE;
    uint64_t started = mach_absolute_time();
    if (!EvaluateANE(
            ane, state->forwardInput, state->forwardOutput, error)) {
        return NO;
    }
    receipt->aneForwardMilliseconds =
        Milliseconds(mach_absolute_time() - started);
    state->forwardCustody.active_readers = 0;
    state->outputCustody.active_writer = THESEUS_WRITER_NONE;
    state->outputCustody.generation = generation;

    state->outputCustody.active_readers = THESEUS_READER_METAL;
    state->dxInputCustody.active_writer = THESEUS_WRITER_METAL;
    if (!EncodeLossAndPackDX(
            metal, state, error, &receipt->lossAndPackDXMilliseconds)) {
        return NO;
    }
    state->outputCustody.active_readers = 0;
    state->dxInputCustody.active_writer = THESEUS_WRITER_NONE;
    state->dxInputCustody.generation = generation;
    state->dxInputCustody.active_readers = THESEUS_READER_ANE;
    state->dxOutputCustody.active_writer = THESEUS_WRITER_ANE;

    __block BOOL dxOK = NO;
    __block NSError *dxError = nil;
    __block double dxMilliseconds = 0.0;
    dispatch_group_t group = dispatch_group_create();
    dispatch_group_async(
        group, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            uint64_t dxStarted = mach_absolute_time();
            dxOK = EvaluateANE(
                ane, state->dxInput, state->dxOutput, &dxError);
            dxMilliseconds = Milliseconds(mach_absolute_time() - dxStarted);
        });
    uint64_t dwStarted = mach_absolute_time();
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        DIM, DIM, ROWS, 1.0f,
        (const float *)[state->x contents], DIM,
        (const float *)[state->dy contents], DIM,
        0.0f, (float *)[state->dw contents], DIM);
    receipt->cpuDWMilliseconds =
        Milliseconds(mach_absolute_time() - dwStarted);
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    if (!dxOK) {
        if (error) {
            *error = dxError;
        }
        return NO;
    }
    receipt->aneDXMilliseconds = dxMilliseconds;
    receipt->concurrentDXDWMilliseconds =
        fmax(receipt->aneDXMilliseconds, receipt->cpuDWMilliseconds);
    state->dxInputCustody.active_readers = 0;
    state->dxOutputCustody.active_writer = THESEUS_WRITER_NONE;
    state->dxOutputCustody.generation = generation;

    if (applyUpdate) {
        if (!EncodeUpdate(
                metal, state, error, &receipt->metalUpdateMilliseconds)) {
            return NO;
        }
        receipt->gradientNorm =
            ((float *)[state->gradientNorm contents])[0];
    } else {
        receipt->metalUpdateMilliseconds = 0.0;
        receipt->gradientNorm = cblas_snrm2(
            (int)WEIGHT_ELEMENTS,
            (const float *)[state->dw contents],
            1);
    }
    receipt->loss = ((float *)[state->loss contents])[0];
    receipt->joinedMilliseconds =
        Milliseconds(mach_absolute_time() - joinedStarted);
    return YES;
}

static BOOL AllFinite(const float *values, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) {
            return NO;
        }
    }
    return YES;
}

static NSData *SurfaceData(IOSurfaceRef surface, size_t bytes) {
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    NSData *data = [NSData dataWithBytes:IOSurfaceGetBaseAddress(surface)
                                 length:bytes];
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return data;
}

static BOOL WriteBuffer(NSString *path, id<MTLBuffer> buffer, size_t bytes) {
    NSData *data = [NSData dataWithBytes:[buffer contents] length:bytes];
    return [data writeToFile:path atomically:YES];
}

static NSDictionary *Summary(NSArray<NSNumber *> *samples) {
    NSArray<NSNumber *> *sorted = [samples
        sortedArrayUsingComparator:^NSComparisonResult(
            NSNumber *left, NSNumber *right
        ) {
            return [left compare:right];
        }];
    double sum = 0.0;
    for (NSNumber *sample in samples) {
        sum += sample.doubleValue;
    }
    NSUInteger count = samples.count;
    NSUInteger p95Index =
        MIN(count - 1, (NSUInteger)ceil((double)count * 0.95) - 1);
    double median = count % 2
        ? [sorted[count / 2] doubleValue]
        : 0.5 * (
            [sorted[count / 2 - 1] doubleValue] +
            [sorted[count / 2] doubleValue]);
    return @{
        @"count" : @(count),
        @"minimum_milliseconds" : sorted.firstObject,
        @"median_milliseconds" : @(median),
        @"mean_milliseconds" : @(sum / (double)count),
        @"p95_milliseconds" : sorted[p95Index],
        @"maximum_milliseconds" : sorted.lastObject,
    };
}

static void AddTiming(
    NSMutableDictionary<NSString *, NSMutableArray<NSNumber *> *> *timings,
    NSString *key, double value
) {
    [timings[key] addObject:@(value)];
}

static double Mean(NSArray<NSNumber *> *samples) {
    double total = 0.0;
    for (NSNumber *sample in samples) {
        total += sample.doubleValue;
    }
    return total / (double)samples.count;
}

static BOOL CopyStateOut(
    TransactionState *state, float *weight, float *first, float *second
) {
    memcpy(weight, [state->weight contents], WEIGHT_ELEMENTS * sizeof(float));
    memcpy(first, [state->first contents], WEIGHT_ELEMENTS * sizeof(float));
    memcpy(second, [state->second contents], WEIGHT_ELEMENTS * sizeof(float));
    return YES;
}

static void RestoreState(
    TransactionState *state, const float *weight, const float *first,
    const float *second
) {
    memcpy([state->weight contents], weight, WEIGHT_ELEMENTS * sizeof(float));
    memcpy([state->first contents], first, WEIGHT_ELEMENTS * sizeof(float));
    memcpy([state->second contents], second, WEIGHT_ELEMENTS * sizeof(float));
}

static BOOL WriteJSON(NSString *path, NSDictionary *report, NSError **error) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:report
                                                   options:NSJSONWritingPrettyPrinted |
                                                           NSJSONWritingSortedKeys
                                                     error:error];
    if (!data) {
        return NO;
    }
    NSMutableData *withNewline = [data mutableCopy];
    const uint8_t newline = '\n';
    [withNewline appendBytes:&newline length:1];
    return [withNewline writeToFile:path options:NSDataWritingAtomic error:error];
}

static BOOL WaitForGo(NSString *readyPath, NSString *goPath) {
    if (!readyPath.length && !goPath.length) return YES;
    if (!readyPath.length || !goPath.length) return NO;
    if (![@"ready\n" writeToFile:readyPath
                        atomically:YES
                          encoding:NSUTF8StringEncoding
                             error:nil]) {
        return NO;
    }
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:30.0];
    while (![[NSFileManager defaultManager] fileExistsAtPath:goPath]) {
        if ([deadline timeIntervalSinceNow] <= 0.0) return NO;
        usleep(1000);
    }
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&Timebase);
        NSString *outputPath = nil;
        NSString *artifactDirectory = nil;
        NSString *inputDirectory = nil;
        NSString *readyPath = nil;
        NSString *goPath = nil;
        int steps = 64;
        int warmup = 2;
        BOOL gradientOnly = NO;
        for (int index = 1; index < argc; ++index) {
            if (!strcmp(argv[index], "--out") && index + 1 < argc) {
                outputPath = [NSString stringWithUTF8String:argv[++index]];
            } else if (
                !strcmp(argv[index], "--artifact-dir") && index + 1 < argc
            ) {
                artifactDirectory =
                    [NSString stringWithUTF8String:argv[++index]];
            } else if (
                !strcmp(argv[index], "--input-dir") && index + 1 < argc
            ) {
                inputDirectory =
                    [NSString stringWithUTF8String:argv[++index]];
            } else if (
                !strcmp(argv[index], "--ready-file") && index + 1 < argc
            ) {
                readyPath = [NSString stringWithUTF8String:argv[++index]];
            } else if (
                !strcmp(argv[index], "--go-file") && index + 1 < argc
            ) {
                goPath = [NSString stringWithUTF8String:argv[++index]];
            } else if (!strcmp(argv[index], "--steps") && index + 1 < argc) {
                steps = atoi(argv[++index]);
            } else if (!strcmp(argv[index], "--warmup") && index + 1 < argc) {
                warmup = atoi(argv[++index]);
            } else if (!strcmp(argv[index], "--gradient-only")) {
                gradientOnly = YES;
            } else {
                fprintf(stderr, "unknown_or_incomplete_argument:%s\n", argv[index]);
                return 2;
            }
        }
        if (!outputPath || !artifactDirectory || steps < 2 || warmup < 0) {
            fprintf(stderr, "usage: %s --out REPORT --artifact-dir DIR "
                            "[--input-dir DIR] "
                            "[--steps 64] [--warmup 2] "
                            "[--gradient-only] "
                            "[--ready-file PATH --go-file PATH]\n", argv[0]);
            return 2;
        }
        NSError *error = nil;
        [[NSFileManager defaultManager]
            createDirectoryAtPath:artifactDirectory
      withIntermediateDirectories:YES
                       attributes:nil
                            error:&error];
        if (error) {
            fprintf(stderr, "artifact_directory_failed:%s\n",
                    error.localizedDescription.UTF8String);
            return 3;
        }
        if (!LoadPrivateANE()) {
            fprintf(stderr, "private_ane_classes_missing\n");
            return 4;
        }
        MetalStation metal = {0};
        if (!BuildMetalStation(&metal, &error)) {
            fprintf(stderr, "metal_station_build_failed:%s\n",
                    error.localizedDescription.UTF8String);
            return 5;
        }
        uint64_t compileStarted = mach_absolute_time();
        ANEModel *ane = CompileANEModel(&error);
        double compileMilliseconds =
            Milliseconds(mach_absolute_time() - compileStarted);
        if (!ane) {
            fprintf(stderr, "ane_compile_failed:%s\n",
                    error.localizedDescription.UTF8String);
            return 6;
        }
        TransactionState state = {0};
        if (!InitializeTransaction(&metal, &state)) {
            fprintf(stderr, "transaction_allocation_failed\n");
            FreeANEModel(ane);
            return 7;
        }
        if (inputDirectory) {
            if (!LoadValues(&state, inputDirectory, &error)) {
                fprintf(stderr, "input_load_failed:%s\n",
                        error.localizedDescription.UTF8String);
                return 8;
            }
        } else {
            InitializeValues(&state);
        }
        BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);

        size_t stateBytes = WEIGHT_ELEMENTS * sizeof(float);
        float *initialWeight = malloc(stateBytes);
        float *initialFirst = malloc(stateBytes);
        float *initialSecond = malloc(stateBytes);
        CopyStateOut(
            &state, initialWeight, initialFirst, initialSecond);
        for (int index = 0; index < warmup; ++index) {
            RestoreState(
                &state, initialWeight, initialFirst, initialSecond);
            StepReceipt ignored = {0};
            if (!RunStep(
                    ane, &metal, &state, 0, !gradientOnly,
                    &ignored, &error)) {
                fprintf(stderr, "warmup_failed:%s\n",
                        error.localizedDescription.UTF8String);
                return 9;
            }
        }
        RestoreState(&state, initialWeight, initialFirst, initialSecond);
        if (!WaitForGo(readyPath, goPath)) {
            fprintf(stderr, "gradient_barrier_failed\n");
            return 9;
        }

        NSArray<NSString *> *timingKeys = @[
            @"pack_forward", @"ane_forward", @"loss_and_pack_dx",
            @"concurrent_dx_dw", @"ane_dx", @"cpu_dw", @"metal_update",
            @"joined"
        ];
        NSMutableDictionary *timings = [NSMutableDictionary dictionary];
        for (NSString *key in timingKeys) {
            timings[key] = [NSMutableArray arrayWithCapacity:(NSUInteger)steps];
        }
        NSMutableArray<NSNumber *> *lossPrefix = [NSMutableArray array];
        BOOL allFinite = YES;
        BOOL saveReloadExact = NO;
        BOOL generationConserved = YES;
        float *replayWeight = NULL;
        float *replayFirst = NULL;
        float *replaySecond = NULL;
        NSData *firstOutput = nil;
        NSData *firstDX = nil;
        NSData *firstDW = nil;

        for (int step = 0; step < steps; ++step) {
            uint64_t generation = (uint64_t)step;
            StepReceipt receipt = {0};
            if (!RunStep(
                    ane, &metal, &state, generation, !gradientOnly,
                    &receipt, &error)) {
                fprintf(stderr, "step_%d_failed:%s\n", step + 1,
                        error.localizedDescription.UTF8String);
                return 10;
            }
            AddTiming(timings, @"pack_forward",
                      receipt.packForwardMilliseconds);
            AddTiming(timings, @"ane_forward",
                      receipt.aneForwardMilliseconds);
            AddTiming(timings, @"loss_and_pack_dx",
                      receipt.lossAndPackDXMilliseconds);
            AddTiming(timings, @"concurrent_dx_dw",
                      receipt.concurrentDXDWMilliseconds);
            AddTiming(timings, @"ane_dx", receipt.aneDXMilliseconds);
            AddTiming(timings, @"cpu_dw", receipt.cpuDWMilliseconds);
            AddTiming(timings, @"metal_update",
                      receipt.metalUpdateMilliseconds);
            AddTiming(timings, @"joined", receipt.joinedMilliseconds);
            if (lossPrefix.count < 8) {
                [lossPrefix addObject:@(receipt.loss)];
            }
            allFinite = allFinite && isfinite(receipt.loss) &&
                isfinite(receipt.gradientNorm) &&
                AllFinite([state.weight contents], WEIGHT_ELEMENTS) &&
                AllFinite([state.first contents], WEIGHT_ELEMENTS) &&
                AllFinite([state.second contents], WEIGHT_ELEMENTS) &&
                AllFinite([state.dw contents], WEIGHT_ELEMENTS);
            generationConserved = generationConserved &&
                state.forwardCustody.generation == generation &&
                state.outputCustody.generation == generation &&
                state.dxInputCustody.generation == generation &&
                state.dxOutputCustody.generation == generation;
            if (step == 0) {
                firstOutput = SurfaceData(
                    state.forwardOutput,
                    ACTIVATION_ELEMENTS * sizeof(_Float16));
                firstDX = SurfaceData(
                    state.dxOutput,
                    ACTIVATION_ELEMENTS * sizeof(_Float16));
                firstDW = [NSData dataWithBytes:[state.dw contents]
                                        length:stateBytes];
            }
            if (step + 1 == steps / 2) {
                replayWeight = malloc(stateBytes);
                replayFirst = malloc(stateBytes);
                replaySecond = malloc(stateBytes);
                CopyStateOut(
                    &state, replayWeight, replayFirst, replaySecond);
                NSString *snapshotPath = [artifactDirectory
                    stringByAppendingPathComponent:@"checkpoint_snapshot.bin"];
                NSMutableData *snapshot =
                    [NSMutableData dataWithCapacity:stateBytes * 3];
                [snapshot appendBytes:replayWeight length:stateBytes];
                [snapshot appendBytes:replayFirst length:stateBytes];
                [snapshot appendBytes:replaySecond length:stateBytes];
                [snapshot writeToFile:snapshotPath atomically:YES];
                NSData *loaded = [NSData dataWithContentsOfFile:snapshotPath];
                saveReloadExact = loaded.length == stateBytes * 3 &&
                    memcmp(loaded.bytes, replayWeight, stateBytes) == 0 &&
                    memcmp(
                        (const uint8_t *)loaded.bytes + stateBytes,
                        replayFirst, stateBytes) == 0 &&
                    memcmp(
                        (const uint8_t *)loaded.bytes + stateBytes * 2,
                        replaySecond, stateBytes) == 0;
                [[NSFileManager defaultManager]
                    removeItemAtPath:snapshotPath error:nil];
            }
        }

        float *finalWeight = malloc(stateBytes);
        float *finalFirst = malloc(stateBytes);
        float *finalSecond = malloc(stateBytes);
        CopyStateOut(&state, finalWeight, finalFirst, finalSecond);
        RestoreState(&state, replayWeight, replayFirst, replaySecond);
        StepReceipt replayReceiptA = {0};
        BOOL replayAOK = RunStep(
            ane, &metal, &state, (uint64_t)(steps / 2),
            !gradientOnly, &replayReceiptA, &error);
        float *replayAWeight = malloc(stateBytes);
        float *replayAFirst = malloc(stateBytes);
        float *replayASecond = malloc(stateBytes);
        CopyStateOut(
            &state, replayAWeight, replayAFirst, replayASecond);
        NSData *replayADX = SurfaceData(
            state.dxOutput, ACTIVATION_ELEMENTS * sizeof(_Float16));
        NSData *replayADW = [NSData dataWithBytes:[state.dw contents]
                                          length:stateBytes];
        RestoreState(&state, replayWeight, replayFirst, replaySecond);
        StepReceipt replayReceiptB = {0};
        BOOL replayBOK = RunStep(
            ane, &metal, &state, (uint64_t)(steps / 2),
            !gradientOnly, &replayReceiptB, &error);
        NSData *replayBDX = SurfaceData(
            state.dxOutput, ACTIVATION_ELEMENTS * sizeof(_Float16));
        BOOL replayExact = replayAOK && replayBOK &&
            replayReceiptA.loss == replayReceiptB.loss &&
            memcmp(replayAWeight, [state.weight contents], stateBytes) == 0 &&
            memcmp(replayAFirst, [state.first contents], stateBytes) == 0 &&
            memcmp(replayASecond, [state.second contents], stateBytes) == 0 &&
            [replayADX isEqualToData:replayBDX] &&
            memcmp(replayADW.bytes, [state.dw contents], stateBytes) == 0;
        RestoreState(&state, finalWeight, finalFirst, finalSecond);

        NSString *weightPath = [artifactDirectory
            stringByAppendingPathComponent:@"final_weight_f32.bin"];
        NSString *firstPath = [artifactDirectory
            stringByAppendingPathComponent:@"final_first_moment_f32.bin"];
        NSString *secondPath = [artifactDirectory
            stringByAppendingPathComponent:@"final_second_moment_f32.bin"];
        NSString *outputArtifact = [artifactDirectory
            stringByAppendingPathComponent:@"step1_output_f16.bin"];
        NSString *dxArtifact = [artifactDirectory
            stringByAppendingPathComponent:@"step1_dx_f16.bin"];
        NSString *dwArtifact = [artifactDirectory
            stringByAppendingPathComponent:@"step1_dw_f32.bin"];
        BOOL artifactsWritten =
            WriteBuffer(weightPath, state.weight, stateBytes) &&
            WriteBuffer(firstPath, state.first, stateBytes) &&
            WriteBuffer(secondPath, state.second, stateBytes) &&
            [firstOutput writeToFile:outputArtifact atomically:YES] &&
            [firstDX writeToFile:dxArtifact atomically:YES] &&
            [firstDW writeToFile:dwArtifact atomically:YES];

        NSMutableDictionary *timingSummary = [NSMutableDictionary dictionary];
        NSMutableDictionary *stationMeans = [NSMutableDictionary dictionary];
        for (NSString *key in timingKeys) {
            timingSummary[key] = Summary(timings[key]);
            stationMeans[key] = @(Mean(timings[key]));
        }
        NSDictionary *report = @{
            @"policy" : @"project_theseus_native_ane_cpu_metal_projection_triad_v1",
            @"trigger_state" : @"INCONCLUSIVE_IMPLEMENTATION",
            @"production_eligible" : @NO,
            @"shape" : @{
                @"logical_batch" : @4,
                @"sequence" : @512,
                @"rows" : @ROWS,
                @"input_channels" : @DIM,
                @"output_channels" : @DIM,
                @"projection" : @"decoder_self_attention_q_proj",
            },
            @"execution" : @{
                @"mode" : (
                    gradientOnly
                        ? @"gradient_contribution_only"
                        : @"optimizer_transaction"
                ),
                @"gradient_transactions" : @(steps),
                @"optimizer_steps" : @(gradientOnly ? 0 : steps),
                @"warmup_steps" : @(warmup),
                @"input_source" : (
                    inputDirectory
                        ? @"frozen_setup_binaries"
                        : @"native_lcg_fallback"
                ),
                @"final_generation" : @(steps),
                @"ane_compile_milliseconds" : @(compileMilliseconds),
                @"blas_threading" : @(BLASGetThreading()),
                @"ane_forward" : @"private_compile_once_dynamic_matmul",
                @"ane_input_gradient" : @"private_compile_once_dynamic_matmul",
                @"cpu_weight_gradient" : @"single_thread_accelerate_sgemm_shared_buffers",
                @"metal_remainder" : (
                    gradientOnly
                        ? @"loss_dy_no_local_optimizer"
                        : @"loss_dy_reductions_clip_adamw"
                ),
            },
            @"custody" : @{
                @"single_generation_conserved" : @(generationConserved),
                @"one_fp32_gradient_accumulator" : @YES,
                @"one_fp32_adamw_update_per_step" : @(!gradientOnly),
                @"local_optimizer_update_forbidden" : @(gradientOnly),
                @"hot_step_python_or_numpy" : @NO,
                @"intermediate_host_tensor_copy" : @NO,
                @"shared_fp32_x_dy_buffers_consumed_by_accelerate" : @YES,
                @"metal_packs_ane_surfaces" : @YES,
                @"canonical_checkpoint_mutated" : @NO,
            },
            @"stability" : @{
                @"all_tensors_finite" : @(allFinite),
                @"save_reload_exact" : @(saveReloadExact),
                @"replay_exact" : @(replayExact),
                @"loss_prefix" : lossPrefix,
            },
            @"timing" : @{
                @"summaries_milliseconds" : timingSummary,
                @"station_means_milliseconds" : stationMeans,
            },
            @"artifacts" : @{
                @"written" : @(artifactsWritten),
                @"final_weight_f32" : weightPath,
                @"final_first_moment_f32" : firstPath,
                @"final_second_moment_f32" : secondPath,
                @"step1_output_f16" : outputArtifact,
                @"step1_dx_f16" : dxArtifact,
                @"step1_dw_f32" : dwArtifact,
            },
            @"open_gates" : @[
                @"matched_mlx_numerical_parity",
                @"matched_mlx_joined_wall",
                @"sustained_resource_and_thermal_qualification",
                @"real_metal_attention_pointer_loss_remainder",
                @"sampler_and_objective_mass_conservation",
                @"independent_gate_audit",
            ],
            @"claim_scope" : (
                gradientOnly
                    ? @"One deterministic q_proj-shaped native gradient contribution. "
                      "No optimizer, transformer, convergence, utility, capability, "
                      "or production training speedup claim is allowed."
                    : @"One deterministic q_proj-shaped native optimizer transaction. "
                      "No transformer, convergence, utility, capability, or production "
                      "training speedup claim is allowed."
            ),
            @"public_benchmark_rows_read" : @0,
            @"external_inference_calls" : @0,
        };
        if (!WriteJSON(outputPath, report, &error)) {
            fprintf(stderr, "report_write_failed:%s\n",
                    error.localizedDescription.UTF8String);
            return 11;
        }
        NSData *stdoutData = [NSJSONSerialization
            dataWithJSONObject:report options:0 error:nil];
        fwrite(stdoutData.bytes, 1, stdoutData.length, stdout);
        fputc('\n', stdout);

        CFRelease(state.forwardInput);
        CFRelease(state.forwardOutput);
        CFRelease(state.dxInput);
        CFRelease(state.dxOutput);
        FreeANEModel(ane);
        free(initialWeight);
        free(initialFirst);
        free(initialSecond);
        free(replayWeight);
        free(replayFirst);
        free(replaySecond);
        free(finalWeight);
        free(finalFirst);
        free(finalSecond);
        free(replayAWeight);
        free(replayAFirst);
        free(replayASecond);
        return artifactsWritten && allFinite && saveReloadExact &&
                       replayExact && generationConserved
            ? 0
            : 12;
    }
}
