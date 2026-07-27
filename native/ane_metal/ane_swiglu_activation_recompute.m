/*
 * Exact-shape ANE recomputation mechanics for the two SwiGLU input projections.
 *
 * Shape: [batch=4, sequence=512, hidden=512] -> gate/up width 1536 each.
 * Metal packs the current FP32 activation and weights into one IOSurface.
 * The private ANE runtime computes the fused 3072-channel projection.  The
 * output is parity-checked against Accelerate, but no production checkpoint is
 * touched and this source does not claim integration with MLX autograd.
 *
 * Private-runtime techniques are independently reimplemented from maderix/ANE
 * (MIT); see THIRD_PARTY_NOTICES.md.
 */

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <dlfcn.h>
#include <mach/mach_time.h>
#include <math.h>

#include "ane_metal_surface_contract.h"

#define ROWS 2048
#define INPUT_DIM 512
#define FF_DIM 1536
#define OUTPUT_DIM 512
#define PROJECTION_COUNT 6
#define INPUT_ELEMENTS ((size_t)ROWS * INPUT_DIM)
#define ONE_WEIGHT_ELEMENTS ((size_t)OUTPUT_DIM * INPUT_DIM)
#define WEIGHT_ELEMENTS (PROJECTION_COUNT * ONE_WEIGHT_ELEMENTS)
#define ONE_OUTPUT_ELEMENTS ((size_t)ROWS * OUTPUT_DIM)
#define OUTPUT_ELEMENTS (PROJECTION_COUNT * ONE_OUTPUT_ELEMENTS)

static Class DescriptorClass;
static Class InMemoryModelClass;
static Class RequestClass;
static Class IOSurfaceObjectClass;
static mach_timebase_info_data_t Timebase;

typedef struct {
    id model;
    NSString *temporaryDirectory;
} ANEModel;

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

static NSString *ProjectionMIL(void) {
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, "
         "{\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [mil appendFormat:
        @"    func main<ios18>(tensor<fp16, [1,%d,1,%d]> packed) {\n",
        INPUT_DIM, ROWS + OUTPUT_DIM];
    [mil appendString:
        @"        tensor<int32,[4]> ba = const()[name=string(\"ba\"),"
         "val=tensor<int32,[4]>([0,0,0,0])];\n"];
    [mil appendFormat:
        @"        tensor<int32,[4]> sa = const()[name=string(\"sa\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        INPUT_DIM, ROWS];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> act = "
         "slice_by_size(x=packed,begin=ba,size=sa)[name=string(\"act\")];\n",
        INPUT_DIM, ROWS];
    [mil appendFormat:
        @"        tensor<int32,[4]> bw = const()[name=string(\"bw\"),"
         "val=tensor<int32,[4]>([0,0,0,%d])];\n",
        ROWS];
    [mil appendFormat:
        @"        tensor<int32,[4]> sw = const()[name=string(\"sw\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        INPUT_DIM, OUTPUT_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> wt = "
         "slice_by_size(x=packed,begin=bw,size=sw)[name=string(\"wt\")];\n",
        INPUT_DIM, OUTPUT_DIM];
    [mil appendFormat:
        @"        tensor<int32,[4]> ra = const()[name=string(\"ra\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n",
        INPUT_DIM, ROWS];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> a2 = "
         "reshape(shape=ra,x=act)[name=string(\"a2\")];\n",
        INPUT_DIM, ROWS];
    [mil appendString:
        @"        tensor<int32,[4]> pm = const()[name=string(\"pm\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> a3 = "
         "transpose(perm=pm,x=a2)[name=string(\"a3\")];\n",
        ROWS, INPUT_DIM];
    [mil appendFormat:
        @"        tensor<int32,[4]> rw = const()[name=string(\"rw\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n",
        INPUT_DIM, OUTPUT_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> w = "
         "reshape(shape=rw,x=wt)[name=string(\"w\")];\n",
        INPUT_DIM, OUTPUT_DIM];
    [mil appendString:
        @"        bool bf = const()[name=string(\"bf\"),val=bool(false)];\n"];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> yh = "
         "matmul(transpose_x=bf,transpose_y=bf,x=a3,y=w)"
         "[name=string(\"projection\")];\n",
        ROWS, OUTPUT_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> yt = "
         "transpose(perm=pm,x=yh)[name=string(\"yt\")];\n",
        OUTPUT_DIM, ROWS];
    [mil appendFormat:
        @"        tensor<int32,[4]> ro = const()[name=string(\"ro\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        OUTPUT_DIM, ROWS];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> out = "
         "reshape(shape=ro,x=yt)[name=string(\"out\")];\n",
        OUTPUT_DIM, ROWS];
    [mil appendString:@"    } -> (out);\n}\n"];
    return mil;
}

static ANEModel *CompileANEModel(NSError **error) {
    NSData *milData =
        [ProjectionMIL() dataUsingEncoding:NSUTF8StringEncoding];
    id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        DescriptorClass,
        @selector(modelWithMILText:weights:optionsPlist:),
        milData, @{}, nil);
    if (!descriptor) return NULL;
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
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:),
            21, @{}, error) ||
        !((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:),
            21, @{}, error)) {
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
    if (!model) return;
    NSError *error = nil;
    ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
        model->model, @selector(unloadWithQoS:error:), 21, &error);
    [[NSFileManager defaultManager]
        removeItemAtPath:model->temporaryDirectory error:nil];
    free(model);
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

static BOOL WaitForGo(NSString *readyPath, NSString *goPath) {
    if (!readyPath.length || !goPath.length) return YES;
    [@"ready\n" writeToFile:readyPath
                 atomically:YES
                   encoding:NSUTF8StringEncoding
                      error:nil];
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:30.0];
    while (![[NSFileManager defaultManager] fileExistsAtPath:goPath]) {
        if ([deadline timeIntervalSinceNow] <= 0.0) return NO;
        usleep(1000);
    }
    return YES;
}

static double Mean(const double *values, int count) {
    double total = 0.0;
    for (int index = 0; index < count; ++index) total += values[index];
    return total / count;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&Timebase);
        int warmup = 3;
        int repetitions = 24;
        NSString *readyPath = @"";
        NSString *goPath = @"";
        for (int index = 1; index + 1 < argc; index += 2) {
            NSString *key = [NSString stringWithUTF8String:argv[index]];
            NSString *value = [NSString stringWithUTF8String:argv[index + 1]];
            if ([key isEqualToString:@"--warmup"]) warmup = value.intValue;
            else if ([key isEqualToString:@"--repetitions"])
                repetitions = value.intValue;
            else if ([key isEqualToString:@"--ready-file"]) readyPath = value;
            else if ([key isEqualToString:@"--go-file"]) goPath = value;
            else {
                fprintf(stderr, "unknown_argument\n");
                return 2;
            }
        }
        if (warmup < 1 || repetitions < 2 || !LoadPrivateANE()) return 3;

        NSError *error = nil;
        uint64_t compileStarted = mach_absolute_time();
        ANEModel *ane = CompileANEModel(&error);
        double compileMilliseconds =
            Milliseconds(mach_absolute_time() - compileStarted);
        if (!ane) {
            fprintf(stderr, "ane_compile_failed:%s\n",
                    error.description.UTF8String);
            return 4;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSString *source = [NSString stringWithFormat:
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void pack(device const float *x [[buffer(0)]],"
             " device const float *w [[buffer(1)]],"
             " texture2d<half,access::write> out [[texture(0)]],"
             " uint2 gid [[thread_position_in_grid]]) {"
             " if (gid.y >= %d || gid.x >= %d) return;"
             " float v = gid.x < %d ? x[gid.x * %d + gid.y]"
             " : w[(gid.x - %d) * %d + gid.y];"
             " out.write(half4(half(v)), gid); }\n",
            INPUT_DIM, ROWS + OUTPUT_DIM, ROWS, INPUT_DIM, ROWS, INPUT_DIM];
        id<MTLLibrary> library =
            [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"pack"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return 5;

        id<MTLBuffer> x = [device newBufferWithLength:
            INPUT_ELEMENTS * sizeof(float)
            options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight = [device newBufferWithLength:
            WEIGHT_ELEMENTS * sizeof(float)
            options:MTLResourceStorageModeShared];
        IOSurfaceRef packedSurfaces[PROJECTION_COUNT] = {0};
        IOSurfaceRef outputSurfaces[PROJECTION_COUNT] = {0};
        id<MTLTexture> packedTextures[PROJECTION_COUNT] = {nil};
        for (int projection = 0; projection < PROJECTION_COUNT; ++projection) {
            packedSurfaces[projection] =
                CreateHalfSurface(ROWS + OUTPUT_DIM, INPUT_DIM);
            outputSurfaces[projection] =
                CreateHalfSurface(ROWS, OUTPUT_DIM);
            packedTextures[projection] = TextureForSurface(
                device, packedSurfaces[projection],
                ROWS + OUTPUT_DIM, INPUT_DIM,
                MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
            if (!packedSurfaces[projection] ||
                !outputSurfaces[projection] ||
                !packedTextures[projection]) return 6;
        }
        if (!x || !weight) return 6;
        float *xValues = x.contents;
        float *weightValues = weight.contents;
        for (size_t i = 0; i < INPUT_ELEMENTS; ++i)
            xValues[i] = sinf((float)(i % 997) * 0.013f) * 0.1f;
        for (size_t i = 0; i < WEIGHT_ELEMENTS; ++i)
            weightValues[i] = cosf((float)(i % 769) * 0.017f) * 0.02f;

        double *packTimes = calloc((size_t)repetitions, sizeof(double));
        double *aneTimes = calloc((size_t)repetitions, sizeof(double));
        double *joinedTimes = calloc((size_t)repetitions, sizeof(double));
        uint64_t generation = 0;
        for (int iteration = -warmup; iteration < repetitions; ++iteration) {
            uint64_t joinedStarted = mach_absolute_time();
            id<MTLCommandBuffer> command = [queue commandBuffer];
            for (int projection = 0;
                 projection < PROJECTION_COUNT;
                 ++projection) {
                id<MTLComputeCommandEncoder> encoder =
                    [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:x offset:0 atIndex:0];
                [encoder setBuffer:weight
                            offset:(size_t)projection *
                                   ONE_WEIGHT_ELEMENTS * sizeof(float)
                           atIndex:1];
                [encoder setTexture:packedTextures[projection] atIndex:0];
                [encoder dispatchThreads:
                    MTLSizeMake(ROWS + OUTPUT_DIM, INPUT_DIM, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
                [encoder endEncoding];
            }
            uint64_t packStarted = mach_absolute_time();
            [command commit];
            [command waitUntilCompleted];
            double packMs = Milliseconds(mach_absolute_time() - packStarted);
            uint64_t aneStarted = mach_absolute_time();
            BOOL ok = YES;
            for (int projection = 0;
                 projection < PROJECTION_COUNT && ok;
                 ++projection) {
                ok = EvaluateANE(
                    ane, packedSurfaces[projection],
                    outputSurfaces[projection], &error);
            }
            double aneMs = Milliseconds(mach_absolute_time() - aneStarted);
            if (!ok) return 7;
            generation += 1;
            if (iteration >= 0) {
                packTimes[iteration] = packMs;
                aneTimes[iteration] = aneMs;
                joinedTimes[iteration] =
                    Milliseconds(mach_absolute_time() - joinedStarted);
            }
        }

        float *reference = malloc(ONE_OUTPUT_ELEMENTS * sizeof(float));
        size_t mismatchCount = 0;
        float maximumAbsoluteDelta = 0.0f;
        for (int projection = 0;
             projection < PROJECTION_COUNT;
             ++projection) {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                ROWS, OUTPUT_DIM, INPUT_DIM, 1.0f,
                xValues, INPUT_DIM,
                weightValues + (size_t)projection * ONE_WEIGHT_ELEMENTS,
                INPUT_DIM, 0.0f, reference, OUTPUT_DIM);
            IOSurfaceLock(
                outputSurfaces[projection], kIOSurfaceLockReadOnly, NULL);
            const _Float16 *actual =
                IOSurfaceGetBaseAddress(outputSurfaces[projection]);
            for (int row = 0; row < ROWS; ++row) {
                for (int channel = 0; channel < OUTPUT_DIM; ++channel) {
                    size_t surfaceIndex = (size_t)channel * ROWS + row;
                    size_t referenceIndex =
                        (size_t)row * OUTPUT_DIM + channel;
                    float delta = fabsf(
                        (float)actual[surfaceIndex] -
                        reference[referenceIndex]);
                    if (delta > maximumAbsoluteDelta)
                        maximumAbsoluteDelta = delta;
                    if (delta > 0.001f) mismatchCount += 1;
                }
            }
            IOSurfaceUnlock(
                outputSurfaces[projection], kIOSurfaceLockReadOnly, NULL);
        }

        if (!WaitForGo(readyPath, goPath)) return 8;
        /*
         * Barrier runs need work after the barrier.  Repeat the already-warm
         * measured transaction so the coordinator observes joined service,
         * not compilation or initialization.
         */
        uint64_t barrierStarted = mach_absolute_time();
        for (int iteration = 0; iteration < repetitions; ++iteration) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            for (int projection = 0;
                 projection < PROJECTION_COUNT;
                 ++projection) {
                id<MTLComputeCommandEncoder> encoder =
                    [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:x offset:0 atIndex:0];
                [encoder setBuffer:weight
                            offset:(size_t)projection *
                                   ONE_WEIGHT_ELEMENTS * sizeof(float)
                           atIndex:1];
                [encoder setTexture:packedTextures[projection] atIndex:0];
                [encoder dispatchThreads:
                    MTLSizeMake(ROWS + OUTPUT_DIM, INPUT_DIM, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
                [encoder endEncoding];
            }
            [command commit];
            [command waitUntilCompleted];
            for (int projection = 0;
                 projection < PROJECTION_COUNT;
                 ++projection) {
                if (!EvaluateANE(
                        ane, packedSurfaces[projection],
                        outputSurfaces[projection], &error)) return 9;
            }
            generation += 1;
        }
        double barrierWorkMilliseconds =
            Milliseconds(mach_absolute_time() - barrierStarted);

        double packMean = Mean(packTimes, repetitions);
        double aneMean = Mean(aneTimes, repetitions);
        double joinedMean = Mean(joinedTimes, repetitions);
        double discardedMiB =
            (double)(2 * ROWS * FF_DIM * sizeof(float)) / (1024.0 * 1024.0);
        double retainedBoundaryMiB =
            (double)(ROWS * INPUT_DIM * sizeof(float)) / (1024.0 * 1024.0);
        printf(
            "{\"policy\":\"project_theseus_ane_swiglu_activation_recompute_v1\","
            "\"trigger_state\":\"%s\","
            "\"shape\":{\"rows\":%d,\"batch\":4,\"sequence\":512,"
            "\"hidden\":%d,\"ff_dim\":%d,\"output_channels\":%d},"
            "\"compile_milliseconds\":%.9f,"
            "\"runtime\":{\"pack_mean_milliseconds\":%.9f,"
            "\"ane_mean_milliseconds\":%.9f,"
            "\"joined_mean_milliseconds\":%.9f,"
            "\"barrier_work_milliseconds\":%.9f,\"repetitions\":%d},"
            "\"parity\":{\"tolerance\":0.001,"
            "\"mismatch_count\":%zu,\"maximum_absolute_delta\":%.9g},"
            "\"memory\":{\"discarded_gate_up_fp32_mib_per_layer\":%.6f,"
            "\"retained_layer_boundary_fp32_mib\":%.6f,"
            "\"maximum_twelve_layer_discarded_mib\":%.6f},"
            "\"custody\":{\"generation_count\":%llu,"
            "\"metal_pack_to_iosurface\":true,"
            "\"intermediate_python_or_numpy_round_trip\":false},"
            "\"canonical_checkpoint_mutated\":false,"
            "\"public_benchmark_rows_read\":0,"
            "\"external_inference_calls\":0,"
            "\"claim_scope\":\"Exact production-shape SwiGLU gate/up "
            "activation recomputation mechanics only; no MLX autograd "
            "integration, optimizer, throughput, memory-allocation, or "
            "capability claim.\"}\n",
            mismatchCount == 0 ? "GREEN_RECOMPUTE_MECHANICS" : "RED_PARITY",
            ROWS, INPUT_DIM, FF_DIM, PROJECTION_COUNT * OUTPUT_DIM,
            compileMilliseconds, packMean, aneMean, joinedMean,
            barrierWorkMilliseconds, repetitions,
            mismatchCount, maximumAbsoluteDelta,
            discardedMiB, retainedBoundaryMiB, discardedMiB * 12.0,
            (unsigned long long)generation);

        free(reference);
        free(packTimes);
        free(aneTimes);
        free(joinedTimes);
        for (int projection = 0;
             projection < PROJECTION_COUNT;
             ++projection) {
            CFRelease(packedSurfaces[projection]);
            CFRelease(outputSurfaces[projection]);
        }
        FreeANEModel(ane);
        return mismatchCount == 0 ? 0 : 10;
    }
}
