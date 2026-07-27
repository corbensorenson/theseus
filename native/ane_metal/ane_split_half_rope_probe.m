/*
 * Production-shape split-half RoPE proof for Project Theseus.
 *
 * This uses undocumented AppleNeuralEngine classes and is research-only.
 * Private-runtime techniques are derived from maderix/ANE under its MIT
 * license; see THIRD_PARTY_NOTICES.md.
 */

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>

#include <dlfcn.h>
#include <math.h>

#define HEADS 8
#define SEQUENCE 512
#define HEAD_DIM 64
#define HALF_HEAD_DIM (HEAD_DIM / 2)
#define ELEMENTS ((size_t)HEADS * SEQUENCE * HEAD_DIM)

static Class DescriptorClass;
static Class InMemoryModelClass;
static Class RequestClass;
static Class IOSurfaceObjectClass;

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

static IOSurfaceRef CreateHalfSurface(size_t elements) {
    const size_t bytes = elements * sizeof(_Float16);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (__bridge NSString *)kIOSurfaceWidth : @(bytes),
        (__bridge NSString *)kIOSurfaceHeight : @1,
        (__bridge NSString *)kIOSurfaceBytesPerElement : @1,
        (__bridge NSString *)kIOSurfaceBytesPerRow : @(bytes),
        (__bridge NSString *)kIOSurfaceAllocSize : @(bytes),
        (__bridge NSString *)kIOSurfacePixelFormat : @0,
    });
}

static NSData *BuildWeightBlob(const float *values, int count) {
    const int weightBytes = count * (int)sizeof(_Float16);
    const int totalBytes = 128 + weightBytes;
    uint8_t *bytes = calloc((size_t)totalBytes, 1);
    bytes[0] = 1;
    bytes[4] = 2;
    bytes[64] = 0xEF;
    bytes[65] = 0xBE;
    bytes[66] = 0xAD;
    bytes[67] = 0xDE;
    bytes[68] = 1;
    *(uint32_t *)(bytes + 72) = (uint32_t)weightBytes;
    *(uint32_t *)(bytes + 80) = 128;
    _Float16 *half = (_Float16 *)(bytes + 128);
    for (int index = 0; index < count; ++index) {
        half[index] = (_Float16)values[index];
    }
    return [NSData dataWithBytesNoCopy:bytes
                                length:(NSUInteger)totalBytes
                          freeWhenDone:YES];
}

static NSString *SplitHalfRoPEMIL(void) {
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, "
         "{\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [mil appendFormat:
        @"    func main<ios18>(tensor<fp16, [1,%d,%d,%d]> x) {\n",
        HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> cosv = const()["
         "name=string(\"cosv\"),val=tensor<fp16, [1,1,%d,%d]>("
         "BLOBFILE(path=string(\"@model_path/weights/cos.bin\"),"
         "offset=uint64(64)))];\n",
        SEQUENCE, HEAD_DIM, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> sinv = const()["
         "name=string(\"sinv\"),val=tensor<fp16, [1,1,%d,%d]>("
         "BLOBFILE(path=string(\"@model_path/weights/sin.bin\"),"
         "offset=uint64(64)))];\n",
        SEQUENCE, HEAD_DIM, SEQUENCE, HEAD_DIM];
    [mil appendString:
        @"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"),"
         "val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [mil appendFormat:
        @"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"),"
         "val=tensor<int32, [4]>([0,0,0,%d])];\n",
        HALF_HEAD_DIM];
    [mil appendFormat:
        @"        tensor<int32, [4]> hs = const()[name=string(\"hs\"),"
         "val=tensor<int32, [4]>([1,%d,%d,%d])];\n",
        HEADS, SEQUENCE, HALF_HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> first = "
         "slice_by_size(x=x,begin=b0,size=hs)[name=string(\"first\")];\n",
        HEADS, SEQUENCE, HALF_HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> second = "
         "slice_by_size(x=x,begin=b1,size=hs)[name=string(\"second\")];\n",
        HEADS, SEQUENCE, HALF_HEAD_DIM];
    [mil appendString:
        @"        fp16 neg = const()[name=string(\"neg\"),val=fp16(-1)];\n"];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> nsecond = "
         "mul(x=second,y=neg)[name=string(\"nsecond\")];\n",
        HEADS, SEQUENCE, HALF_HEAD_DIM];
    [mil appendString:
        @"        int32 axis = const()[name=string(\"axis\"),val=int32(3)];\n"
         "        bool interleave = const()[name=string(\"interleave\"),"
         "val=bool(false)];\n"];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> rotated = "
         "concat(axis=axis,interleave=interleave,values=(nsecond,first))"
         "[name=string(\"rotated\")];\n",
        HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> xc = "
         "mul(x=x,y=cosv)[name=string(\"xc\")];\n",
        HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> rs = "
         "mul(x=rotated,y=sinv)[name=string(\"rs\")];\n",
        HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> y = "
         "add(x=xc,y=rs)[name=string(\"y\")];\n",
        HEADS, SEQUENCE, HEAD_DIM];
    [mil appendString:@"    } -> (y);\n}\n"];
    return mil;
}

static id CompileModel(NSString *mil, NSDictionary *weights,
                       NSString **temporaryDirectory, NSError **error) {
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        DescriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
        milData, weights, nil);
    if (!descriptor) return nil;
    id model = ((id(*)(Class, SEL, id))objc_msgSend)(
        InMemoryModelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
    NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
        model, @selector(hexStringIdentifier));
    NSString *directory =
        [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
    NSFileManager *fileManager = [NSFileManager defaultManager];
    [fileManager removeItemAtPath:directory error:nil];
    [fileManager
        createDirectoryAtPath:
            [directory stringByAppendingPathComponent:@"weights"]
  withIntermediateDirectories:YES
                   attributes:nil
                        error:nil];
    [milData writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
              atomically:YES];
    for (NSString *path in weights) {
        NSString *relativePath =
            [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                             withString:@""];
        [weights[path][@"data"]
            writeToFile:[directory stringByAppendingPathComponent:relativePath]
             atomically:YES];
    }
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, error)) {
        return nil;
    }
    if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, error)) {
        return nil;
    }
    *temporaryDirectory = directory;
    return model;
}

static BOOL Evaluate(id model, IOSurfaceRef input, IOSurfaceRef output,
                     NSError **error) {
    id wrappedInput = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        IOSurfaceObjectClass, @selector(objectWithIOSurface:), input);
    id wrappedOutput = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
        IOSurfaceObjectClass, @selector(objectWithIOSurface:), output);
    id request =
        ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            RequestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                                     weightsBuffer:perfStats:procedureIndex:),
            @[ wrappedInput ], @[ @0 ], @[ wrappedOutput ], @[ @0 ],
            nil, nil, @0);
    return ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, request, error);
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        if (!LoadPrivateANE()) {
            fprintf(stderr, "private_ane_classes_missing\n");
            return 2;
        }
        float *cosine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
        float *sine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
        for (int position = 0; position < SEQUENCE; ++position) {
            for (int channel = 0; channel < HALF_HEAD_DIM; ++channel) {
                const float theta =
                    position /
                    powf(10000.0f, 2.0f * channel / (float)HEAD_DIM);
                const float cosValue = cosf(theta);
                const float sinValue = sinf(theta);
                cosine[position * HEAD_DIM + channel] = cosValue;
                cosine[position * HEAD_DIM + channel + HALF_HEAD_DIM] =
                    cosValue;
                sine[position * HEAD_DIM + channel] = sinValue;
                sine[position * HEAD_DIM + channel + HALF_HEAD_DIM] =
                    sinValue;
            }
        }
        NSError *error = nil;
        NSString *temporaryDirectory = nil;
        id model = CompileModel(
            SplitHalfRoPEMIL(),
            @{
                @"@model_path/weights/cos.bin" :
                    @{@"offset" : @0,
                      @"data" :
                          BuildWeightBlob(cosine, SEQUENCE * HEAD_DIM)},
                @"@model_path/weights/sin.bin" :
                    @{@"offset" : @0,
                      @"data" :
                          BuildWeightBlob(sine, SEQUENCE * HEAD_DIM)}
            },
            &temporaryDirectory, &error);
        if (!model) {
            fprintf(stderr, "ane_compile_failed: %s\n",
                    error.description.UTF8String);
            return 3;
        }
        IOSurfaceRef input = CreateHalfSurface(ELEMENTS);
        IOSurfaceRef output = CreateHalfSurface(ELEMENTS);
        if (!input || !output) {
            fprintf(stderr, "iosurface_create_failed\n");
            return 4;
        }
        IOSurfaceLock(input, 0, NULL);
        _Float16 *inputValues = IOSurfaceGetBaseAddress(input);
        for (size_t index = 0; index < ELEMENTS; ++index) {
            inputValues[index] = (_Float16)(
                sinf((float)(index % 997) * 0.013f) * 0.75f +
                cosf((float)(index % 113) * 0.021f) * 0.25f);
        }
        IOSurfaceUnlock(input, 0, NULL);
        if (!Evaluate(model, input, output, &error)) {
            fprintf(stderr, "ane_eval_failed: %s\n",
                    error.description.UTF8String);
            return 5;
        }
        IOSurfaceLock(input, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
        inputValues = IOSurfaceGetBaseAddress(input);
        const _Float16 *outputValues = IOSurfaceGetBaseAddress(output);
        size_t mismatchCount = 0;
        float maximumAbsoluteDelta = 0.0f;
        double squaredError = 0.0;
        for (int head = 0; head < HEADS; ++head) {
            for (int position = 0; position < SEQUENCE; ++position) {
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    const size_t index =
                        ((size_t)head * SEQUENCE + position) * HEAD_DIM +
                        channel;
                    const int pairChannel = channel < HALF_HEAD_DIM
                        ? channel + HALF_HEAD_DIM
                        : channel - HALF_HEAD_DIM;
                    const size_t pairIndex =
                        ((size_t)head * SEQUENCE + position) * HEAD_DIM +
                        pairChannel;
                    const _Float16 cosValue =
                        (_Float16)cosine[position * HEAD_DIM + channel];
                    const _Float16 sinValue =
                        (_Float16)sine[position * HEAD_DIM + channel];
                    const _Float16 rotated = channel < HALF_HEAD_DIM
                        ? (_Float16)(-(float)inputValues[pairIndex])
                        : inputValues[pairIndex];
                    const _Float16 productA = (_Float16)(
                        (float)inputValues[index] * (float)cosValue);
                    const _Float16 productB =
                        (_Float16)((float)rotated * (float)sinValue);
                    const _Float16 expected =
                        (_Float16)((float)productA + (float)productB);
                    const float delta =
                        fabsf((float)outputValues[index] - (float)expected);
                    if (delta > 0.001f) ++mismatchCount;
                    maximumAbsoluteDelta =
                        fmaxf(maximumAbsoluteDelta, delta);
                    squaredError += (double)delta * delta;
                }
            }
        }
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(input, kIOSurfaceLockReadOnly, NULL);
        printf(
            "{\"policy\":\"theseus_ane_split_half_rope_v1\","
            "\"heads\":%d,\"sequence\":%d,\"head_dim\":%d,"
            "\"elements\":%zu,\"tolerance\":0.001,"
            "\"mismatch_count\":%zu,\"maximum_absolute_delta\":%.9g,"
            "\"rmse\":%.9g,\"trigger_state\":\"%s\"}\n",
            HEADS, SEQUENCE, HEAD_DIM, ELEMENTS, mismatchCount,
            maximumAbsoluteDelta, sqrt(squaredError / ELEMENTS),
            mismatchCount == 0 ? "GREEN" : "RED");
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        [[NSFileManager defaultManager]
            removeItemAtPath:temporaryDirectory error:nil];
        CFRelease(input);
        CFRelease(output);
        free(cosine);
        free(sine);
        return mismatchCount == 0 ? 0 : 1;
    }
}
