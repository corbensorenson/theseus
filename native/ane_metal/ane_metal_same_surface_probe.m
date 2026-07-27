/*
 * Sequential zero-copy visibility proof: Metal writes an IOSurface-backed
 * texture and ANE immediately consumes the same IOSurface generation.
 *
 * This uses undocumented AppleNeuralEngine classes and is research-only.
 * The private-runtime and MIL techniques are derived from maderix/ANE under
 * its MIT license; see THIRD_PARTY_NOTICES.md.
 */

#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <objc/message.h>

#include <dlfcn.h>
#include <math.h>
#include <mach/mach_time.h>

#define DIM 512
#define SEQ 64
#define INPUT_CHANNELS (2 * DIM)
#define LINEAR_INPUT_CHANNELS 512
#define ANE_LINEAR_OUTPUT_CHANNELS 384
#define METAL_LINEAR_OUTPUT_CHANNELS 384
#define LINEAR_OUTPUT_CHANNELS \
    (ANE_LINEAR_OUTPUT_CHANNELS + METAL_LINEAR_OUTPUT_CHANNELS)

static Class DescriptorClass;
static Class InMemoryModelClass;
static Class RequestClass;
static Class IOSurfaceObjectClass;

static double Milliseconds(uint64_t ticks) {
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    return (double)ticks * timebase.numer / timebase.denom / 1e6;
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
    const size_t bytesPerRow = width * sizeof(_Float16);
    NSDictionary *properties = @{
        (__bridge NSString *)kIOSurfaceWidth : @(width),
        (__bridge NSString *)kIOSurfaceHeight : @(height),
        (__bridge NSString *)kIOSurfaceBytesPerElement : @2,
        (__bridge NSString *)kIOSurfaceBytesPerRow : @(bytesPerRow),
        (__bridge NSString *)kIOSurfaceAllocSize : @(bytesPerRow * height),
        (__bridge NSString *)kIOSurfacePixelFormat :
            @(kCVPixelFormatType_OneComponent16Half),
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)properties);
}

static NSData *BuildWeightBlob(const float *weights, int count) {
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
        half[index] = (_Float16)weights[index];
    }
    return [NSData dataWithBytesNoCopy:bytes
                                length:(NSUInteger)totalBytes
                          freeWhenDone:YES];
}

static NSString *RMSNormBackwardMIL(void) {
    const float inverseDimension = 1.0f / (float)DIM;
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, "
         "{\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [mil appendFormat:
        @"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> inp) {\n",
        INPUT_CHANNELS, SEQ];
    [mil appendFormat:
        @"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), "
         "val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [mil appendString:
        @"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), "
         "val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dy = "
         "slice_by_size(x=inp,begin=b0,size=sz)[name=string(\"sdy\")];\n",
        DIM, SEQ];
    [mil appendFormat:
        @"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), "
         "val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> x = "
         "slice_by_size(x=inp,begin=b1,size=sz)[name=string(\"sx\")];\n",
        DIM, SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> sq = "
         "mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [mil appendString:
        @"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), "
         "val=tensor<int32, [1]>([1])];\n"
         "        bool kd = const()[name=string(\"kd\"),val=bool(true)];\n"];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> ss = "
         "reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n",
        SEQ];
    [mil appendFormat:
        @"        fp16 invd = const()[name=string(\"invd\"),val=fp16(%f)];\n",
        inverseDimension];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> ss2 = "
         "mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [mil appendString:
        @"        fp16 eps = const()[name=string(\"eps\"),"
         "val=fp16(0.00001)];\n"];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> ss3 = "
         "add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [mil appendString:
        @"        fp16 nhalf = const()[name=string(\"nhalf\"),"
         "val=fp16(-0.5)];\n"];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> rrms = "
         "pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,1]> w = const()[name=string(\"w\"), "
         "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE("
         "path=string(\"@model_path/weights/rms_w.bin\"),"
         "offset=uint64(64)))];\n", DIM, DIM];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dyw = "
         "mul(x=dy,y=w)[name=string(\"dyw\")];\n", DIM, SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dywx = "
         "mul(x=dyw,y=x)[name=string(\"dywx\")];\n", DIM, SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> dots = "
         "reduce_sum(x=dywx,axes=rax,keep_dims=kd)[name=string(\"ds\")];\n",
        SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> dsc = "
         "mul(x=dots,y=invd)[name=string(\"dsc\")];\n", SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> rr2 = "
         "mul(x=rrms,y=rrms)[name=string(\"rr2\")];\n", SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,1,1,%d]> coeff = "
         "mul(x=dsc,y=rr2)[name=string(\"cof\")];\n", SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> xc = "
         "mul(x=x,y=coeff)[name=string(\"xc\")];\n", DIM, SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> diff = "
         "sub(x=dyw,y=xc)[name=string(\"dif\")];\n", DIM, SEQ];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> out = "
         "mul(x=diff,y=rrms)[name=string(\"out\")];\n", DIM, SEQ];
    [mil appendString:@"    } -> (out);\n}\n"];
    return mil;
}

static id CompileANEModel(NSString *mil, NSDictionary *weights,
                          NSString **temporaryDirectory, NSError **error) {
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id descriptor =
        ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            DescriptorClass,
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, weights, nil);
    if (!descriptor) {
        return nil;
    }
    id model = ((id(*)(Class, SEL, id))objc_msgSend)(
        InMemoryModelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
    NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
        model, @selector(hexStringIdentifier));
    NSString *directory =
        [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
    NSString *weightDirectory =
        [directory stringByAppendingPathComponent:@"weights"];
    [[NSFileManager defaultManager]
        createDirectoryAtPath:weightDirectory
  withIntermediateDirectories:YES
                   attributes:nil
                        error:nil];
    [milData writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
              atomically:YES];
    for (NSString *path in weights) {
        NSString *relativePath =
            [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                             withString:@""];
        NSData *data = weights[path][@"data"];
        [data writeToFile:[directory stringByAppendingPathComponent:relativePath]
               atomically:YES];
    }
    BOOL compiled = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(compileWithQoS:options:error:), 21, @{}, error);
    if (!compiled) {
        return nil;
    }
    BOOL loaded = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        model, @selector(loadWithQoS:options:error:), 21, @{}, error);
    if (!loaded) {
        return nil;
    }
    *temporaryDirectory = directory;
    return model;
}

static BOOL EvaluateANE(id model, IOSurfaceRef input, IOSurfaceRef output,
                        NSError **error) {
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
        model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, request, error);
}

static BOOL MetalCopyToSurface(id<MTLDevice> device, IOSurfaceRef surface,
                               const _Float16 *values, size_t count,
                               size_t height,
                               double *milliseconds, NSError **error) {
    MTLTextureDescriptor *descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:SEQ
                                                          height:height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderWrite;
    id<MTLTexture> texture =
        [device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
    if (!texture) {
        return NO;
    }
    id<MTLBuffer> buffer =
        [device newBufferWithBytes:values
                           length:count * sizeof(_Float16)
                          options:MTLResourceStorageModeShared];
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void copy_half(device const half *input [[buffer(0)]],"
         " texture2d<half, access::write> output [[texture(0)]],"
         " uint2 gid [[thread_position_in_grid]]) {"
         "  if (gid.x < output.get_width() && gid.y < output.get_height()) {"
         "    uint index = gid.y * output.get_width() + gid.x;"
         "    output.write(half4(input[index]), gid);"
         "  }"
         "}\n";
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:nil error:error];
    id<MTLFunction> function = [library newFunctionWithName:@"copy_half"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:error];
    if (!pipeline) {
        return NO;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setTexture:texture atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(SEQ, height, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [encoder endEncoding];
    uint64_t started = mach_absolute_time();
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    *milliseconds = Milliseconds(mach_absolute_time() - started);
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        *error = commandBuffer.error;
        return NO;
    }
    return YES;
}

static NSString *LinearMIL(void) {
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, "
         "{\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [mil appendFormat:
        @"    func main<ios18>(tensor<fp16, [1,%d,1,%d]> x) {\n",
        LINEAR_INPUT_CHANNELS, SEQ];
    [mil appendString:
        @"        string pt = const()[name=string(\"pt\"),"
         "val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"),"
         "val=tensor<int32, [2]>([1,1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"),"
         "val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"),"
         "val=tensor<int32, [2]>([1,1])];\n"
         "        int32 gr = const()[name=string(\"gr\"),val=int32(1)];\n"];
    [mil appendFormat:
        @"        tensor<fp16, [%d,%d,1,1]> w = "
         "const()[name=string(\"w\"),"
         "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE("
         "path=string(\"@model_path/weights/linear.bin\"),"
         "offset=uint64(64)))];\n",
        ANE_LINEAR_OUTPUT_CHANNELS, LINEAR_INPUT_CHANNELS,
        ANE_LINEAR_OUTPUT_CHANNELS, LINEAR_INPUT_CHANNELS];
    [mil appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> out = "
         "conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,"
         "strides=st,weight=w,x=x)[name=string(\"linear\")];\n",
        ANE_LINEAR_OUTPUT_CHANNELS, SEQ];
    [mil appendString:@"    } -> (out);\n}\n"];
    return mil;
}

static id<MTLCommandBuffer> StartMetalLinear(
    id<MTLDevice> device, IOSurfaceRef inputSurface,
    IOSurfaceRef outputSurface, const _Float16 *weights,
    size_t outputChannels,
    NSError **error) {
    MTLTextureDescriptor *inputDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:SEQ
                                                          height:LINEAR_INPUT_CHANNELS
                                                       mipmapped:NO];
    inputDescriptor.storageMode = MTLStorageModeShared;
    inputDescriptor.usage = MTLTextureUsageShaderRead;
    MTLTextureDescriptor *outputDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:SEQ
                                                          height:outputChannels
                                                       mipmapped:NO];
    outputDescriptor.storageMode = MTLStorageModeShared;
    outputDescriptor.usage = MTLTextureUsageShaderWrite;
    id<MTLTexture> inputTexture =
        [device newTextureWithDescriptor:inputDescriptor
                               iosurface:inputSurface plane:0];
    id<MTLTexture> outputTexture =
        [device newTextureWithDescriptor:outputDescriptor
                               iosurface:outputSurface plane:0];
    const size_t weightCount =
        outputChannels * LINEAR_INPUT_CHANNELS;
    id<MTLBuffer> weightBuffer =
        [device newBufferWithBytes:weights
                           length:weightCount * sizeof(_Float16)
                          options:MTLResourceStorageModeShared];
    if (!inputTexture || !outputTexture || !weightBuffer) {
        return nil;
    }
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void split_linear("
         " texture2d<half, access::read> input [[texture(0)]],"
         " texture2d<half, access::write> output [[texture(1)]],"
         " device const half *weights [[buffer(0)]],"
         " uint2 gid [[thread_position_in_grid]]) {"
         "  if (gid.x < output.get_width() && gid.y < output.get_height()) {"
         "    float sum = 0.0f;"
         "    for (uint channel = 0; channel < 512; ++channel) {"
         "      sum += float(input.read(uint2(gid.x, channel)).x) *"
         "             float(weights[gid.y * 512 + channel]);"
         "    }"
         "    output.write(half4(half(sum)), gid);"
         "  }"
         "}\n";
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:nil error:error];
    id<MTLFunction> function = [library newFunctionWithName:@"split_linear"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:error];
    if (!pipeline) {
        return nil;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:inputTexture atIndex:0];
    [encoder setTexture:outputTexture atIndex:1];
    [encoder setBuffer:weightBuffer offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(SEQ, outputChannels, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    return commandBuffer;
}

static id<MTLCommandBuffer> StartMetalJoin(
    id<MTLDevice> device, IOSurfaceRef aneSurface,
    IOSurfaceRef metalSurface, IOSurfaceRef joinedSurface,
    NSError **error) {
    MTLTextureDescriptor *aneDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:SEQ
                                                          height:ANE_LINEAR_OUTPUT_CHANNELS
                                                       mipmapped:NO];
    aneDescriptor.storageMode = MTLStorageModeShared;
    aneDescriptor.usage = MTLTextureUsageShaderRead;
    MTLTextureDescriptor *metalDescriptor = [aneDescriptor copy];
    metalDescriptor.height = METAL_LINEAR_OUTPUT_CHANNELS;
    MTLTextureDescriptor *joinedDescriptor = [aneDescriptor copy];
    joinedDescriptor.height = LINEAR_OUTPUT_CHANNELS;
    joinedDescriptor.usage = MTLTextureUsageShaderWrite;
    id<MTLTexture> aneTexture =
        [device newTextureWithDescriptor:aneDescriptor
                               iosurface:aneSurface plane:0];
    id<MTLTexture> metalTexture =
        [device newTextureWithDescriptor:metalDescriptor
                               iosurface:metalSurface plane:0];
    id<MTLTexture> joinedTexture =
        [device newTextureWithDescriptor:joinedDescriptor
                               iosurface:joinedSurface plane:0];
    if (!aneTexture || !metalTexture || !joinedTexture) {
        return nil;
    }
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void join_partitions("
         " texture2d<half, access::read> ane [[texture(0)]],"
         " texture2d<half, access::read> metal [[texture(1)]],"
         " texture2d<half, access::write> joined [[texture(2)]],"
         " uint2 gid [[thread_position_in_grid]]) {"
         "  if (gid.x < joined.get_width() && gid.y < joined.get_height()) {"
         "    half value = gid.y < 384"
         "      ? ane.read(gid).x"
         "      : metal.read(uint2(gid.x, gid.y - 384)).x;"
         "    joined.write(half4(value), gid);"
         "  }"
         "}\n";
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:nil error:error];
    id<MTLFunction> function =
        [library newFunctionWithName:@"join_partitions"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:error];
    if (!pipeline) {
        return nil;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:aneTexture atIndex:0];
    [encoder setTexture:metalTexture atIndex:1];
    [encoder setTexture:joinedTexture atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(SEQ, LINEAR_OUTPUT_CHANNELS, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    return commandBuffer;
}

static id<MTLCommandBuffer> StartMetalReadWorkload(
    id<MTLDevice> device, IOSurfaceRef inputSurface,
    IOSurfaceRef outputSurface, uint32_t iterations, NSError **error) {
    MTLTextureDescriptor *inputDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:SEQ
                                                          height:INPUT_CHANNELS
                                                       mipmapped:NO];
    inputDescriptor.storageMode = MTLStorageModeShared;
    inputDescriptor.usage = MTLTextureUsageShaderRead;
    MTLTextureDescriptor *outputDescriptor = [inputDescriptor copy];
    outputDescriptor.usage = MTLTextureUsageShaderWrite;
    id<MTLTexture> inputTexture =
        [device newTextureWithDescriptor:inputDescriptor
                               iosurface:inputSurface
                                   plane:0];
    id<MTLTexture> outputTexture =
        [device newTextureWithDescriptor:outputDescriptor
                               iosurface:outputSurface
                                   plane:0];
    if (!inputTexture || !outputTexture) {
        return nil;
    }
    NSString *source =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void sustained_read("
         " texture2d<half, access::read> input [[texture(0)]],"
         " texture2d<half, access::write> output [[texture(1)]],"
         " constant uint &iterations [[buffer(0)]],"
         " uint2 gid [[thread_position_in_grid]]) {"
         "  if (gid.x < output.get_width() && gid.y < output.get_height()) {"
         "    half value = input.read(gid).x;"
         "    for (uint index = 0; index < iterations; ++index) {"
         "      value = fma(value, half(0.999), half(0.0001));"
         "    }"
         "    output.write(half4(value), gid);"
         "  }"
         "}\n";
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:nil error:error];
    id<MTLFunction> function =
        [library newFunctionWithName:@"sustained_read"];
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:error];
    if (!pipeline) {
        return nil;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:inputTexture atIndex:0];
    [encoder setTexture:outputTexture atIndex:1];
    [encoder setBytes:&iterations length:sizeof(iterations) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(SEQ, INPUT_CHANNELS, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    return commandBuffer;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        if (!LoadPrivateANE()) {
            fprintf(stderr, "private_ane_classes_missing\n");
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "metal_device_missing\n");
            return 3;
        }

        const size_t inputCount = (size_t)INPUT_CHANNELS * SEQ;
        const size_t outputCount = (size_t)DIM * SEQ;
        _Float16 *inputValues = malloc(inputCount * sizeof(_Float16));
        float *weights = malloc(DIM * sizeof(float));
        for (size_t index = 0; index < inputCount; ++index) {
            const int signedValue = (int)(index % 257) - 128;
            inputValues[index] = (_Float16)((float)signedValue / 1024.0f);
        }
        for (int index = 0; index < DIM; ++index) {
            weights[index] = 0.75f + (float)(index % 17) / 64.0f;
        }

        NSError *error = nil;
        NSString *temporaryDirectory = nil;
        uint64_t compileStarted = mach_absolute_time();
        id model = CompileANEModel(
            RMSNormBackwardMIL(),
            @{@"@model_path/weights/rms_w.bin" :
                  @{@"offset" : @0,
                    @"data" : BuildWeightBlob(weights, DIM)}},
            &temporaryDirectory, &error);
        double compileMilliseconds =
            Milliseconds(mach_absolute_time() - compileStarted);
        if (!model) {
            fprintf(stderr, "ane_compile_failed: %s\n",
                    error.description.UTF8String);
            return 4;
        }

        IOSurfaceRef inputSurface =
            CreateHalfSurface(SEQ, INPUT_CHANNELS);
        IOSurfaceRef outputSurface = CreateHalfSurface(SEQ, DIM);
        if (!inputSurface || !outputSurface) {
            fprintf(stderr, "iosurface_create_failed\n");
            return 5;
        }

        double metalMilliseconds = 0.0;
        if (!MetalCopyToSurface(device, inputSurface, inputValues, inputCount,
                                INPUT_CHANNELS,
                                &metalMilliseconds, &error)) {
            fprintf(stderr, "metal_copy_failed: %s\n",
                    error.description.UTF8String);
            return 6;
        }
        uint64_t aneStarted = mach_absolute_time();
        if (!EvaluateANE(model, inputSurface, outputSurface, &error)) {
            fprintf(stderr, "ane_eval_failed: %s\n",
                    error.description.UTF8String);
            return 7;
        }
        double metalToANEMilliseconds =
            Milliseconds(mach_absolute_time() - aneStarted);
        _Float16 *metalOutput = malloc(outputCount * sizeof(_Float16));
        IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        memcpy(metalOutput, IOSurfaceGetBaseAddress(outputSurface),
               outputCount * sizeof(_Float16));
        IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);

        IOSurfaceLock(inputSurface, 0, NULL);
        memcpy(IOSurfaceGetBaseAddress(inputSurface), inputValues,
               inputCount * sizeof(_Float16));
        IOSurfaceUnlock(inputSurface, 0, NULL);
        uint64_t hostControlStarted = mach_absolute_time();
        if (!EvaluateANE(model, inputSurface, outputSurface, &error)) {
            fprintf(stderr, "ane_control_eval_failed: %s\n",
                    error.description.UTF8String);
            return 8;
        }
        double hostToANEMilliseconds =
            Milliseconds(mach_absolute_time() - hostControlStarted);

        size_t bitMismatchCount = 0;
        float maximumAbsoluteDelta = 0.0f;
        _Float16 *hostControlOutput =
            malloc(outputCount * sizeof(_Float16));
        IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *hostOutput = IOSurfaceGetBaseAddress(outputSurface);
        memcpy(hostControlOutput, hostOutput,
               outputCount * sizeof(_Float16));
        for (size_t index = 0; index < outputCount; ++index) {
            if (memcmp(&metalOutput[index], &hostOutput[index],
                       sizeof(_Float16)) != 0) {
                ++bitMismatchCount;
            }
            maximumAbsoluteDelta =
                fmaxf(maximumAbsoluteDelta,
                      fabsf((float)metalOutput[index] -
                            (float)hostOutput[index]));
        }
        IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);

        const uint32_t metalIterations = 2048;
        const int aneRepeats = 64;
        IOSurfaceRef metalReadOutput =
            CreateHalfSurface(SEQ, INPUT_CHANNELS);
        id<MTLCommandBuffer> warmupCommand = StartMetalReadWorkload(
            device, inputSurface, metalReadOutput, metalIterations, &error);
        if (!warmupCommand ||
            !EvaluateANE(model, inputSurface, outputSurface, &error)) {
            fprintf(stderr, "concurrent_read_warmup_failed: %s\n",
                    error.description.UTF8String);
            return 9;
        }
        [warmupCommand waitUntilCompleted];

        uint64_t metalStandaloneStarted = mach_absolute_time();
        id<MTLCommandBuffer> metalStandaloneCommand = StartMetalReadWorkload(
            device, inputSurface, metalReadOutput, metalIterations, &error);
        [metalStandaloneCommand waitUntilCompleted];
        double metalStandaloneMilliseconds =
            Milliseconds(mach_absolute_time() - metalStandaloneStarted);

        uint64_t aneStandaloneStarted = mach_absolute_time();
        for (int repeat = 0; repeat < aneRepeats; ++repeat) {
            if (!EvaluateANE(model, inputSurface, outputSurface, &error)) {
                fprintf(stderr, "ane_standalone_repeat_failed: %s\n",
                        error.description.UTF8String);
                return 10;
            }
        }
        double aneStandaloneMilliseconds =
            Milliseconds(mach_absolute_time() - aneStandaloneStarted);

        uint64_t concurrentStarted = mach_absolute_time();
        id<MTLCommandBuffer> concurrentCommand = StartMetalReadWorkload(
            device, inputSurface, metalReadOutput, metalIterations, &error);
        for (int repeat = 0; repeat < aneRepeats; ++repeat) {
            if (!EvaluateANE(model, inputSurface, outputSurface, &error)) {
                fprintf(stderr, "ane_concurrent_repeat_failed: %s\n",
                        error.description.UTF8String);
                return 11;
            }
        }
        [concurrentCommand waitUntilCompleted];
        double concurrentMilliseconds =
            Milliseconds(mach_absolute_time() - concurrentStarted);
        const double concurrentReadSpeedup =
            (metalStandaloneMilliseconds + aneStandaloneMilliseconds) /
            concurrentMilliseconds;
        const BOOL concurrentReadOverlapObserved =
            concurrentMilliseconds <
            metalStandaloneMilliseconds + aneStandaloneMilliseconds;

        size_t concurrentANEMismatchCount = 0;
        IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *concurrentANEOutput =
            IOSurfaceGetBaseAddress(outputSurface);
        for (size_t index = 0; index < outputCount; ++index) {
            if (memcmp(&hostControlOutput[index], &concurrentANEOutput[index],
                       sizeof(_Float16)) != 0) {
                ++concurrentANEMismatchCount;
            }
        }
        IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        size_t nonfiniteMetalOutputCount = 0;
        IOSurfaceLock(metalReadOutput, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *metalReadValues =
            IOSurfaceGetBaseAddress(metalReadOutput);
        for (size_t index = 0; index < inputCount; ++index) {
            if (!isfinite((float)metalReadValues[index])) {
                ++nonfiniteMetalOutputCount;
            }
        }
        IOSurfaceUnlock(metalReadOutput, kIOSurfaceLockReadOnly, NULL);

        const size_t linearInputCount =
            (size_t)LINEAR_INPUT_CHANNELS * SEQ;
        const size_t aneLinearWeightCount =
            (size_t)ANE_LINEAR_OUTPUT_CHANNELS * LINEAR_INPUT_CHANNELS;
        const size_t metalLinearWeightCount =
            (size_t)METAL_LINEAR_OUTPUT_CHANNELS * LINEAR_INPUT_CHANNELS;
        _Float16 *linearInput =
            malloc(linearInputCount * sizeof(_Float16));
        float *aneLinearWeights =
            malloc(aneLinearWeightCount * sizeof(float));
        _Float16 *metalLinearWeights =
            malloc(metalLinearWeightCount * sizeof(_Float16));
        const size_t fullLinearWeightCount =
            (size_t)LINEAR_OUTPUT_CHANNELS * LINEAR_INPUT_CHANNELS;
        _Float16 *fullLinearWeights =
            malloc(fullLinearWeightCount * sizeof(_Float16));
        for (size_t index = 0; index < linearInputCount; ++index) {
            linearInput[index] =
                (_Float16)(((int)(index % 113) - 56) / 512.0f);
        }
        for (size_t index = 0; index < aneLinearWeightCount; ++index) {
            aneLinearWeights[index] =
                ((int)(index % 97) - 48) / 4096.0f;
        }
        for (size_t index = 0; index < metalLinearWeightCount; ++index) {
            metalLinearWeights[index] =
                (_Float16)(((int)(index % 89) - 44) / 4096.0f);
        }
        for (size_t index = 0; index < aneLinearWeightCount; ++index) {
            fullLinearWeights[index] = (_Float16)aneLinearWeights[index];
        }
        memcpy(fullLinearWeights + aneLinearWeightCount, metalLinearWeights,
               metalLinearWeightCount * sizeof(_Float16));
        NSString *linearTemporaryDirectory = nil;
        uint64_t linearCompileStarted = mach_absolute_time();
        id linearModel = CompileANEModel(
            LinearMIL(),
            @{@"@model_path/weights/linear.bin" :
                  @{@"offset" : @0,
                    @"data" : BuildWeightBlob(
                        aneLinearWeights, (int)aneLinearWeightCount)}},
            &linearTemporaryDirectory, &error);
        double linearCompileMilliseconds =
            Milliseconds(mach_absolute_time() - linearCompileStarted);
        IOSurfaceRef linearInputSurface =
            CreateHalfSurface(SEQ, LINEAR_INPUT_CHANNELS);
        IOSurfaceRef aneLinearOutput =
            CreateHalfSurface(SEQ, ANE_LINEAR_OUTPUT_CHANNELS);
        IOSurfaceRef metalLinearOutput =
            CreateHalfSurface(SEQ, METAL_LINEAR_OUTPUT_CHANNELS);
        IOSurfaceRef joinedLinearOutput =
            CreateHalfSurface(SEQ, LINEAR_OUTPUT_CHANNELS);
        IOSurfaceRef fullMetalLinearOutput =
            CreateHalfSurface(SEQ, LINEAR_OUTPUT_CHANNELS);
        double linearFillMilliseconds = 0.0;
        BOOL linearSetupGreen =
            linearModel && linearInputSurface && aneLinearOutput &&
            metalLinearOutput && joinedLinearOutput && fullMetalLinearOutput &&
            MetalCopyToSurface(device, linearInputSurface, linearInput,
                               linearInputCount, LINEAR_INPUT_CHANNELS,
                               &linearFillMilliseconds, &error);
        if (!linearSetupGreen) {
            fprintf(stderr, "split_linear_setup_failed: %s\n",
                    error.description.UTF8String);
            return 12;
        }

        id<MTLCommandBuffer> linearWarmupCommand = StartMetalLinear(
            device, linearInputSurface, metalLinearOutput,
            metalLinearWeights, METAL_LINEAR_OUTPUT_CHANNELS, &error);
        [linearWarmupCommand waitUntilCompleted];
        id<MTLCommandBuffer> fullMetalWarmupCommand = StartMetalLinear(
            device, linearInputSurface, fullMetalLinearOutput,
            fullLinearWeights, LINEAR_OUTPUT_CHANNELS, &error);
        [fullMetalWarmupCommand waitUntilCompleted];
        EvaluateANE(linearModel, linearInputSurface, aneLinearOutput, &error);
        id<MTLCommandBuffer> joinWarmupCommand = StartMetalJoin(
            device, aneLinearOutput, metalLinearOutput,
            joinedLinearOutput, &error);
        [joinWarmupCommand waitUntilCompleted];

        uint64_t fullMetalControlStarted = mach_absolute_time();
        id<MTLCommandBuffer> fullMetalControlCommand = StartMetalLinear(
            device, linearInputSurface, fullMetalLinearOutput,
            fullLinearWeights, LINEAR_OUTPUT_CHANNELS, &error);
        [fullMetalControlCommand waitUntilCompleted];
        double fullMetalControlMilliseconds =
            Milliseconds(mach_absolute_time() - fullMetalControlStarted);

        uint64_t aneLinearStandaloneStarted = mach_absolute_time();
        BOOL aneLinearGreen = EvaluateANE(
            linearModel, linearInputSurface, aneLinearOutput, &error);
        double aneLinearStandaloneMilliseconds =
            Milliseconds(mach_absolute_time() - aneLinearStandaloneStarted);
        const size_t aneLinearOutputCount =
            (size_t)ANE_LINEAR_OUTPUT_CHANNELS * SEQ;
        const size_t metalLinearOutputCount =
            (size_t)METAL_LINEAR_OUTPUT_CHANNELS * SEQ;
        _Float16 *aneLinearControl =
            malloc(aneLinearOutputCount * sizeof(_Float16));
        _Float16 *metalLinearControl =
            malloc(metalLinearOutputCount * sizeof(_Float16));
        IOSurfaceLock(aneLinearOutput, kIOSurfaceLockReadOnly, NULL);
        memcpy(aneLinearControl, IOSurfaceGetBaseAddress(aneLinearOutput),
               aneLinearOutputCount * sizeof(_Float16));
        IOSurfaceUnlock(aneLinearOutput, kIOSurfaceLockReadOnly, NULL);

        uint64_t metalLinearStandaloneStarted = mach_absolute_time();
        id<MTLCommandBuffer> metalLinearStandaloneCommand = StartMetalLinear(
            device, linearInputSurface, metalLinearOutput,
            metalLinearWeights, METAL_LINEAR_OUTPUT_CHANNELS, &error);
        [metalLinearStandaloneCommand waitUntilCompleted];
        double metalLinearStandaloneMilliseconds =
            Milliseconds(mach_absolute_time() - metalLinearStandaloneStarted);
        IOSurfaceLock(metalLinearOutput, kIOSurfaceLockReadOnly, NULL);
        memcpy(metalLinearControl, IOSurfaceGetBaseAddress(metalLinearOutput),
               metalLinearOutputCount * sizeof(_Float16));
        IOSurfaceUnlock(metalLinearOutput, kIOSurfaceLockReadOnly, NULL);

        uint64_t splitConcurrentStarted = mach_absolute_time();
        id<MTLCommandBuffer> splitConcurrentCommand = StartMetalLinear(
            device, linearInputSurface, metalLinearOutput,
            metalLinearWeights, METAL_LINEAR_OUTPUT_CHANNELS, &error);
        BOOL concurrentANEGreen = EvaluateANE(
            linearModel, linearInputSurface, aneLinearOutput, &error);
        [splitConcurrentCommand waitUntilCompleted];
        double splitConcurrentMilliseconds =
            Milliseconds(mach_absolute_time() - splitConcurrentStarted);
        uint64_t joinStarted = mach_absolute_time();
        id<MTLCommandBuffer> joinCommand = StartMetalJoin(
            device, aneLinearOutput, metalLinearOutput,
            joinedLinearOutput, &error);
        [joinCommand waitUntilCompleted];
        double joinMilliseconds =
            Milliseconds(mach_absolute_time() - joinStarted);

        size_t anePartitionMismatchCount = 0;
        size_t metalPartitionMismatchCount = 0;
        size_t joinedPartitionMismatchCount = 0;
        float joinedVersusFullMetalMaximumAbsoluteDelta = 0.0f;
        double joinedVersusFullMetalMeanAbsoluteDelta = 0.0;
        IOSurfaceLock(aneLinearOutput, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *aneConcurrent =
            IOSurfaceGetBaseAddress(aneLinearOutput);
        for (size_t index = 0; index < aneLinearOutputCount; ++index) {
            if (memcmp(&aneConcurrent[index], &aneLinearControl[index],
                       sizeof(_Float16)) != 0) {
                ++anePartitionMismatchCount;
            }
        }
        IOSurfaceUnlock(aneLinearOutput, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(metalLinearOutput, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *metalConcurrent =
            IOSurfaceGetBaseAddress(metalLinearOutput);
        for (size_t index = 0; index < metalLinearOutputCount; ++index) {
            if (memcmp(&metalConcurrent[index], &metalLinearControl[index],
                       sizeof(_Float16)) != 0) {
                ++metalPartitionMismatchCount;
            }
        }
        IOSurfaceUnlock(metalLinearOutput, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(joinedLinearOutput, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *joined = IOSurfaceGetBaseAddress(joinedLinearOutput);
        for (size_t channel = 0; channel < LINEAR_OUTPUT_CHANNELS; ++channel) {
            const _Float16 *expected =
                channel < ANE_LINEAR_OUTPUT_CHANNELS
                    ? aneLinearControl +
                          channel * SEQ
                    : metalLinearControl +
                          (channel - ANE_LINEAR_OUTPUT_CHANNELS) * SEQ;
            for (size_t position = 0; position < SEQ; ++position) {
                const size_t joinedIndex = channel * SEQ + position;
                if (memcmp(&joined[joinedIndex], &expected[position],
                           sizeof(_Float16)) != 0) {
                    ++joinedPartitionMismatchCount;
                }
            }
        }
        IOSurfaceUnlock(joinedLinearOutput, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(joinedLinearOutput, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceLock(fullMetalLinearOutput, kIOSurfaceLockReadOnly, NULL);
        joined = IOSurfaceGetBaseAddress(joinedLinearOutput);
        const _Float16 *fullMetal =
            IOSurfaceGetBaseAddress(fullMetalLinearOutput);
        const size_t joinedOutputCount =
            (size_t)LINEAR_OUTPUT_CHANNELS * SEQ;
        for (size_t index = 0; index < joinedOutputCount; ++index) {
            const float delta =
                fabsf((float)joined[index] - (float)fullMetal[index]);
            joinedVersusFullMetalMaximumAbsoluteDelta =
                fmaxf(joinedVersusFullMetalMaximumAbsoluteDelta, delta);
            joinedVersusFullMetalMeanAbsoluteDelta += delta;
        }
        joinedVersusFullMetalMeanAbsoluteDelta /= joinedOutputCount;
        IOSurfaceUnlock(fullMetalLinearOutput, kIOSurfaceLockReadOnly, NULL);
        IOSurfaceUnlock(joinedLinearOutput, kIOSurfaceLockReadOnly, NULL);
        const double splitSerialSumMilliseconds =
            aneLinearStandaloneMilliseconds +
            metalLinearStandaloneMilliseconds;
        const double splitSpeedup =
            splitSerialSumMilliseconds / splitConcurrentMilliseconds;
        const double splitJoinedMilliseconds =
            splitConcurrentMilliseconds + joinMilliseconds;
        const double splitJoinedSpeedupVsFullMetal =
            fullMetalControlMilliseconds / splitJoinedMilliseconds;
        const BOOL splitLinearMechanicsGreen =
            aneLinearGreen && concurrentANEGreen &&
            metalLinearStandaloneCommand.status ==
                MTLCommandBufferStatusCompleted &&
            splitConcurrentCommand.status ==
                MTLCommandBufferStatusCompleted &&
            joinCommand.status == MTLCommandBufferStatusCompleted &&
            anePartitionMismatchCount == 0 &&
            metalPartitionMismatchCount == 0 &&
            joinedPartitionMismatchCount == 0;

        printf("{\"state\":\"%s\",\"device\":\"%s\","
               "\"input_elements\":%zu,\"output_elements\":%zu,"
               "\"compile_milliseconds\":%.3f,"
               "\"metal_write_milliseconds\":%.3f,"
               "\"ane_after_metal_milliseconds\":%.3f,"
               "\"ane_after_host_control_milliseconds\":%.3f,"
               "\"bit_mismatch_count\":%zu,"
               "\"maximum_absolute_delta\":%.9g,"
               "\"concurrent_read\":{\"metal_iterations\":%u,"
               "\"ane_repeats\":%d,"
               "\"metal_standalone_milliseconds\":%.3f,"
               "\"ane_standalone_milliseconds\":%.3f,"
               "\"concurrent_milliseconds\":%.3f,"
               "\"speedup_vs_serial_sum\":%.6f,"
               "\"overlap_observed\":%s,"
               "\"ane_bit_mismatch_count\":%zu,"
               "\"metal_nonfinite_output_count\":%zu},"
               "\"split_linear\":{\"state\":\"%s\","
               "\"input_channels\":%d,\"ane_output_channels\":%d,"
               "\"metal_output_channels\":%d,\"sequence\":%d,"
               "\"compile_milliseconds\":%.3f,"
               "\"input_fill_milliseconds\":%.3f,"
               "\"ane_standalone_milliseconds\":%.3f,"
               "\"metal_standalone_milliseconds\":%.3f,"
               "\"serial_sum_milliseconds\":%.3f,"
               "\"concurrent_milliseconds\":%.3f,"
               "\"speedup_vs_serial_sum\":%.6f,"
               "\"join_milliseconds\":%.3f,"
               "\"joined_milliseconds\":%.3f,"
               "\"full_metal_control_milliseconds\":%.3f,"
               "\"joined_speedup_vs_full_metal_control\":%.6f,"
               "\"joined_vs_full_metal_maximum_absolute_delta\":%.9g,"
               "\"joined_vs_full_metal_mean_absolute_delta\":%.9g,"
               "\"ane_partition_mismatch_count\":%zu,"
               "\"metal_partition_mismatch_count\":%zu,"
               "\"joined_partition_mismatch_count\":%zu},"
               "\"intermediate_host_round_trip\":false,"
               "\"host_verification_read\":true}\n",
               bitMismatchCount == 0 && concurrentANEMismatchCount == 0 &&
                       nonfiniteMetalOutputCount == 0 &&
                       splitLinearMechanicsGreen
                   ? "GREEN"
                   : "RED",
               device.name.UTF8String, inputCount, outputCount,
               compileMilliseconds, metalMilliseconds,
               metalToANEMilliseconds, hostToANEMilliseconds,
               bitMismatchCount, maximumAbsoluteDelta,
               metalIterations, aneRepeats,
               metalStandaloneMilliseconds, aneStandaloneMilliseconds,
               concurrentMilliseconds, concurrentReadSpeedup,
               concurrentReadOverlapObserved ? "true" : "false",
               concurrentANEMismatchCount, nonfiniteMetalOutputCount,
               splitLinearMechanicsGreen ? "GREEN" : "RED",
               LINEAR_INPUT_CHANNELS, ANE_LINEAR_OUTPUT_CHANNELS,
               METAL_LINEAR_OUTPUT_CHANNELS, SEQ,
               linearCompileMilliseconds, linearFillMilliseconds,
               aneLinearStandaloneMilliseconds,
               metalLinearStandaloneMilliseconds,
               splitSerialSumMilliseconds, splitConcurrentMilliseconds,
               splitSpeedup, joinMilliseconds, splitJoinedMilliseconds,
               fullMetalControlMilliseconds,
               splitJoinedSpeedupVsFullMetal,
               joinedVersusFullMetalMaximumAbsoluteDelta,
               joinedVersusFullMetalMeanAbsoluteDelta,
               anePartitionMismatchCount, metalPartitionMismatchCount,
               joinedPartitionMismatchCount);

        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            linearModel, @selector(unloadWithQoS:error:), 21, &error);
        [[NSFileManager defaultManager] removeItemAtPath:temporaryDirectory
                                                   error:nil];
        [[NSFileManager defaultManager]
            removeItemAtPath:linearTemporaryDirectory error:nil];
        CFRelease(inputSurface);
        CFRelease(outputSurface);
        CFRelease(metalReadOutput);
        CFRelease(linearInputSurface);
        CFRelease(aneLinearOutput);
        CFRelease(metalLinearOutput);
        CFRelease(joinedLinearOutput);
        CFRelease(fullMetalLinearOutput);
        free(inputValues);
        free(weights);
        free(metalOutput);
        free(hostControlOutput);
        free(linearInput);
        free(aneLinearWeights);
        free(metalLinearWeights);
        free(fullLinearWeights);
        free(aneLinearControl);
        free(metalLinearControl);
        return bitMismatchCount == 0 && concurrentANEMismatchCount == 0 &&
                       nonfiniteMetalOutputCount == 0 &&
                       splitLinearMechanicsGreen
                   ? 0
                   : 13;
    }
}
