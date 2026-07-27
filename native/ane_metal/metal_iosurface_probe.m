#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <math.h>

static IOSurfaceRef CreateFloatSurface(size_t width) {
    const size_t bytesPerElement = sizeof(float);
    const size_t bytesPerRow = width * bytesPerElement;
    NSDictionary *properties = @{
        (__bridge NSString *)kIOSurfaceWidth : @(width),
        (__bridge NSString *)kIOSurfaceHeight : @1,
        (__bridge NSString *)kIOSurfaceBytesPerElement : @(bytesPerElement),
        (__bridge NSString *)kIOSurfaceBytesPerRow : @(bytesPerRow),
        (__bridge NSString *)kIOSurfaceAllocSize : @(bytesPerRow),
        (__bridge NSString *)kIOSurfacePixelFormat :
            @(kCVPixelFormatType_OneComponent32Float),
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)properties);
}

int main(void) {
    @autoreleasepool {
        const size_t width = 4096;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "no_metal_device\n");
            return 2;
        }
        IOSurfaceRef inputSurface = CreateFloatSurface(width);
        IOSurfaceRef outputSurface = CreateFloatSurface(width);
        if (inputSurface == NULL || outputSurface == NULL) {
            fprintf(stderr, "iosurface_create_failed\n");
            return 3;
        }

        IOSurfaceLock(inputSurface, 0, NULL);
        float *input = IOSurfaceGetBaseAddress(inputSurface);
        for (size_t index = 0; index < width; ++index) {
            input[index] = (float)index * 0.25f;
        }
        IOSurfaceUnlock(inputSurface, 0, NULL);

        MTLTextureDescriptor *descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                               width:width
                                                              height:1
                                                           mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> inputTexture =
            [device newTextureWithDescriptor:descriptor iosurface:inputSurface plane:0];
        id<MTLTexture> outputTexture =
            [device newTextureWithDescriptor:descriptor iosurface:outputSurface plane:0];
        if (inputTexture == nil || outputTexture == nil) {
            fprintf(stderr, "metal_iosurface_import_failed\n");
            return 4;
        }

        NSString *source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void add_one(texture2d<float, access::read> input [[texture(0)]],"
             "                    texture2d<float, access::write> output [[texture(1)]],"
             "                    uint2 gid [[thread_position_in_grid]]) {"
             "  if (gid.x < output.get_width()) {"
             "    float4 value = input.read(gid);"
             "    output.write(float4(value.x + 1.0f), gid);"
             "  }"
             "}\n";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"add_one"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            fprintf(stderr, "metal_compile_failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 5;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:inputTexture atIndex:0];
        [encoder setTexture:outputTexture atIndex:1];
        NSUInteger threads = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(width, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "metal_command_failed: %s\n",
                    commandBuffer.error.localizedDescription.UTF8String);
            return 6;
        }

        IOSurfaceLock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        const float *output = IOSurfaceGetBaseAddress(outputSurface);
        float maximumDelta = 0.0f;
        for (size_t index = 0; index < width; ++index) {
            const float expected = (float)index * 0.25f + 1.0f;
            maximumDelta = fmaxf(maximumDelta,
                                 fabsf(output[index] - expected));
        }
        IOSurfaceUnlock(outputSurface, kIOSurfaceLockReadOnly, NULL);
        printf("{\"state\":\"%s\",\"device\":\"%s\",\"elements\":%zu,"
               "\"maximum_absolute_delta\":%.9g,"
               "\"intermediate_host_round_trip\":false,"
               "\"host_verification_read\":true}\n",
               maximumDelta == 0.0f ? "GREEN" : "RED",
               device.name.UTF8String, width, maximumDelta);
        CFRelease(inputSurface);
        CFRelease(outputSurface);
        return maximumDelta == 0.0f ? 0 : 7;
    }
}
