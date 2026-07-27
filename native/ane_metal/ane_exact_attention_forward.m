/*
 * Exact Project Theseus self-attention forward qualification.
 *
 * This is the first native slice of the frozen decoder-block ABI:
 *   hidden -> RMSNorm -> dynamic Q/K/V -> split-half RoPE
 *          -> contiguous 4:1 GQA -> causal SDPA.
 *
 * It emits normalized input plus Q/K/V taps for the later backward kernels.
 * It does not implement out_proj, the residual, SwiGLU, backward, or update,
 * and therefore cannot support a decoder-block or training-speed claim.
 *
 * Private-runtime and dynamic-weight MIL techniques are derived from
 * maderix/ANE (MIT) at the source identity in THIRD_PARTY_NOTICES.md.
 */

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>

#include <dlfcn.h>
#include <mach/mach_time.h>
#include <math.h>

#define SEQUENCE 128
#define DIM 512
#define QUERY_HEADS 8
#define KV_HEADS 2
#define HEAD_DIM 64
#define QUERY_GROUPS (QUERY_HEADS / KV_HEADS)
#define QUERY_DIM (QUERY_HEADS * HEAD_DIM)
#define KV_DIM (KV_HEADS * HEAD_DIM)
#define HALF_HEAD_DIM (HEAD_DIM / 2)
#ifndef NORM_SCALE_SPAN
#define NORM_SCALE_SPAN 128
#endif
#define PACKED_SPATIAL \
    (SEQUENCE + NORM_SCALE_SPAN + QUERY_DIM + KV_DIM + KV_DIM)
#ifndef QUALIFICATION_STAGE
#define QUALIFICATION_STAGE 4
#endif
#if QUALIFICATION_STAGE == 0 || QUALIFICATION_STAGE == 1
#define OUTPUT_CHANNELS DIM
#elif QUALIFICATION_STAGE == 2 || QUALIFICATION_STAGE == 3
#define OUTPUT_CHANNELS (QUERY_DIM + KV_DIM + KV_DIM + DIM)
#else
#define OUTPUT_CHANNELS (QUERY_DIM + QUERY_DIM + KV_DIM + KV_DIM + DIM)
#endif
#define INPUT_ELEMENTS ((size_t)DIM * PACKED_SPATIAL)
#define OUTPUT_ELEMENTS ((size_t)OUTPUT_CHANNELS * SEQUENCE)

static Class DescriptorClass;
static Class InMemoryModelClass;
static Class RequestClass;
static Class IOSurfaceObjectClass;
static mach_timebase_info_data_t Timebase;

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

static void AppendDynamicProjection(
    NSMutableString *mil, NSString *name, int outputChannels, int weightOffset
) {
    [mil appendFormat:
        @"        tensor<int32,[4]> %@_bw = const()[name=string(\"%@_bw\"),"
         "val=tensor<int32,[4]>([0,0,0,%d])];\n",
        name, name, weightOffset];
    [mil appendFormat:
        @"        tensor<int32,[4]> %@_sw = const()[name=string(\"%@_sw\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        name, name, DIM, outputChannels];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> %@_wt = "
         "slice_by_size(x=packed,begin=%@_bw,size=%@_sw)"
         "[name=string(\"%@_wt\")];\n",
        DIM, outputChannels, name, name, name, name];
    [mil appendFormat:
        @"        tensor<int32,[4]> %@_rw = const()[name=string(\"%@_rw\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n",
        name, name, DIM, outputChannels];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> %@_w = "
         "reshape(shape=%@_rw,x=%@_wt)[name=string(\"%@_w\")];\n",
        DIM, outputChannels, name, name, name, name];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> %@_rows = "
         "matmul(transpose_x=bf,transpose_y=bf,x=xn_rows,y=%@_w)"
         "[name=string(\"%@_rows\")];\n",
        SEQUENCE, outputChannels, name, name, name];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> %@_channels2 = "
         "transpose(perm=transpose_last,x=%@_rows)"
         "[name=string(\"%@_channels2\")];\n",
        outputChannels, SEQUENCE, name, name, name];
    [mil appendFormat:
        @"        tensor<int32,[4]> %@_channels_shape = const()["
         "name=string(\"%@_channels_shape\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        name, name, outputChannels, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> %@_channels = "
         "reshape(shape=%@_channels_shape,x=%@_channels2)"
         "[name=string(\"%@_channels\")];\n",
        outputChannels, SEQUENCE, name, name, name, name];
}

static void AppendSplitHalfRoPE(
    NSMutableString *mil, NSString *prefix, NSString *input, int heads
) {
    [mil appendFormat:
        @"        tensor<int32,[4]> %@_b0 = const()[name=string(\"%@_b0\"),"
         "val=tensor<int32,[4]>([0,0,0,0])];\n",
        prefix, prefix];
    [mil appendFormat:
        @"        tensor<int32,[4]> %@_b1 = const()[name=string(\"%@_b1\"),"
         "val=tensor<int32,[4]>([0,0,0,%d])];\n",
        prefix, prefix, HALF_HEAD_DIM];
    [mil appendFormat:
        @"        tensor<int32,[4]> %@_half_shape = const()["
         "name=string(\"%@_half_shape\"),"
         "val=tensor<int32,[4]>([1,%d,%d,%d])];\n",
        prefix, prefix, heads, SEQUENCE, HALF_HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> %@_first = "
         "slice_by_size(x=%@,begin=%@_b0,size=%@_half_shape)"
         "[name=string(\"%@_first\")];\n",
        heads, SEQUENCE, HALF_HEAD_DIM, prefix, input, prefix, prefix, prefix];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> %@_second = "
         "slice_by_size(x=%@,begin=%@_b1,size=%@_half_shape)"
         "[name=string(\"%@_second\")];\n",
        heads, SEQUENCE, HALF_HEAD_DIM, prefix, input, prefix, prefix, prefix];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> %@_negative_second = "
         "mul(x=%@_second,y=negative_one)"
         "[name=string(\"%@_negative_second\")];\n",
        heads, SEQUENCE, HALF_HEAD_DIM, prefix, prefix, prefix];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> %@_rotated = "
         "concat(axis=last_axis,interleave=no_interleave,"
         "values=(%@_negative_second,%@_first))"
         "[name=string(\"%@_rotated\")];\n",
        heads, SEQUENCE, HEAD_DIM, prefix, prefix, prefix, prefix];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> %@_cos = "
         "mul(x=%@,y=rope_cos)[name=string(\"%@_cos\")];\n",
        heads, SEQUENCE, HEAD_DIM, prefix, input, prefix];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> %@_sin = "
         "mul(x=%@_rotated,y=rope_sin)[name=string(\"%@_sin\")];\n",
        heads, SEQUENCE, HEAD_DIM, prefix, prefix, prefix];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> %@_rope = "
         "add(x=%@_cos,y=%@_sin)[name=string(\"%@_rope\")];\n",
        heads, SEQUENCE, HEAD_DIM, prefix, prefix, prefix, prefix];
}

static NSString *ExactAttentionMIL(void) {
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, "
         "{\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [mil appendFormat:
        @"    func main<ios18>(tensor<fp16,[1,%d,1,%d]> packed) {\n",
        DIM, PACKED_SPATIAL];
    [mil appendString:
        @"        bool bf = const()[name=string(\"bf\"),val=bool(false)];\n"
         "        bool bt = const()[name=string(\"bt\"),val=bool(true)];\n"
         "        bool keep = const()[name=string(\"keep\"),val=bool(true)];\n"
         "        bool no_interleave = const()[name=string(\"no_interleave\"),"
         "val=bool(false)];\n"
         "        int32 head_axis = const()[name=string(\"head_axis\"),"
         "val=int32(1)];\n"
         "        int32 last_axis = const()[name=string(\"last_axis\"),"
         "val=int32(3)];\n"
         "        int32 softmax_axis = const()[name=string(\"softmax_axis\"),"
         "val=int32(-1)];\n"
         "        fp16 negative_one = const()[name=string(\"negative_one\"),"
         "val=fp16(-1)];\n"
         "        tensor<int32,[4]> transpose_last = const()["
         "name=string(\"transpose_last\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"
         "        tensor<int32,[1]> channel_axis = const()["
         "name=string(\"channel_axis\"),val=tensor<int32,[1]>([1])];\n"
         "        tensor<int32,[4]> hidden_begin = const()["
         "name=string(\"hidden_begin\"),"
         "val=tensor<int32,[4]>([0,0,0,0])];\n"];
    [mil appendFormat:
        @"        tensor<int32,[4]> hidden_size = const()["
         "name=string(\"hidden_size\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> hidden = "
         "slice_by_size(x=packed,begin=hidden_begin,size=hidden_size)"
         "[name=string(\"hidden\")];\n",
        DIM, SEQUENCE];
    if (QUALIFICATION_STAGE == 0) {
        [mil appendString:@"    } -> (hidden);\n}\n"];
        return mil;
    }
    [mil appendFormat:
        @"        tensor<int32,[4]> scale_begin = const()["
         "name=string(\"scale_begin\"),"
         "val=tensor<int32,[4]>([0,0,0,%d])];\n",
        SEQUENCE];
    [mil appendFormat:
        @"        tensor<int32,[4]> scale_size = const()["
         "name=string(\"scale_size\"),"
         "val=tensor<int32,[4]>([1,%d,1,1])];\n",
        DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,1]> norm_scale = "
         "slice_by_size(x=packed,begin=scale_begin,size=scale_size)"
         "[name=string(\"norm_scale\")];\n",
        DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> hidden_square = "
         "mul(x=hidden,y=hidden)[name=string(\"hidden_square\")];\n",
        DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,1,%d]> square_sum = "
         "reduce_sum(x=hidden_square,axes=channel_axis,keep_dims=keep)"
         "[name=string(\"square_sum\")];\n",
        SEQUENCE];
    [mil appendFormat:
        @"        fp16 inverse_dim = const()[name=string(\"inverse_dim\"),"
         "val=fp16(%.12f)];\n",
        1.0 / DIM];
    [mil appendString:
        @"        fp16 epsilon = const()[name=string(\"epsilon\"),"
         "val=fp16(0.00001)];\n"
         "        fp16 negative_half = const()[name=string(\"negative_half\"),"
         "val=fp16(-0.5)];\n"];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,1,%d]> square_mean = "
         "mul(x=square_sum,y=inverse_dim)[name=string(\"square_mean\")];\n",
        SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,1,%d]> variance = "
         "add(x=square_mean,y=epsilon)[name=string(\"variance\")];\n",
        SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,1,%d]> inverse_rms = "
         "pow(x=variance,y=negative_half)[name=string(\"inverse_rms\")];\n",
        SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> normalized = "
         "mul(x=hidden,y=inverse_rms)[name=string(\"normalized\")];\n",
        DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> xn = "
         "mul(x=normalized,y=norm_scale)[name=string(\"xn\")];\n",
        DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<int32,[4]> xn_matrix_shape = const()["
         "name=string(\"xn_matrix_shape\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n",
        DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> xn_matrix = "
         "reshape(shape=xn_matrix_shape,x=xn)[name=string(\"xn_matrix\")];\n",
        DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> xn_rows = "
         "transpose(perm=transpose_last,x=xn_matrix)"
         "[name=string(\"xn_rows\")];\n",
        SEQUENCE, DIM];
    if (QUALIFICATION_STAGE == 1) {
        [mil appendString:@"    } -> (xn);\n}\n"];
        return mil;
    }

    const int qOffset = SEQUENCE + NORM_SCALE_SPAN;
    const int kOffset = qOffset + QUERY_DIM;
    const int vOffset = kOffset + KV_DIM;
    AppendDynamicProjection(mil, @"q", QUERY_DIM, qOffset);
    AppendDynamicProjection(mil, @"k", KV_DIM, kOffset);
    AppendDynamicProjection(mil, @"v", KV_DIM, vOffset);
    if (QUALIFICATION_STAGE == 2) {
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,1,%d]> projection_output = "
             "concat(axis=head_axis,interleave=no_interleave,"
             "values=(q_channels,k_channels,v_channels,xn))"
             "[name=string(\"projection_output\")];\n",
            OUTPUT_CHANNELS, SEQUENCE];
        [mil appendString:@"    } -> (projection_output);\n}\n"];
        return mil;
    }

    [mil appendFormat:
        @"        tensor<int32,[4]> q_head_shape = const()["
         "name=string(\"q_head_shape\"),"
         "val=tensor<int32,[4]>([1,%d,%d,%d])];\n",
        QUERY_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<int32,[4]> kv_head_shape = const()["
         "name=string(\"kv_head_shape\"),"
         "val=tensor<int32,[4]>([1,%d,%d,%d])];\n",
        KV_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> q_head_channels = "
         "reshape(shape=q_head_shape,x=q_channels)"
         "[name=string(\"q_head_channels\")];\n",
        QUERY_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> k_head_channels = "
         "reshape(shape=kv_head_shape,x=k_channels)"
         "[name=string(\"k_head_channels\")];\n",
        KV_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> v_head_channels = "
         "reshape(shape=kv_head_shape,x=v_channels)"
         "[name=string(\"v_head_channels\")];\n",
        KV_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> q_heads = "
         "transpose(perm=transpose_last,x=q_head_channels)"
         "[name=string(\"q_heads\")];\n",
        QUERY_HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> k_heads = "
         "transpose(perm=transpose_last,x=k_head_channels)"
         "[name=string(\"k_heads\")];\n",
        KV_HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> v_heads = "
         "transpose(perm=transpose_last,x=v_head_channels)"
         "[name=string(\"v_heads\")];\n",
        KV_HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> rope_cos = const()["
         "name=string(\"rope_cos\"),val=tensor<fp16,[1,1,%d,%d]>("
         "BLOBFILE(path=string(\"@model_path/weights/cos.bin\"),"
         "offset=uint64(64)))];\n",
        SEQUENCE, HEAD_DIM, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> rope_sin = const()["
         "name=string(\"rope_sin\"),val=tensor<fp16,[1,1,%d,%d]>("
         "BLOBFILE(path=string(\"@model_path/weights/sin.bin\"),"
         "offset=uint64(64)))];\n",
        SEQUENCE, HEAD_DIM, SEQUENCE, HEAD_DIM];
    AppendSplitHalfRoPE(mil, @"q", @"q_heads", QUERY_HEADS);
    AppendSplitHalfRoPE(mil, @"k", @"k_heads", KV_HEADS);
    if (QUALIFICATION_STAGE == 3) {
        [mil appendFormat:
            @"        tensor<int32,[4]> stage3_query_shape = const()["
             "name=string(\"stage3_query_shape\"),"
             "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
            QUERY_DIM, SEQUENCE];
        [mil appendFormat:
            @"        tensor<int32,[4]> stage3_kv_shape = const()["
             "name=string(\"stage3_kv_shape\"),"
             "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
            KV_DIM, SEQUENCE];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,%d,%d]> stage3_q_channels = "
             "transpose(perm=transpose_last,x=q_rope)"
             "[name=string(\"stage3_q_channels\")];\n",
            QUERY_HEADS, HEAD_DIM, SEQUENCE];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,1,%d]> stage3_q = "
             "reshape(shape=stage3_query_shape,x=stage3_q_channels)"
             "[name=string(\"stage3_q\")];\n",
            QUERY_DIM, SEQUENCE];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,%d,%d]> stage3_k_channels = "
             "transpose(perm=transpose_last,x=k_rope)"
             "[name=string(\"stage3_k_channels\")];\n",
            KV_HEADS, HEAD_DIM, SEQUENCE];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,1,%d]> stage3_k = "
             "reshape(shape=stage3_kv_shape,x=stage3_k_channels)"
             "[name=string(\"stage3_k\")];\n",
            KV_DIM, SEQUENCE];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,1,%d]> stage3_output = "
             "concat(axis=head_axis,interleave=no_interleave,"
             "values=(stage3_q,stage3_k,v_channels,xn))"
             "[name=string(\"stage3_output\")];\n",
            OUTPUT_CHANNELS, SEQUENCE];
        [mil appendString:@"    } -> (stage3_output);\n}\n"];
        return mil;
    }

    [mil appendFormat:
        @"        tensor<int32,[4]> one_kv_size = const()["
         "name=string(\"one_kv_size\"),"
         "val=tensor<int32,[4]>([1,1,%d,%d])];\n",
        SEQUENCE, HEAD_DIM];
    NSMutableString *keys = [NSMutableString string];
    NSMutableString *values = [NSMutableString string];
    for (int kv = 0; kv < KV_HEADS; ++kv) {
        [mil appendFormat:
            @"        tensor<int32,[4]> kv_begin_%d = const()["
             "name=string(\"kv_begin_%d\"),"
             "val=tensor<int32,[4]>([0,%d,0,0])];\n",
            kv, kv, kv];
        [mil appendFormat:
            @"        tensor<fp16,[1,1,%d,%d]> key_%d = "
             "slice_by_size(x=k_rope,begin=kv_begin_%d,size=one_kv_size)"
             "[name=string(\"key_%d\")];\n",
            SEQUENCE, HEAD_DIM, kv, kv, kv];
        [mil appendFormat:
            @"        tensor<fp16,[1,1,%d,%d]> value_%d = "
             "slice_by_size(x=v_heads,begin=kv_begin_%d,size=one_kv_size)"
             "[name=string(\"value_%d\")];\n",
            SEQUENCE, HEAD_DIM, kv, kv, kv];
        for (int group = 0; group < QUERY_GROUPS; ++group) {
            if (kv || group) {
                [keys appendString:@","];
                [values appendString:@","];
            }
            [keys appendFormat:@"key_%d", kv];
            [values appendFormat:@"value_%d", kv];
        }
    }
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> tiled_key = "
         "concat(axis=head_axis,interleave=no_interleave,values=(%@))"
         "[name=string(\"tiled_key\")];\n",
        QUERY_HEADS, SEQUENCE, HEAD_DIM, keys];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> tiled_value = "
         "concat(axis=head_axis,interleave=no_interleave,values=(%@))"
         "[name=string(\"tiled_value\")];\n",
        QUERY_HEADS, SEQUENCE, HEAD_DIM, values];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> raw_scores = "
         "matmul(transpose_x=bf,transpose_y=bt,x=q_rope,y=tiled_key)"
         "[name=string(\"raw_scores\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        fp16 attention_scale = const()[name=string(\"attention_scale\"),"
         "val=fp16(%.12f)];\n",
        1.0 / sqrt((double)HEAD_DIM)];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> scaled_scores = "
         "mul(x=raw_scores,y=attention_scale)"
         "[name=string(\"scaled_scores\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,1,%d,%d]> causal_mask = const()["
         "name=string(\"causal_mask\"),val=tensor<fp16,[1,1,%d,%d]>("
         "BLOBFILE(path=string(\"@model_path/weights/mask.bin\"),"
         "offset=uint64(64)))];\n",
        SEQUENCE, SEQUENCE, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> masked_scores = "
         "add(x=scaled_scores,y=causal_mask)"
         "[name=string(\"masked_scores\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> probabilities = "
         "softmax(axis=softmax_axis,x=masked_scores)"
         "[name=string(\"probabilities\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> attended_heads = "
         "matmul(transpose_x=bf,transpose_y=bf,x=probabilities,y=tiled_value)"
         "[name=string(\"attended_heads\")];\n",
        QUERY_HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> attended_channels = "
         "transpose(perm=transpose_last,x=attended_heads)"
         "[name=string(\"attended_channels\")];\n",
        QUERY_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<int32,[4]> query_channels_shape = const()["
         "name=string(\"query_channels_shape\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        QUERY_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<int32,[4]> kv_channels_shape = const()["
         "name=string(\"kv_channels_shape\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        KV_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> attended = "
         "reshape(shape=query_channels_shape,x=attended_channels)"
         "[name=string(\"attended\")];\n",
        QUERY_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> q_rope_channels = "
         "transpose(perm=transpose_last,x=q_rope)"
         "[name=string(\"q_rope_channels\")];\n",
        QUERY_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> q_tap = "
         "reshape(shape=query_channels_shape,x=q_rope_channels)"
         "[name=string(\"q_tap\")];\n",
        QUERY_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> k_rope_channels = "
         "transpose(perm=transpose_last,x=k_rope)"
         "[name=string(\"k_rope_channels\")];\n",
        KV_HEADS, HEAD_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> k_tap = "
         "reshape(shape=kv_channels_shape,x=k_rope_channels)"
         "[name=string(\"k_tap\")];\n",
        KV_DIM, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> output = "
         "concat(axis=head_axis,interleave=no_interleave,"
         "values=(attended,q_tap,k_tap,v_channels,xn))"
         "[name=string(\"output\")];\n",
        OUTPUT_CHANNELS, SEQUENCE];
    [mil appendString:@"    } -> (output);\n}\n"];
    return mil;
}

static id CompileModel(
    NSString *mil, NSDictionary *weights, NSString **temporaryDirectory,
    NSError **error
) {
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

static BOOL Evaluate(
    id model, IOSurfaceRef input, IOSurfaceRef output, NSError **error
) {
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

static void BuildRoPE(float *cosine, float *sine) {
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int channel = 0; channel < HALF_HEAD_DIM; ++channel) {
            const float theta =
                position /
                powf(10000.0f, 2.0f * channel / (float)HEAD_DIM);
            const float cosValue = cosf(theta);
            const float sinValue = sinf(theta);
            cosine[position * HEAD_DIM + channel] = cosValue;
            cosine[position * HEAD_DIM + channel + HALF_HEAD_DIM] = cosValue;
            sine[position * HEAD_DIM + channel] = sinValue;
            sine[position * HEAD_DIM + channel + HALF_HEAD_DIM] = sinValue;
        }
    }
}

static void RMSNormReference(
    float *output, const float *hidden, const float *scale
) {
    for (int position = 0; position < SEQUENCE; ++position) {
        double sum = 0.0;
        for (int channel = 0; channel < DIM; ++channel) {
            const float value = hidden[position * DIM + channel];
            sum += (double)value * value;
        }
        const float inverse =
            1.0f / sqrtf((float)(sum / DIM) + 0.00001f);
        for (int channel = 0; channel < DIM; ++channel) {
            output[position * DIM + channel] =
                hidden[position * DIM + channel] * inverse * scale[channel];
        }
    }
}

static void ApplyRoPEReference(
    float *values, int heads, const float *cosine, const float *sine
) {
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int head = 0; head < heads; ++head) {
            float *row = values + (position * heads + head) * HEAD_DIM;
            float prior[HEAD_DIM];
            memcpy(prior, row, sizeof(prior));
            for (int channel = 0; channel < HEAD_DIM; ++channel) {
                const int paired = channel < HALF_HEAD_DIM
                    ? channel + HALF_HEAD_DIM
                    : channel - HALF_HEAD_DIM;
                const float rotated =
                    channel < HALF_HEAD_DIM ? -prior[paired] : prior[paired];
                row[channel] =
                    prior[channel] * cosine[position * HEAD_DIM + channel] +
                    rotated * sine[position * HEAD_DIM + channel];
            }
        }
    }
}

static void AttentionReference(
    float *output, const float *query, const float *key, const float *value
) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float scores[SEQUENCE];
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int head = 0; head < QUERY_HEADS; ++head) {
            const int kv = head / QUERY_GROUPS;
            const float *q = query + (position * QUERY_HEADS + head) * HEAD_DIM;
            float maximum = -INFINITY;
            for (int source = 0; source <= position; ++source) {
                const float *k =
                    key + (source * KV_HEADS + kv) * HEAD_DIM;
                float score = 0.0f;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    score += q[channel] * k[channel];
                }
                scores[source] = score * scale;
                maximum = fmaxf(maximum, scores[source]);
            }
            float denominator = 0.0f;
            for (int source = 0; source <= position; ++source) {
                scores[source] = expf(scores[source] - maximum);
                denominator += scores[source];
            }
            float *out =
                output + (position * QUERY_HEADS + head) * HEAD_DIM;
            memset(out, 0, HEAD_DIM * sizeof(float));
            for (int source = 0; source <= position; ++source) {
                const float probability = scores[source] / denominator;
                const float *v =
                    value + (source * KV_HEADS + kv) * HEAD_DIM;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    out[channel] += probability * v[channel];
                }
            }
        }
    }
}

typedef struct {
    size_t mismatches;
    float maximum;
    double squared;
    size_t elements;
} Comparison;

static Comparison CompareChannelMajorToRows(
    const _Float16 *actual, const float *expected, int channels, float tolerance
) {
    Comparison result = {0};
    result.elements = (size_t)channels * SEQUENCE;
    for (int channel = 0; channel < channels; ++channel) {
        for (int position = 0; position < SEQUENCE; ++position) {
            const float value = actual[(size_t)channel * SEQUENCE + position];
            const float target =
                expected[(size_t)position * channels + channel];
            const float delta = fabsf(value - target);
            if (delta > tolerance) ++result.mismatches;
            result.maximum = fmaxf(result.maximum, delta);
            result.squared += (double)delta * delta;
        }
    }
    return result;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&Timebase);
        if (!LoadPrivateANE()) {
            fprintf(stderr, "private_ane_classes_missing\n");
            return 2;
        }
        float *cosine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
        float *sine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
        float *mask = malloc(SEQUENCE * SEQUENCE * sizeof(float));
        BuildRoPE(cosine, sine);
        for (int row = 0; row < SEQUENCE; ++row) {
            for (int column = 0; column < SEQUENCE; ++column) {
                mask[row * SEQUENCE + column] =
                    column <= row ? 0.0f : -10000.0f;
            }
        }
        NSError *error = nil;
        NSString *temporaryDirectory = nil;
        const uint64_t compileStarted = mach_absolute_time();
        id model = CompileModel(
            ExactAttentionMIL(),
            @{
                @"@model_path/weights/cos.bin" :
                    @{@"offset" : @0,
                      @"data" :
                          BuildWeightBlob(cosine, SEQUENCE * HEAD_DIM)},
                @"@model_path/weights/sin.bin" :
                    @{@"offset" : @0,
                      @"data" :
                          BuildWeightBlob(sine, SEQUENCE * HEAD_DIM)},
                @"@model_path/weights/mask.bin" :
                    @{@"offset" : @0,
                      @"data" :
                          BuildWeightBlob(mask, SEQUENCE * SEQUENCE)}
            },
            &temporaryDirectory, &error);
        const double compileMilliseconds =
            Milliseconds(mach_absolute_time() - compileStarted);
        if (!model) {
            fprintf(stderr, "ane_compile_failed: %s\n",
                    error.description.UTF8String);
            return 3;
        }
        IOSurfaceRef input = CreateHalfSurface(INPUT_ELEMENTS);
        IOSurfaceRef output = CreateHalfSurface(OUTPUT_ELEMENTS);
        if (!input || !output) {
            fprintf(stderr, "iosurface_create_failed\n");
            return 4;
        }

        float *hidden = malloc(SEQUENCE * DIM * sizeof(float));
        float *normScale = malloc(DIM * sizeof(float));
        float *wq = malloc(DIM * QUERY_DIM * sizeof(float));
        float *wk = malloc(DIM * KV_DIM * sizeof(float));
        float *wv = malloc(DIM * KV_DIM * sizeof(float));
        for (int position = 0; position < SEQUENCE; ++position) {
            for (int channel = 0; channel < DIM; ++channel) {
                hidden[position * DIM + channel] =
                    sinf((position * 17 + channel * 3) * 0.013f) * 0.125f +
                    cosf((position * 5 + channel * 11) * 0.007f) * 0.03125f;
            }
        }
        for (int channel = 0; channel < DIM; ++channel) {
            normScale[channel] =
                1.0f + sinf(channel * 0.019f) * 0.01f;
            for (int out = 0; out < QUERY_DIM; ++out) {
                wq[channel * QUERY_DIM + out] =
                    sinf((channel * 29 + out * 7) * 0.003f) * 0.03125f;
            }
            for (int out = 0; out < KV_DIM; ++out) {
                wk[channel * KV_DIM + out] =
                    cosf((channel * 13 + out * 5) * 0.005f) * 0.03125f;
                wv[channel * KV_DIM + out] =
                    sinf((channel * 23 + out * 3) * 0.004f) * 0.03125f;
            }
        }

        IOSurfaceLock(input, 0, NULL);
        _Float16 *packed = IOSurfaceGetBaseAddress(input);
        memset(packed, 0, INPUT_ELEMENTS * sizeof(_Float16));
        const int qOffset = SEQUENCE + NORM_SCALE_SPAN;
        const int kOffset = qOffset + QUERY_DIM;
        const int vOffset = kOffset + KV_DIM;
        for (int channel = 0; channel < DIM; ++channel) {
            for (int position = 0; position < SEQUENCE; ++position) {
                packed[(size_t)channel * PACKED_SPATIAL + position] =
                    (_Float16)hidden[position * DIM + channel];
            }
            packed[(size_t)channel * PACKED_SPATIAL + SEQUENCE] =
                (_Float16)normScale[channel];
            for (int out = 0; out < QUERY_DIM; ++out) {
                packed[(size_t)channel * PACKED_SPATIAL + qOffset + out] =
                    (_Float16)wq[channel * QUERY_DIM + out];
            }
            for (int out = 0; out < KV_DIM; ++out) {
                packed[(size_t)channel * PACKED_SPATIAL + kOffset + out] =
                    (_Float16)wk[channel * KV_DIM + out];
                packed[(size_t)channel * PACKED_SPATIAL + vOffset + out] =
                    (_Float16)wv[channel * KV_DIM + out];
            }
        }
        IOSurfaceUnlock(input, 0, NULL);

        for (int warmup = 0; warmup < 2; ++warmup) {
            if (!Evaluate(model, input, output, &error)) {
                fprintf(stderr, "ane_warmup_failed: %s\n",
                        error.description.UTF8String);
                return 5;
            }
        }
        const int repetitions = 8;
        const uint64_t evaluationStarted = mach_absolute_time();
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            if (!Evaluate(model, input, output, &error)) {
                fprintf(stderr, "ane_eval_failed: %s\n",
                        error.description.UTF8String);
                return 6;
            }
        }
        const double meanMilliseconds =
            Milliseconds(mach_absolute_time() - evaluationStarted) / repetitions;
        if (QUALIFICATION_STAGE < 4) {
            printf(
                "{\"policy\":\"project_theseus_exact_ane_attention_bisect_v1\","
                "\"qualification_stage\":%d,\"compile_milliseconds\":%.6f,"
                "\"mean_evaluation_milliseconds\":%.6f,"
                "\"trigger_state\":\"GREEN_RUNTIME_EXECUTION\"}\n",
                QUALIFICATION_STAGE, compileMilliseconds, meanMilliseconds);
            return 0;
        }

        float *normalized = malloc(SEQUENCE * DIM * sizeof(float));
        float *query = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *key = malloc(SEQUENCE * KV_DIM * sizeof(float));
        float *value = malloc(SEQUENCE * KV_DIM * sizeof(float));
        float *attended = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        RMSNormReference(normalized, hidden, normScale);
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            SEQUENCE, QUERY_DIM, DIM, 1.0f, normalized, DIM,
            wq, QUERY_DIM, 0.0f, query, QUERY_DIM);
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            SEQUENCE, KV_DIM, DIM, 1.0f, normalized, DIM,
            wk, KV_DIM, 0.0f, key, KV_DIM);
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            SEQUENCE, KV_DIM, DIM, 1.0f, normalized, DIM,
            wv, KV_DIM, 0.0f, value, KV_DIM);
        ApplyRoPEReference(query, QUERY_HEADS, cosine, sine);
        ApplyRoPEReference(key, KV_HEADS, cosine, sine);
        AttentionReference(attended, query, key, value);

        IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *actual = IOSurfaceGetBaseAddress(output);
        size_t offset = 0;
        Comparison attendedComparison = CompareChannelMajorToRows(
            actual + offset, attended, QUERY_DIM, 0.02f);
        offset += (size_t)QUERY_DIM * SEQUENCE;
        Comparison queryComparison = CompareChannelMajorToRows(
            actual + offset, query, QUERY_DIM, 0.01f);
        offset += (size_t)QUERY_DIM * SEQUENCE;
        Comparison keyComparison = CompareChannelMajorToRows(
            actual + offset, key, KV_DIM, 0.01f);
        offset += (size_t)KV_DIM * SEQUENCE;
        Comparison valueComparison = CompareChannelMajorToRows(
            actual + offset, value, KV_DIM, 0.01f);
        offset += (size_t)KV_DIM * SEQUENCE;
        Comparison normComparison = CompareChannelMajorToRows(
            actual + offset, normalized, DIM, 0.01f);
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
        const size_t mismatches =
            attendedComparison.mismatches + queryComparison.mismatches +
            keyComparison.mismatches + valueComparison.mismatches +
            normComparison.mismatches;
        printf(
            "{\"policy\":\"project_theseus_exact_ane_attention_forward_v1\","
            "\"shape\":{\"sequence\":%d,\"d_model\":%d,"
            "\"query_heads\":%d,\"kv_heads\":%d,\"head_dim\":%d},"
            "\"parameter_generation\":0,"
            "\"compile_milliseconds\":%.6f,"
            "\"mean_evaluation_milliseconds\":%.6f,"
            "\"comparisons\":{"
            "\"attended\":{\"tolerance\":0.02,\"maximum_absolute_delta\":%.9g,"
            "\"mismatch_count\":%zu},"
            "\"query_rope\":{\"tolerance\":0.01,\"maximum_absolute_delta\":%.9g,"
            "\"mismatch_count\":%zu},"
            "\"key_rope\":{\"tolerance\":0.01,\"maximum_absolute_delta\":%.9g,"
            "\"mismatch_count\":%zu},"
            "\"value\":{\"tolerance\":0.01,\"maximum_absolute_delta\":%.9g,"
            "\"mismatch_count\":%zu},"
            "\"attention_norm\":{\"tolerance\":0.01,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu}},"
            "\"gates\":{\"dynamic_weights\":true,"
            "\"split_half_rope\":true,\"contiguous_gqa\":true,"
            "\"causal_attention\":true,\"backward\":false,"
            "\"complete_decoder_block\":false,\"production_eligible\":false},"
            "\"mismatch_count\":%zu,\"trigger_state\":\"%s\","
            "\"capability_claim\":\"NONE_ENGINEERING_FORWARD_SLICE_ONLY\"}\n",
            SEQUENCE, DIM, QUERY_HEADS, KV_HEADS, HEAD_DIM,
            compileMilliseconds, meanMilliseconds,
            attendedComparison.maximum, attendedComparison.mismatches,
            queryComparison.maximum, queryComparison.mismatches,
            keyComparison.maximum, keyComparison.mismatches,
            valueComparison.maximum, valueComparison.mismatches,
            normComparison.maximum, normComparison.mismatches,
            mismatches, mismatches == 0 ? "GREEN" : "RED");

        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        [[NSFileManager defaultManager]
            removeItemAtPath:temporaryDirectory error:nil];
        CFRelease(input);
        CFRelease(output);
        free(cosine);
        free(sine);
        free(mask);
        free(hidden);
        free(normScale);
        free(wq);
        free(wk);
        free(wv);
        free(normalized);
        free(query);
        free(key);
        free(value);
        free(attended);
        return mismatches == 0 ? 0 : 1;
    }
}
