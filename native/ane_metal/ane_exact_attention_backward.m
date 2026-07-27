/*
 * Exact Project Theseus causal-attention core backward qualification.
 *
 * Input: RoPE'd Q, contiguously tiled K/V, and upstream dAttention.
 * Output: full-query-head dQ, dK, and dV. The later closure must reduce
 * dK/dV to KV heads, apply inverse split-half RoPE, compute RMSNorm dX and
 * every FP32 parameter gradient, then join the rest of the decoder block.
 *
 * This translation unit reuses the attributed private-runtime transport from
 * the exact forward probe. See THIRD_PARTY_NOTICES.md.
 */

#define main theseus_exact_attention_forward_embedded_main
#include "ane_exact_attention_forward.m"
#undef main

#define BACKWARD_INPUT_CHANNELS (4 * QUERY_DIM)
#define BACKWARD_OUTPUT_CHANNELS (3 * QUERY_DIM)
#define BACKWARD_INPUT_ELEMENTS ((size_t)BACKWARD_INPUT_CHANNELS * SEQUENCE)
#define BACKWARD_OUTPUT_ELEMENTS ((size_t)BACKWARD_OUTPUT_CHANNELS * SEQUENCE)

static NSString *ExactAttentionBackwardMIL(void) {
    NSMutableString *mil = [NSMutableString stringWithString:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({"
         "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, "
         "{\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [mil appendFormat:
        @"    func main<ios18>(tensor<fp16,[1,%d,1,%d]> packed) {\n",
        BACKWARD_INPUT_CHANNELS, SEQUENCE];
    [mil appendString:
        @"        bool bf = const()[name=string(\"bf\"),val=bool(false)];\n"
         "        bool bt = const()[name=string(\"bt\"),val=bool(true)];\n"
         "        bool keep = const()[name=string(\"keep\"),val=bool(true)];\n"
         "        bool no_interleave = const()[name=string(\"no_interleave\"),"
         "val=bool(false)];\n"
         "        int32 channel_axis = const()[name=string(\"channel_axis\"),"
         "val=int32(1)];\n"
         "        int32 softmax_axis = const()[name=string(\"softmax_axis\"),"
         "val=int32(-1)];\n"
         "        tensor<int32,[4]> transpose_last = const()["
         "name=string(\"transpose_last\"),"
         "val=tensor<int32,[4]>([0,1,3,2])];\n"
         "        tensor<int32,[1]> last_axis_vector = const()["
         "name=string(\"last_axis_vector\"),"
         "val=tensor<int32,[1]>([3])];\n"];
    [mil appendFormat:
        @"        tensor<int32,[4]> one_begin = const()["
         "name=string(\"one_begin\"),"
         "val=tensor<int32,[4]>([0,0,0,0])];\n"
         "        tensor<int32,[4]> one_size = const()["
         "name=string(\"one_size\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        QUERY_DIM, SEQUENCE];
    NSArray<NSString *> *names = @[ @"q", @"k", @"v", @"da" ];
    for (int index = 0; index < (int)names.count; ++index) {
        NSString *name = names[index];
        [mil appendFormat:
            @"        tensor<int32,[4]> %@_begin = const()["
             "name=string(\"%@_begin\"),"
             "val=tensor<int32,[4]>([0,%d,0,0])];\n",
            name, name, index * QUERY_DIM];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,1,%d]> %@_channels = "
             "slice_by_size(x=packed,begin=%@_begin,size=one_size)"
             "[name=string(\"%@_channels\")];\n",
            QUERY_DIM, SEQUENCE, name, name, name];
        [mil appendFormat:
            @"        tensor<int32,[4]> %@_head_shape = const()["
             "name=string(\"%@_head_shape\"),"
             "val=tensor<int32,[4]>([1,%d,%d,%d])];\n",
            name, name, QUERY_HEADS, HEAD_DIM, SEQUENCE];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,%d,%d]> %@_head_channels = "
             "reshape(shape=%@_head_shape,x=%@_channels)"
             "[name=string(\"%@_head_channels\")];\n",
            QUERY_HEADS, HEAD_DIM, SEQUENCE, name, name, name, name];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,%d,%d]> %@_heads = "
             "transpose(perm=transpose_last,x=%@_head_channels)"
             "[name=string(\"%@_heads\")];\n",
            QUERY_HEADS, SEQUENCE, HEAD_DIM, name, name, name];
    }
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> raw_scores = "
         "matmul(transpose_x=bf,transpose_y=bt,x=q_heads,y=k_heads)"
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
        @"        tensor<fp16,[1,%d,%d,%d]> dv_heads = "
         "matmul(transpose_x=bt,transpose_y=bf,x=probabilities,y=da_heads)"
         "[name=string(\"dv_heads\")];\n",
        QUERY_HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> dprobabilities = "
         "matmul(transpose_x=bf,transpose_y=bt,x=da_heads,y=v_heads)"
         "[name=string(\"dprobabilities\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> probability_products = "
         "mul(x=probabilities,y=dprobabilities)"
         "[name=string(\"probability_products\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,1]> row_dot = "
         "reduce_sum(x=probability_products,axes=last_axis_vector,"
         "keep_dims=keep)[name=string(\"row_dot\")];\n",
        QUERY_HEADS, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> centered = "
         "sub(x=dprobabilities,y=row_dot)[name=string(\"centered\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> probability_gradient = "
         "mul(x=probabilities,y=centered)"
         "[name=string(\"probability_gradient\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> score_gradient = "
         "mul(x=probability_gradient,y=attention_scale)"
         "[name=string(\"score_gradient\")];\n",
        QUERY_HEADS, SEQUENCE, SEQUENCE];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> dq_heads = "
         "matmul(transpose_x=bf,transpose_y=bf,x=score_gradient,y=k_heads)"
         "[name=string(\"dq_heads\")];\n",
        QUERY_HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,%d,%d]> dk_heads = "
         "matmul(transpose_x=bt,transpose_y=bf,x=score_gradient,y=q_heads)"
         "[name=string(\"dk_heads\")];\n",
        QUERY_HEADS, SEQUENCE, HEAD_DIM];
    [mil appendFormat:
        @"        tensor<int32,[4]> output_shape = const()["
         "name=string(\"output_shape\"),"
         "val=tensor<int32,[4]>([1,%d,1,%d])];\n",
        QUERY_DIM, SEQUENCE];
    for (NSString *name in @[ @"dq", @"dk", @"dv" ]) {
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,%d,%d]> %@_channels4 = "
             "transpose(perm=transpose_last,x=%@_heads)"
             "[name=string(\"%@_channels4\")];\n",
            QUERY_HEADS, HEAD_DIM, SEQUENCE, name, name, name];
        [mil appendFormat:
            @"        tensor<fp16,[1,%d,1,%d]> %@_output = "
             "reshape(shape=output_shape,x=%@_channels4)"
             "[name=string(\"%@_output\")];\n",
            QUERY_DIM, SEQUENCE, name, name, name];
    }
    [mil appendFormat:
        @"        tensor<fp16,[1,%d,1,%d]> output = "
         "concat(axis=channel_axis,interleave=no_interleave,"
         "values=(dq_output,dk_output,dv_output))"
         "[name=string(\"output\")];\n",
        BACKWARD_OUTPUT_CHANNELS, SEQUENCE];
    [mil appendString:@"    } -> (output);\n}\n"];
    return mil;
}

static void BuildInputs(
    float *query, float *key, float *value, float *upstream
) {
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int head = 0; head < QUERY_HEADS; ++head) {
            for (int channel = 0; channel < HEAD_DIM; ++channel) {
                const size_t index =
                    ((size_t)position * QUERY_HEADS + head) * HEAD_DIM + channel;
                const int kv = head / QUERY_GROUPS;
                query[index] =
                    sinf((position * 17 + head * 23 + channel * 3) * 0.011f)
                    * 0.125f;
                key[index] =
                    cosf((position * 13 + kv * 31 + channel * 5) * 0.009f)
                    * 0.125f;
                value[index] =
                    sinf((position * 7 + kv * 19 + channel * 11) * 0.008f)
                    * 0.0625f;
                upstream[index] =
                    cosf((position * 29 + head * 5 + channel * 7) * 0.006f)
                    * 0.00390625f;
            }
        }
    }
}

static void AttentionBackwardReference(
    float *dq, float *dk, float *dv, const float *query, const float *key,
    const float *value, const float *upstream
) {
    memset(dq, 0, QUERY_DIM * SEQUENCE * sizeof(float));
    memset(dk, 0, QUERY_DIM * SEQUENCE * sizeof(float));
    memset(dv, 0, QUERY_DIM * SEQUENCE * sizeof(float));
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    float *probabilities = calloc(
        (size_t)QUERY_HEADS * SEQUENCE * SEQUENCE, sizeof(float));
    float scores[SEQUENCE];
    for (int head = 0; head < QUERY_HEADS; ++head) {
        for (int row = 0; row < SEQUENCE; ++row) {
            const float *q =
                query + ((size_t)row * QUERY_HEADS + head) * HEAD_DIM;
            float maximum = -INFINITY;
            for (int column = 0; column <= row; ++column) {
                const float *k =
                    key + ((size_t)column * QUERY_HEADS + head) * HEAD_DIM;
                float score = 0.0f;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    score += q[channel] * k[channel];
                }
                scores[column] = score * scale;
                maximum = fmaxf(maximum, scores[column]);
            }
            float denominator = 0.0f;
            for (int column = 0; column <= row; ++column) {
                scores[column] = expf(scores[column] - maximum);
                denominator += scores[column];
            }
            for (int column = 0; column <= row; ++column) {
                probabilities[((size_t)head * SEQUENCE + row) * SEQUENCE
                              + column] = scores[column] / denominator;
            }
        }
    }
    for (int head = 0; head < QUERY_HEADS; ++head) {
        for (int row = 0; row < SEQUENCE; ++row) {
            const float *da =
                upstream + ((size_t)row * QUERY_HEADS + head) * HEAD_DIM;
            float dprobability[SEQUENCE] = {0};
            float rowDot = 0.0f;
            for (int column = 0; column <= row; ++column) {
                const float *v =
                    value + ((size_t)column * QUERY_HEADS + head) * HEAD_DIM;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    dprobability[column] += da[channel] * v[channel];
                }
                const float probability =
                    probabilities[((size_t)head * SEQUENCE + row) * SEQUENCE
                                  + column];
                rowDot += probability * dprobability[column];
                float *dvRow =
                    dv + ((size_t)column * QUERY_HEADS + head) * HEAD_DIM;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    dvRow[channel] += probability * da[channel];
                }
            }
            float *dqRow =
                dq + ((size_t)row * QUERY_HEADS + head) * HEAD_DIM;
            const float *qRow =
                query + ((size_t)row * QUERY_HEADS + head) * HEAD_DIM;
            for (int column = 0; column <= row; ++column) {
                const float probability =
                    probabilities[((size_t)head * SEQUENCE + row) * SEQUENCE
                                  + column];
                const float ds =
                    probability * (dprobability[column] - rowDot) * scale;
                const float *kRow =
                    key + ((size_t)column * QUERY_HEADS + head) * HEAD_DIM;
                float *dkRow =
                    dk + ((size_t)column * QUERY_HEADS + head) * HEAD_DIM;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    dqRow[channel] += ds * kRow[channel];
                    dkRow[channel] += ds * qRow[channel];
                }
            }
        }
    }
    free(probabilities);
}

static void PackChannelMajor(
    _Float16 *destination, const float *rows, int channels
) {
    for (int channel = 0; channel < channels; ++channel) {
        for (int position = 0; position < SEQUENCE; ++position) {
            destination[(size_t)channel * SEQUENCE + position] =
                (_Float16)rows[(size_t)position * channels + channel];
        }
    }
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        mach_timebase_info(&Timebase);
        if (!LoadPrivateANE()) {
            fprintf(stderr, "private_ane_classes_missing\n");
            return 2;
        }
        float *mask = malloc(SEQUENCE * SEQUENCE * sizeof(float));
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
            ExactAttentionBackwardMIL(),
            @{
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
        IOSurfaceRef input = CreateHalfSurface(BACKWARD_INPUT_ELEMENTS);
        IOSurfaceRef output = CreateHalfSurface(BACKWARD_OUTPUT_ELEMENTS);
        if (!input || !output) {
            fprintf(stderr, "iosurface_create_failed\n");
            return 4;
        }
        float *query = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *key = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *value = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *upstream = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        BuildInputs(query, key, value, upstream);
        IOSurfaceLock(input, 0, NULL);
        _Float16 *packed = IOSurfaceGetBaseAddress(input);
        PackChannelMajor(packed, query, QUERY_DIM);
        PackChannelMajor(
            packed + (size_t)QUERY_DIM * SEQUENCE, key, QUERY_DIM);
        PackChannelMajor(
            packed + (size_t)2 * QUERY_DIM * SEQUENCE, value, QUERY_DIM);
        PackChannelMajor(
            packed + (size_t)3 * QUERY_DIM * SEQUENCE, upstream, QUERY_DIM);
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
        float *dq = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *dk = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *dv = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        AttentionBackwardReference(
            dq, dk, dv, query, key, value, upstream);
        IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *actual = IOSurfaceGetBaseAddress(output);
        Comparison dqComparison =
            CompareChannelMajorToRows(actual, dq, QUERY_DIM, 0.002f);
        Comparison dkComparison = CompareChannelMajorToRows(
            actual + (size_t)QUERY_DIM * SEQUENCE,
            dk, QUERY_DIM, 0.002f);
        Comparison dvComparison = CompareChannelMajorToRows(
            actual + (size_t)2 * QUERY_DIM * SEQUENCE,
            dv, QUERY_DIM, 0.002f);
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
        const size_t mismatches =
            dqComparison.mismatches + dkComparison.mismatches
            + dvComparison.mismatches;
        printf(
            "{\"policy\":\"project_theseus_exact_ane_attention_backward_v1\","
            "\"shape\":{\"sequence\":%d,\"query_heads\":%d,"
            "\"kv_heads\":%d,\"head_dim\":%d},"
            "\"compile_milliseconds\":%.6f,"
            "\"mean_evaluation_milliseconds\":%.6f,"
            "\"comparisons\":{"
            "\"dq_rope\":{\"tolerance\":0.002,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"dk_tiled_rope\":{\"tolerance\":0.002,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"dv_tiled\":{\"tolerance\":0.002,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu}},"
            "\"gates\":{\"causal_softmax_backward\":true,"
            "\"full_query_head_gradients\":true,"
            "\"kv_reduction\":false,\"inverse_split_half_rope\":false,"
            "\"input_gradient\":false,\"parameter_gradients\":false,"
            "\"complete_decoder_block\":false,"
            "\"production_eligible\":false},"
            "\"mismatch_count\":%zu,\"trigger_state\":\"%s\","
            "\"capability_claim\":\"NONE_ENGINEERING_BACKWARD_CORE_ONLY\"}\n",
            SEQUENCE, QUERY_HEADS, KV_HEADS, HEAD_DIM,
            compileMilliseconds, meanMilliseconds,
            dqComparison.maximum, dqComparison.mismatches,
            dkComparison.maximum, dkComparison.mismatches,
            dvComparison.maximum, dvComparison.mismatches,
            mismatches, mismatches == 0 ? "GREEN" : "RED");
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        [[NSFileManager defaultManager]
            removeItemAtPath:temporaryDirectory error:nil];
        CFRelease(input);
        CFRelease(output);
        free(mask);
        free(query);
        free(key);
        free(value);
        free(upstream);
        free(dq);
        free(dk);
        free(dv);
        return mismatches == 0 ? 0 : 1;
    }
}
