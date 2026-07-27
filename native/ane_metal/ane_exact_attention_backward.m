/*
 * Exact Project Theseus causal-attention core backward qualification.
 *
 * Input: forward-coherent RoPE'd Q, contiguously tiled K/V, and upstream
 * dAttention. Output: full-query-head dQ, dK, and dV. The host-side
 * qualification closes contiguous GQA reduction, inverse split-half RoPE,
 * FP32 Q/K/V parameter gradients, and RMSNorm dX/scale against an independent
 * scalar reference. The rest of the decoder block remains outside this slice.
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

static void TileContiguousKV(
    float *tiled, const float *compact
) {
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int head = 0; head < QUERY_HEADS; ++head) {
            const int kv = head / QUERY_GROUPS;
            for (int channel = 0; channel < HEAD_DIM; ++channel) {
                const size_t destination =
                    ((size_t)position * QUERY_HEADS + head) * HEAD_DIM + channel;
                const size_t source =
                    ((size_t)position * KV_HEADS + kv) * HEAD_DIM + channel;
                tiled[destination] = compact[source];
            }
        }
    }
}

static void QuantizeHalfInPlace(float *values, size_t elements) {
    for (size_t index = 0; index < elements; ++index) {
        values[index] = (float)(_Float16)values[index];
    }
}

static void BuildCoherentInputs(
    float *hidden, float *normScale, float *normalized,
    float *wq, float *wk, float *wv,
    float *query, float *compactKey, float *compactValue, float *upstream
) {
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int channel = 0; channel < DIM; ++channel) {
            hidden[position * DIM + channel] =
                sinf((position * 17 + channel * 3) * 0.013f) * 0.125f
                + cosf((position * 5 + channel * 11) * 0.007f) * 0.03125f;
        }
    }
    for (int channel = 0; channel < DIM; ++channel) {
        normScale[channel] = 1.0f + sinf(channel * 0.019f) * 0.01f;
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
    QuantizeHalfInPlace(hidden, (size_t)SEQUENCE * DIM);
    QuantizeHalfInPlace(normScale, DIM);
    QuantizeHalfInPlace(wq, (size_t)DIM * QUERY_DIM);
    QuantizeHalfInPlace(wk, (size_t)DIM * KV_DIM);
    QuantizeHalfInPlace(wv, (size_t)DIM * KV_DIM);
    RMSNormReference(normalized, hidden, normScale);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        SEQUENCE, QUERY_DIM, DIM, 1.0f, normalized, DIM,
        wq, QUERY_DIM, 0.0f, query, QUERY_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        SEQUENCE, KV_DIM, DIM, 1.0f, normalized, DIM,
        wk, KV_DIM, 0.0f, compactKey, KV_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        SEQUENCE, KV_DIM, DIM, 1.0f, normalized, DIM,
        wv, KV_DIM, 0.0f, compactValue, KV_DIM);
    float *cosine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
    float *sine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
    BuildRoPE(cosine, sine);
    ApplyRoPEReference(query, QUERY_HEADS, cosine, sine);
    ApplyRoPEReference(compactKey, KV_HEADS, cosine, sine);
    free(cosine);
    free(sine);
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int head = 0; head < QUERY_HEADS; ++head) {
            for (int channel = 0; channel < HEAD_DIM; ++channel) {
                const size_t index =
                    ((size_t)position * QUERY_HEADS + head) * HEAD_DIM + channel;
                upstream[index] =
                    cosf((position * 29 + head * 5 + channel * 7) * 0.006f)
                    * 0.00390625f;
            }
        }
    }
    QuantizeHalfInPlace(normalized, (size_t)SEQUENCE * DIM);
    QuantizeHalfInPlace(query, (size_t)SEQUENCE * QUERY_DIM);
    QuantizeHalfInPlace(compactKey, (size_t)SEQUENCE * KV_DIM);
    QuantizeHalfInPlace(compactValue, (size_t)SEQUENCE * KV_DIM);
    QuantizeHalfInPlace(upstream, (size_t)SEQUENCE * QUERY_DIM);
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

static void ReadChannelMajorToRows(
    float *rows, const _Float16 *source, int channels
) {
    for (int channel = 0; channel < channels; ++channel) {
        for (int position = 0; position < SEQUENCE; ++position) {
            rows[(size_t)position * channels + channel] =
                source[(size_t)channel * SEQUENCE + position];
        }
    }
}

static void ReduceContiguousKV(float *reduced, const float *tiled) {
    memset(reduced, 0, (size_t)SEQUENCE * KV_DIM * sizeof(float));
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int kv = 0; kv < KV_HEADS; ++kv) {
            for (int group = 0; group < QUERY_GROUPS; ++group) {
                const int head = kv * QUERY_GROUPS + group;
                const float *source =
                    tiled + ((size_t)position * QUERY_HEADS + head) * HEAD_DIM;
                float *destination =
                    reduced + ((size_t)position * KV_HEADS + kv) * HEAD_DIM;
                for (int channel = 0; channel < HEAD_DIM; ++channel) {
                    destination[channel] += source[channel];
                }
            }
        }
    }
}

static void InverseSplitHalfRoPEGradient(
    float *gradient, int heads, const float *cosine, const float *sine
) {
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int head = 0; head < heads; ++head) {
            float *row = gradient + ((size_t)position * heads + head) * HEAD_DIM;
            float prior[HEAD_DIM];
            memcpy(prior, row, sizeof(prior));
            for (int channel = 0; channel < HEAD_DIM; ++channel) {
                const int paired = channel < HALF_HEAD_DIM
                    ? channel + HALF_HEAD_DIM
                    : channel - HALF_HEAD_DIM;
                const float rotated =
                    channel < HALF_HEAD_DIM ? -prior[paired] : prior[paired];
                row[channel] =
                    prior[channel] * cosine[position * HEAD_DIM + channel]
                    - rotated * sine[position * HEAD_DIM + channel];
            }
        }
    }
}

static void ProjectionGradientsAccelerate(
    const float *normalized, const float *wq, const float *wk, const float *wv,
    const float *dq, const float *dk, const float *dv,
    float *dwq, float *dwk, float *dwv, float *dnormalized
) {
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        DIM, QUERY_DIM, SEQUENCE, 1.0f, normalized, DIM,
        dq, QUERY_DIM, 0.0f, dwq, QUERY_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        DIM, KV_DIM, SEQUENCE, 1.0f, normalized, DIM,
        dk, KV_DIM, 0.0f, dwk, KV_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        DIM, KV_DIM, SEQUENCE, 1.0f, normalized, DIM,
        dv, KV_DIM, 0.0f, dwv, KV_DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        SEQUENCE, DIM, QUERY_DIM, 1.0f, dq, QUERY_DIM,
        wq, QUERY_DIM, 0.0f, dnormalized, DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        SEQUENCE, DIM, KV_DIM, 1.0f, dk, KV_DIM,
        wk, KV_DIM, 1.0f, dnormalized, DIM);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        SEQUENCE, DIM, KV_DIM, 1.0f, dv, KV_DIM,
        wv, KV_DIM, 1.0f, dnormalized, DIM);
}

static void ProjectionGradientScalarOne(
    const float *normalized, const float *weight, const float *gradient,
    int outputChannels, float *weightGradient, float *inputGradient
) {
    for (int input = 0; input < DIM; ++input) {
        for (int output = 0; output < outputChannels; ++output) {
            double sum = 0.0;
            for (int position = 0; position < SEQUENCE; ++position) {
                sum += (double)normalized[position * DIM + input]
                    * gradient[position * outputChannels + output];
            }
            weightGradient[input * outputChannels + output] = (float)sum;
        }
    }
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int input = 0; input < DIM; ++input) {
            double sum = 0.0;
            for (int output = 0; output < outputChannels; ++output) {
                sum += (double)gradient[position * outputChannels + output]
                    * weight[input * outputChannels + output];
            }
            inputGradient[position * DIM + input] += (float)sum;
        }
    }
}

static void ProjectionGradientsScalar(
    const float *normalized, const float *wq, const float *wk, const float *wv,
    const float *dq, const float *dk, const float *dv,
    float *dwq, float *dwk, float *dwv, float *dnormalized
) {
    memset(dnormalized, 0, (size_t)SEQUENCE * DIM * sizeof(float));
    ProjectionGradientScalarOne(
        normalized, wq, dq, QUERY_DIM, dwq, dnormalized);
    ProjectionGradientScalarOne(
        normalized, wk, dk, KV_DIM, dwk, dnormalized);
    ProjectionGradientScalarOne(
        normalized, wv, dv, KV_DIM, dwv, dnormalized);
}

static void RMSNormBackwardReference(
    const float *hidden, const float *scale, const float *dnormalized,
    float *dhidden, float *dscale
) {
    memset(dscale, 0, DIM * sizeof(float));
    for (int position = 0; position < SEQUENCE; ++position) {
        double squareSum = 0.0;
        for (int channel = 0; channel < DIM; ++channel) {
            const float value = hidden[position * DIM + channel];
            squareSum += (double)value * value;
        }
        const float inverse =
            1.0f / sqrtf((float)(squareSum / DIM) + 0.00001f);
        double weightedDot = 0.0;
        for (int channel = 0; channel < DIM; ++channel) {
            const size_t index = (size_t)position * DIM + channel;
            dscale[channel] +=
                dnormalized[index] * hidden[index] * inverse;
            weightedDot += (double)dnormalized[index] * scale[channel]
                * hidden[index];
        }
        const float correction =
            inverse * inverse * inverse * (float)(weightedDot / DIM);
        for (int channel = 0; channel < DIM; ++channel) {
            const size_t index = (size_t)position * DIM + channel;
            dhidden[index] =
                inverse * dnormalized[index] * scale[channel]
                - hidden[index] * correction;
        }
    }
}

static Comparison CompareFloatArrays(
    const float *actual, const float *expected, size_t elements, float tolerance
) {
    Comparison result = {0};
    result.elements = elements;
    for (size_t index = 0; index < elements; ++index) {
        const float delta = fabsf(actual[index] - expected[index]);
        if (delta > tolerance) ++result.mismatches;
        result.maximum = fmaxf(result.maximum, delta);
        result.squared += (double)delta * delta;
    }
    return result;
}

typedef struct {
    Comparison comparison;
    float maximumAllowed;
} PropagatedComparison;

static PropagatedComparison CompareWeightGradientPropagation(
    const float *actual, const float *expected, const float *activation,
    const float *actualGradient, const float *expectedGradient,
    int outputChannels
) {
    PropagatedComparison result = {0};
    result.comparison.elements = (size_t)DIM * outputChannels;
    for (int input = 0; input < DIM; ++input) {
        for (int output = 0; output < outputChannels; ++output) {
            double bound = 0.0001;
            for (int position = 0; position < SEQUENCE; ++position) {
                bound += fabs(
                    (double)activation[position * DIM + input]
                    * (actualGradient[position * outputChannels + output]
                       - expectedGradient[
                           position * outputChannels + output]));
            }
            const size_t index = (size_t)input * outputChannels + output;
            const float delta = fabsf(actual[index] - expected[index]);
            if (delta > bound) ++result.comparison.mismatches;
            result.comparison.maximum =
                fmaxf(result.comparison.maximum, delta);
            result.maximumAllowed =
                fmaxf(result.maximumAllowed, (float)bound);
        }
    }
    return result;
}

static PropagatedComparison CompareProjectionInputPropagation(
    const float *actual, const float *expected,
    const float *wq, const float *wk, const float *wv,
    const float *actualDQ, const float *expectedDQ,
    const float *actualDK, const float *expectedDK,
    const float *actualDV, const float *expectedDV
) {
    PropagatedComparison result = {0};
    result.comparison.elements = (size_t)SEQUENCE * DIM;
    for (int position = 0; position < SEQUENCE; ++position) {
        for (int input = 0; input < DIM; ++input) {
            double bound = 0.0001;
            for (int output = 0; output < QUERY_DIM; ++output) {
                bound += fabs(
                    (double)(actualDQ[position * QUERY_DIM + output]
                             - expectedDQ[position * QUERY_DIM + output])
                    * wq[input * QUERY_DIM + output]);
            }
            for (int output = 0; output < KV_DIM; ++output) {
                bound += fabs(
                    (double)(actualDK[position * KV_DIM + output]
                             - expectedDK[position * KV_DIM + output])
                    * wk[input * KV_DIM + output]);
                bound += fabs(
                    (double)(actualDV[position * KV_DIM + output]
                             - expectedDV[position * KV_DIM + output])
                    * wv[input * KV_DIM + output]);
            }
            const size_t index = (size_t)position * DIM + input;
            const float delta = fabsf(actual[index] - expected[index]);
            if (delta > bound) ++result.comparison.mismatches;
            result.comparison.maximum =
                fmaxf(result.comparison.maximum, delta);
            result.maximumAllowed =
                fmaxf(result.maximumAllowed, (float)bound);
        }
    }
    return result;
}

static PropagatedComparison CompareRMSScalePropagation(
    const float *actual, const float *expected, const float *hidden,
    const float *actualDNormalized, const float *expectedDNormalized
) {
    PropagatedComparison result = {0};
    result.comparison.elements = DIM;
    for (int channel = 0; channel < DIM; ++channel) {
        double bound = 0.0001;
        for (int position = 0; position < SEQUENCE; ++position) {
            double squareSum = 0.0;
            for (int inner = 0; inner < DIM; ++inner) {
                const float value = hidden[position * DIM + inner];
                squareSum += (double)value * value;
            }
            const float inverse =
                1.0f / sqrtf((float)(squareSum / DIM) + 0.00001f);
            const size_t index = (size_t)position * DIM + channel;
            bound += fabs(
                (double)(actualDNormalized[index] - expectedDNormalized[index])
                * hidden[index] * inverse);
        }
        const float delta = fabsf(actual[channel] - expected[channel]);
        if (delta > bound) ++result.comparison.mismatches;
        result.comparison.maximum = fmaxf(result.comparison.maximum, delta);
        result.maximumAllowed = fmaxf(result.maximumAllowed, (float)bound);
    }
    return result;
}

static PropagatedComparison CompareRMSInputPropagation(
    const float *actual, const float *expected, const float *hidden,
    const float *scale, const float *actualDNormalized,
    const float *expectedDNormalized
) {
    PropagatedComparison result = {0};
    result.comparison.elements = (size_t)SEQUENCE * DIM;
    for (int position = 0; position < SEQUENCE; ++position) {
        double squareSum = 0.0;
        double weightedDelta = 0.0;
        for (int channel = 0; channel < DIM; ++channel) {
            const size_t index = (size_t)position * DIM + channel;
            squareSum += (double)hidden[index] * hidden[index];
            weightedDelta += fabs(
                (double)(actualDNormalized[index]
                         - expectedDNormalized[index])
                * scale[channel] * hidden[index]);
        }
        const float inverse =
            1.0f / sqrtf((float)(squareSum / DIM) + 0.00001f);
        for (int channel = 0; channel < DIM; ++channel) {
            const size_t index = (size_t)position * DIM + channel;
            const double local = fabs(
                (double)inverse * scale[channel]
                * (actualDNormalized[index] - expectedDNormalized[index]));
            const double correction =
                fabs((double)hidden[index] * inverse * inverse * inverse
                     * weightedDelta / DIM);
            const double bound = 0.0001 + local + correction;
            const float delta = fabsf(actual[index] - expected[index]);
            if (delta > bound) ++result.comparison.mismatches;
            result.comparison.maximum =
                fmaxf(result.comparison.maximum, delta);
            result.maximumAllowed =
                fmaxf(result.maximumAllowed, (float)bound);
        }
    }
    return result;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
        setenv("BLAS_NUM_THREADS", "1", 1);
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
        float *hidden = malloc(SEQUENCE * DIM * sizeof(float));
        float *normScale = malloc(DIM * sizeof(float));
        float *normalized = malloc(SEQUENCE * DIM * sizeof(float));
        float *wq = malloc(DIM * QUERY_DIM * sizeof(float));
        float *wk = malloc(DIM * KV_DIM * sizeof(float));
        float *wv = malloc(DIM * KV_DIM * sizeof(float));
        float *query = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *compactKey = malloc(SEQUENCE * KV_DIM * sizeof(float));
        float *compactValue = malloc(SEQUENCE * KV_DIM * sizeof(float));
        float *key = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *value = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *upstream = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        BuildCoherentInputs(
            hidden, normScale, normalized, wq, wk, wv,
            query, compactKey, compactValue, upstream);
        TileContiguousKV(key, compactKey);
        TileContiguousKV(value, compactValue);
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
        float *actualDQ = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *actualDK = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        float *actualDV = malloc(SEQUENCE * QUERY_DIM * sizeof(float));
        ReadChannelMajorToRows(actualDQ, actual, QUERY_DIM);
        ReadChannelMajorToRows(
            actualDK, actual + (size_t)QUERY_DIM * SEQUENCE, QUERY_DIM);
        ReadChannelMajorToRows(
            actualDV, actual + (size_t)2 * QUERY_DIM * SEQUENCE, QUERY_DIM);
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
        float *cosine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
        float *sine = malloc(SEQUENCE * HEAD_DIM * sizeof(float));
        BuildRoPE(cosine, sine);
        float *reducedDK = malloc(SEQUENCE * KV_DIM * sizeof(float));
        float *reducedDV = malloc(SEQUENCE * KV_DIM * sizeof(float));
        float *expectedReducedDK = malloc(SEQUENCE * KV_DIM * sizeof(float));
        float *expectedReducedDV = malloc(SEQUENCE * KV_DIM * sizeof(float));
        ReduceContiguousKV(reducedDK, actualDK);
        ReduceContiguousKV(reducedDV, actualDV);
        ReduceContiguousKV(expectedReducedDK, dk);
        ReduceContiguousKV(expectedReducedDV, dv);
        InverseSplitHalfRoPEGradient(actualDQ, QUERY_HEADS, cosine, sine);
        InverseSplitHalfRoPEGradient(reducedDK, KV_HEADS, cosine, sine);
        InverseSplitHalfRoPEGradient(dq, QUERY_HEADS, cosine, sine);
        InverseSplitHalfRoPEGradient(expectedReducedDK, KV_HEADS, cosine, sine);
        Comparison inverseDQComparison = {0};
        inverseDQComparison.elements = (size_t)SEQUENCE * QUERY_DIM;
        for (size_t index = 0; index < inverseDQComparison.elements; ++index) {
            const float delta = fabsf(actualDQ[index] - dq[index]);
            if (delta > 0.002f) ++inverseDQComparison.mismatches;
            inverseDQComparison.maximum =
                fmaxf(inverseDQComparison.maximum, delta);
        }
        Comparison inverseDKComparison = {0};
        inverseDKComparison.elements = (size_t)SEQUENCE * KV_DIM;
        Comparison reducedDVComparison = {0};
        reducedDVComparison.elements = (size_t)SEQUENCE * KV_DIM;
        for (size_t index = 0; index < inverseDKComparison.elements; ++index) {
            const float dkDelta =
                fabsf(reducedDK[index] - expectedReducedDK[index]);
            if (dkDelta > 0.004f) ++inverseDKComparison.mismatches;
            inverseDKComparison.maximum =
                fmaxf(inverseDKComparison.maximum, dkDelta);
            const float dvDelta =
                fabsf(reducedDV[index] - expectedReducedDV[index]);
            if (dvDelta > 0.004f) ++reducedDVComparison.mismatches;
            reducedDVComparison.maximum =
                fmaxf(reducedDVComparison.maximum, dvDelta);
        }
        float *actualDWQ = malloc(DIM * QUERY_DIM * sizeof(float));
        float *actualDWK = malloc(DIM * KV_DIM * sizeof(float));
        float *actualDWV = malloc(DIM * KV_DIM * sizeof(float));
        float *actualDNormalized =
            malloc(SEQUENCE * DIM * sizeof(float));
        float *expectedDWQ = malloc(DIM * QUERY_DIM * sizeof(float));
        float *expectedDWK = malloc(DIM * KV_DIM * sizeof(float));
        float *expectedDWV = malloc(DIM * KV_DIM * sizeof(float));
        float *expectedDNormalized =
            malloc(SEQUENCE * DIM * sizeof(float));
        float *operatorDWQ = malloc(DIM * QUERY_DIM * sizeof(float));
        float *operatorDWK = malloc(DIM * KV_DIM * sizeof(float));
        float *operatorDWV = malloc(DIM * KV_DIM * sizeof(float));
        float *operatorDNormalized =
            malloc(SEQUENCE * DIM * sizeof(float));
        const int cpuRepetitions = 4;
        const uint64_t cpuStarted = mach_absolute_time();
        for (int repetition = 0; repetition < cpuRepetitions; ++repetition) {
            ProjectionGradientsAccelerate(
                normalized, wq, wk, wv,
                actualDQ, reducedDK, reducedDV,
                actualDWQ, actualDWK, actualDWV, actualDNormalized);
        }
        const double meanCPUProjectionMilliseconds =
            Milliseconds(mach_absolute_time() - cpuStarted) / cpuRepetitions;
        ProjectionGradientsScalar(
            normalized, wq, wk, wv,
            dq, expectedReducedDK, expectedReducedDV,
            expectedDWQ, expectedDWK, expectedDWV, expectedDNormalized);
        ProjectionGradientsAccelerate(
            normalized, wq, wk, wv,
            dq, expectedReducedDK, expectedReducedDV,
            operatorDWQ, operatorDWK, operatorDWV, operatorDNormalized);
        Comparison projectionOperatorComparison = {0};
        Comparison operatorSlices[] = {
            CompareFloatArrays(
                operatorDWQ, expectedDWQ,
                (size_t)DIM * QUERY_DIM, 0.0001f),
            CompareFloatArrays(
                operatorDWK, expectedDWK,
                (size_t)DIM * KV_DIM, 0.0001f),
            CompareFloatArrays(
                operatorDWV, expectedDWV,
                (size_t)DIM * KV_DIM, 0.0001f),
            CompareFloatArrays(
                operatorDNormalized, expectedDNormalized,
                (size_t)SEQUENCE * DIM, 0.0001f),
        };
        for (size_t index = 0;
             index < sizeof(operatorSlices) / sizeof(operatorSlices[0]);
             ++index) {
            projectionOperatorComparison.mismatches +=
                operatorSlices[index].mismatches;
            projectionOperatorComparison.elements +=
                operatorSlices[index].elements;
            projectionOperatorComparison.maximum = fmaxf(
                projectionOperatorComparison.maximum,
                operatorSlices[index].maximum);
        }
        float *actualDHidden = malloc(SEQUENCE * DIM * sizeof(float));
        float *actualDScale = malloc(DIM * sizeof(float));
        float *expectedDHidden = malloc(SEQUENCE * DIM * sizeof(float));
        float *expectedDScale = malloc(DIM * sizeof(float));
        RMSNormBackwardReference(
            hidden, normScale, actualDNormalized,
            actualDHidden, actualDScale);
        RMSNormBackwardReference(
            hidden, normScale, operatorDNormalized,
            expectedDHidden, expectedDScale);
        PropagatedComparison dwqComparison =
            CompareWeightGradientPropagation(
                actualDWQ, operatorDWQ, normalized, actualDQ, dq, QUERY_DIM);
        PropagatedComparison dwkComparison =
            CompareWeightGradientPropagation(
                actualDWK, operatorDWK, normalized,
                reducedDK, expectedReducedDK, KV_DIM);
        PropagatedComparison dwvComparison =
            CompareWeightGradientPropagation(
                actualDWV, operatorDWV, normalized,
                reducedDV, expectedReducedDV, KV_DIM);
        PropagatedComparison dnormalizedComparison =
            CompareProjectionInputPropagation(
                actualDNormalized, operatorDNormalized, wq, wk, wv,
                actualDQ, dq, reducedDK, expectedReducedDK,
                reducedDV, expectedReducedDV);
        PropagatedComparison dhiddenComparison =
            CompareRMSInputPropagation(
                actualDHidden, expectedDHidden, hidden, normScale,
                actualDNormalized, operatorDNormalized);
        PropagatedComparison dscaleComparison =
            CompareRMSScalePropagation(
                actualDScale, expectedDScale, hidden,
                actualDNormalized, operatorDNormalized);
        const size_t mismatches =
            dqComparison.mismatches + dkComparison.mismatches
            + dvComparison.mismatches + inverseDQComparison.mismatches
            + inverseDKComparison.mismatches + reducedDVComparison.mismatches
            + projectionOperatorComparison.mismatches
            + dwqComparison.comparison.mismatches
            + dwkComparison.comparison.mismatches
            + dwvComparison.comparison.mismatches
            + dnormalizedComparison.comparison.mismatches
            + dhiddenComparison.comparison.mismatches
            + dscaleComparison.comparison.mismatches;
        printf(
            "{\"policy\":\"project_theseus_exact_ane_attention_backward_v1\","
            "\"shape\":{\"sequence\":%d,\"query_heads\":%d,"
            "\"kv_heads\":%d,\"head_dim\":%d},"
            "\"parameter_generation\":0,"
            "\"compile_milliseconds\":%.6f,"
            "\"mean_evaluation_milliseconds\":%.6f,"
            "\"mean_cpu_projection_gradient_milliseconds\":%.6f,"
            "\"comparisons\":{"
            "\"dq_rope\":{\"tolerance\":0.002,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"dk_tiled_rope\":{\"tolerance\":0.002,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"dv_tiled\":{\"tolerance\":0.002,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"dq_inverse_split_half_rope\":{\"tolerance\":0.002,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"dk_contiguous_reduce_inverse_split_half_rope\":{"
            "\"tolerance\":0.004,\"maximum_absolute_delta\":%.9g,"
            "\"mismatch_count\":%zu},"
            "\"dv_contiguous_reduce\":{\"tolerance\":0.004,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"accelerate_projection_operator\":{\"tolerance\":0.0001,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"q_proj_weight_gradient\":{\"tolerance_policy\":"
            "\"analytical_fp16_boundary_propagation\","
            "\"maximum_allowed_delta\":%.9g,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"k_proj_weight_gradient\":{\"tolerance_policy\":"
            "\"analytical_fp16_boundary_propagation\","
            "\"maximum_allowed_delta\":%.9g,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"v_proj_weight_gradient\":{\"tolerance_policy\":"
            "\"analytical_fp16_boundary_propagation\","
            "\"maximum_allowed_delta\":%.9g,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"projection_input_gradient\":{\"tolerance_policy\":"
            "\"analytical_fp16_boundary_propagation\","
            "\"maximum_allowed_delta\":%.9g,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"attention_norm_input_gradient\":{\"tolerance_policy\":"
            "\"analytical_fp16_boundary_propagation\","
            "\"maximum_allowed_delta\":%.9g,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu},"
            "\"attention_norm_scale_gradient\":{\"tolerance_policy\":"
            "\"analytical_fp16_boundary_propagation\","
            "\"maximum_allowed_delta\":%.9g,"
            "\"maximum_absolute_delta\":%.9g,\"mismatch_count\":%zu}},"
            "\"gates\":{\"causal_softmax_backward\":true,"
            "\"full_query_head_gradients\":true,"
            "\"kv_reduction\":true,\"inverse_split_half_rope\":true,"
            "\"input_gradient\":true,\"parameter_gradients\":true,"
            "\"fp32_gradient_accumulation\":true,"
            "\"single_thread_accelerate\":true,"
            "\"complete_decoder_block\":false,"
            "\"production_eligible\":false},"
            "\"mismatch_count\":%zu,\"trigger_state\":\"%s\","
            "\"capability_claim\":\"NONE_ENGINEERING_ATTENTION_GRADIENT_ONLY\"}\n",
            SEQUENCE, QUERY_HEADS, KV_HEADS, HEAD_DIM,
            compileMilliseconds, meanMilliseconds,
            meanCPUProjectionMilliseconds,
            dqComparison.maximum, dqComparison.mismatches,
            dkComparison.maximum, dkComparison.mismatches,
            dvComparison.maximum, dvComparison.mismatches,
            inverseDQComparison.maximum, inverseDQComparison.mismatches,
            inverseDKComparison.maximum, inverseDKComparison.mismatches,
            reducedDVComparison.maximum, reducedDVComparison.mismatches,
            projectionOperatorComparison.maximum,
            projectionOperatorComparison.mismatches,
            dwqComparison.maximumAllowed,
            dwqComparison.comparison.maximum,
            dwqComparison.comparison.mismatches,
            dwkComparison.maximumAllowed,
            dwkComparison.comparison.maximum,
            dwkComparison.comparison.mismatches,
            dwvComparison.maximumAllowed,
            dwvComparison.comparison.maximum,
            dwvComparison.comparison.mismatches,
            dnormalizedComparison.maximumAllowed,
            dnormalizedComparison.comparison.maximum,
            dnormalizedComparison.comparison.mismatches,
            dhiddenComparison.maximumAllowed,
            dhiddenComparison.comparison.maximum,
            dhiddenComparison.comparison.mismatches,
            dscaleComparison.maximumAllowed,
            dscaleComparison.comparison.maximum,
            dscaleComparison.comparison.mismatches,
            mismatches, mismatches == 0 ? "GREEN" : "RED");
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &error);
        [[NSFileManager defaultManager]
            removeItemAtPath:temporaryDirectory error:nil];
        CFRelease(input);
        CFRelease(output);
        free(mask);
        free(hidden);
        free(normScale);
        free(normalized);
        free(wq);
        free(wk);
        free(wv);
        free(query);
        free(compactKey);
        free(compactValue);
        free(key);
        free(value);
        free(upstream);
        free(dq);
        free(dk);
        free(dv);
        free(actualDQ);
        free(actualDK);
        free(actualDV);
        free(cosine);
        free(sine);
        free(reducedDK);
        free(reducedDV);
        free(expectedReducedDK);
        free(expectedReducedDV);
        free(actualDWQ);
        free(actualDWK);
        free(actualDWV);
        free(actualDNormalized);
        free(expectedDWQ);
        free(expectedDWK);
        free(expectedDWV);
        free(expectedDNormalized);
        free(operatorDWQ);
        free(operatorDWK);
        free(operatorDWV);
        free(operatorDNormalized);
        free(actualDHidden);
        free(actualDScale);
        free(expectedDHidden);
        free(expectedDScale);
        return mismatches == 0 ? 0 : 1;
    }
}
