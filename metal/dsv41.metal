// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

/* Shared and noinline so the attention's fused q RoPE compiles to the same
 * arithmetic as this kernel's. */
static __attribute__((noinline)) float2 dsv41_rope_pair(float re, float im, float theta,
                                                        uint inverse) {
    const float c = precise::cos(theta);
    const float s = inverse ? -precise::sin(theta) : precise::sin(theta);
    return float2(dsv41_bf16(re * c - im * s), dsv41_bf16(re * s + im * c));
}

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float2 y = dsv41_rope_pair(x[i], x[i + 1u], theta, args.inverse);
    x[i] = y.x;
    x[i + 1u] = y.y;
}

// Decode attention reduce followed by the BF16 store and the RoPE of the one
// head row each threadgroup owns: the same arithmetic as the plain reduce,
// kernel_dsv41_bf16_linear and kernel_dsv41_rope in turn.
kernel void kernel_dsv41_flash_attn_reduce_rope(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        constant ds4_metal_args_dsv41_rope & rope,
        constant uint & splits,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    ds4_flash_attn_vec_reduce_row(args, htmp, dst, tgpig, tiisg, sgitg,
                                  (short)FC_flash_attn_ext_vec_reduce_NWG,
                                  (short)FC_flash_attn_ext_vec_reduce_DV, (short)splits);
    threadgroup_barrier(mem_flags::mem_device);
    device float *x = (device float *)dst + (ulong)tgpig * rope.width;
    const uint tail = rope.width - 64u;
    for (uint i = tiitg; i < tail; i += 32u * FC_flash_attn_ext_vec_reduce_NWG)
        x[i] = dsv41_bf16(x[i]);
    if (sgitg != 0) return;
    const float theta = float(rope.start) * rope.frequencies[tiisg];
    const float c = precise::cos(theta);
    const float s = rope.inverse ? -precise::sin(theta) : precise::sin(theta);
    const uint i = tail + 2u * tiisg;
    const float re = dsv41_bf16(x[i]), im = dsv41_bf16(x[i + 1u]);
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

struct ds4_metal_args_dsv41_flash_attn {
    uint n_raw;
    uint raw_cap;
    uint raw_start;
    uint n_comp;
    uint n_head;
    float scale;
};

// Decode FlashAttention over the raw window ring and the selected compressed
// rows, gathered in place: the arithmetic of kernel_flash_attn_ext_vec_f16_dk512_dv512
// (one simdgroup per head and split, 32 splits of 32-key chunks, zero mask)
// without its F16 staging copies. Eight heads share each chunk staged in
// threadgroup memory; the partials feed kernel_dsv41_flash_attn_reduce_rope.
// Only the splits that hold keys run.
// It also rounds q to BF16 and applies kernel_dsv41_rope to it as it loads
// it, and stores the new KV row in its window slot while reading that row
// from kv.
kernel void kernel_dsv41_flash_attn_decode(
        constant ds4_metal_args_dsv41_flash_attn &args,
        device const float *q,
        device const float *kv,
        device float *raw,
        device const float *comp,
        device const int *ids,
        device const float *sinks,
        device float *dst,
        constant ds4_metal_args_dsv41_rope &rope,
        threadgroup half4 *kv4 [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort nsg [[simdgroups_per_threadgroup]]) {
    constexpr short C = 32, NWG = 32, DV4 = 128;
    const uint head = tgpig.x * nsg + sgitg;
    const short iwg = tgpig.y;
    const uint n_keys = args.n_raw + args.n_comp;

    const uint last = (args.raw_start + args.n_raw - 1u) % args.raw_cap;
    if (tgpig.x == 0 && tgpig.y == 0 && tiitg < DV4)
        ((device float4 *)raw)[last * DV4 + tiitg] = ((device const float4 *)kv)[tiitg];

    device const float4 *q4 = (device const float4 *)(q + (ulong)head * 512u);
    half4 sq4[4];
    float4 so4[4];
    FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
        float4 x = dsv41_bf16(q4[ii * 32 + tiisg]);
        if (ii == 3 && tiisg >= 16) {
            FOR_UNROLL (short p = 0; p < 2; ++p) {
                const float theta = float(rope.start) * rope.frequencies[2 * (tiisg - 16) + p];
                const float2 y = dsv41_rope_pair(x[2 * p], x[2 * p + 1], theta, rope.inverse);
                x[2 * p] = y.x;
                x[2 * p + 1] = y.y;
            }
        }
        sq4[ii] = (half4)x;
        so4[ii] = 0.0f;
    }

    float S = 0.0f;
    float M = -FLT_MAX/2;

    for (uint ic = iwg * C; ic < n_keys; ic += NWG * C) {
        /* Stage the chunk's 32 rows in F16, as the F16 KV copy did. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            /* Eight threads per row, all loads in flight at once. */
            const uint r = tiitg / 8u, part = tiitg % 8u, key = ic + r;
            device const float4 *src = nullptr;
            if (key + 1u == args.n_raw) {
                src = (device const float4 *)kv;
            } else if (key < args.n_raw) {
                src = (device const float4 *)(raw + (ulong)((args.raw_start + key) % args.raw_cap) * 512u);
            } else if (key < n_keys) {
                src = (device const float4 *)(comp + (ulong)ids[key - args.n_raw] * 512u);
            }
            float4 x[DV4 / 8];
            FOR_UNROLL (short k = 0; k < DV4 / 8; ++k) {
                x[k] = src ? src[part + 8 * k] : 0.0f;
            }
            FOR_UNROLL (short k = 0; k < DV4 / 8; ++k) {
                kv4[r * DV4 + part + 8 * k] = (half4)x[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const half sm = ic + tiisg < n_keys ? 0.0h : -MAXHALF;

        /* Fully unrolled, the two key loops cost several times more. */
        float s = 0.0f;
#pragma clang loop unroll_count(4)
        for (short cc = 0; cc < C; ++cc) {
            float mqk = 0.0f;
            FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                mqk += dot((float4)kv4[cc * DV4 + ii * 32 + tiisg], (float4)sq4[ii]);
            }
            mqk = simd_sum(mqk);
            if (cc == tiisg) s = fma(mqk, args.scale, (float)sm);
        }

        {
            const float m = M;

            M = simd_max(max(M, s));

            const float ms = exp(m - M);
            const float vs = exp(s - M);

            S = S*ms + simd_sum(vs);

            s = vs;

            FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                so4[ii] *= ms;
            }
        }

        {
            float4 lo[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma clang loop unroll_count(4)
            for (short cc = 0; cc < C; ++cc) {
                const float w = simd_shuffle(s, cc);
                FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                    lo[ii] += float4(float4(kv4[cc * DV4 + ii * 32 + tiisg])*float4(w));
                }
            }
            FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                so4[ii] += lo[ii];
            }
        }
    }

    if (iwg == 0) {
        const float m = M;
        const float s = tiisg == 0 ? sinks[head] : -FLT_MAX/2;

        M = simd_max(max(M, s));

        const float ms = exp(m - M);
        const float vs = exp(s - M);

        S = S*ms + simd_sum(vs);

        FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
            so4[ii] *= ms;
        }
    }

    device float4 *dst4 = (device float4 *)dst;
    device float *dst1 = dst + (ulong)args.n_head * 512u * NWG;
    FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
        dst4[((ulong)head * NWG + iwg) * DV4 + ii * 32 + tiisg] = so4[ii];
    }
    if (tiisg == 0) {
        dst1[head * (2 * NWG) + 2 * iwg + 0] = S;
        dst1[head * (2 * NWG) + 2 * iwg + 1] = M;
    }
}


/* One lane of a quantization block: the block's simdgroup lanes hold the
 * BF16-rounded values (zero past the block). */
static inline float dsv41_quantize_lane(float value, uint mode) {
    const float amax = simd_max(abs(value));
    float result = value;
    if (mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (mode == 2u || mode == 3u) {
        const float scale = mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    return dsv41_bf16(result);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float result = dsv41_quantize_lane(valid ? dsv41_bf16(x[index]) : 0.0f, args.mode);
    if (valid) x[index] = result;
}

// kernel_dsv41_bf16_linear, kernel_dsv41_rope and kernel_dsv41_quantize
// (32-wide blocks) in turn on one head row per threadgroup of width threads:
// the RoPE tail pairs adjacent lanes.
kernel void kernel_dsv41_rope_quantize(
        constant ds4_metal_args_dsv41_rope &args,
        constant uint &mode,
        device float *x,
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const ulong i = (ulong)head * args.width + tid;
    float value = dsv41_bf16(x[i]);
    const float other = simd_shuffle_xor(value, 1u);
    const uint tail = args.width - 64u;
    if (tid >= tail) {
        const float theta = float(args.start) * args.frequencies[(tid - tail) / 2u];
        value = tid & 1u ? dsv41_rope_pair(other, value, theta, args.inverse).y
                         : dsv41_rope_pair(value, other, theta, args.inverse).x;
    }
    x[i] = dsv41_quantize_lane(dsv41_bf16(value), mode);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

#define DSV41_ENGRAM_CHUNK 2560u

/* One threadgroup per (row, HC): all threads stage a chunk of the inputs in
 * threadgroup memory, and the first simdgroup sums them per lane in column
 * order (as a lone simdgroup reading device memory would, but without
 * waiting on each load). */
kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        threadgroup float *shared [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint2 nt2 [[threads_per_threadgroup]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const uint nt = nt2.x;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    device const float *q = q_weight + group.y * args.width;
    device const float *kw = k_weight + group.y * args.width;
    threadgroup float *sh = shared;
    threadgroup float *sk = shared + DSV41_ENGRAM_CHUNK;
    threadgroup float *sw = shared + 2u * DSV41_ENGRAM_CHUNK;
    threadgroup float *sgate = shared + 3u * DSV41_ENGRAM_CHUNK;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint c = 0; c < args.width; c += DSV41_ENGRAM_CHUNK) {
        const uint n = min(DSV41_ENGRAM_CHUNK, args.width - c);
        for (uint i = tid; i < n; i += nt) {
            sh[i] = residual[offset + c + i];
            sk[i] = dsv41_bf16(kv[key_offset + c + i]);
            sw[i] = q[c + i] * kw[c + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            for (uint i = lane; i < n; i += 32u) {
                const float h = sh[i];
                const float k = sk[i];
                h2 += h * h;
                k2 += k * k;
                dot += h * sw[i] * k;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32u) {
        h2 = simd_sum(h2);
        k2 = simd_sum(k2);
        dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
              rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
        if (tid == 0)
            sgate[0] = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = sgate[0];
    for (uint i = tid; i < args.width; i += nt)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

// V4.1 decode router for one token over 384 experts: sqrt(softplus(logit)),
// the biased top-6 and the normalized expert weights in one dispatch of 512
// threads. Each step repeats the unary, bin, argsort, get_rows and sum_rows
// kernels of the generic path, so the selection and weights stay identical.
// Each SIMD group keeps its 7 best scores by SIMD max rounds, and every SIMD
// group merges those 84 candidates: if the top 7 strictly decrease, the
// first 6 are what the descending sort puts first. Otherwise (ties or NaN)
// the threads run the bitonic sort, one sort element each; compare-exchanges
// within a SIMD group use shuffles, the wider ones go through double-buffered
// threadgroup memory.
kernel void kernel_dsv41_router_select_one(
        constant float &scale,
        device const float *logits,
        device float *probs,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int n = 384;
    constexpr int K = 7;
    /* One threadgroup per token row. */
    logits += row*n;
    probs += row*n;
    selected += row*6;
    weights += row*6;
    threadgroup float *prob = scratch;
    threadgroup float *score_tg[2] = {scratch + 512, scratch + 1024};
    threadgroup int32_t *idx_tg[2] = {(threadgroup int32_t *)(scratch + 1536),
                                      (threadgroup int32_t *)(scratch + 2048)};
    /* The candidates use the second sort buffers, free until the sort. */
    threadgroup float *cand_score = score_tg[1];
    threadgroup int32_t *cand_idx = idx_tg[1];
    threadgroup int32_t *unordered = idx_tg[1] + 511;
    if (tid == 0) *unordered = 0;
    if (tid < n / 4) {
        const float4 x = ((device const float4 *)logits)[tid];
        const float4 ex = exp(x);
        const float4 em = min(ex, 0.03125f);
        const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float4 p = sqrt(select(select(log(1 + ex), poly, ex < 0.03125f), x, x > 20));
        ((device float4 *)probs)[tid] = p;
        ((threadgroup float4 *)prob)[tid] = p;
        ((threadgroup float4 *)score_tg[0])[tid] = p + ((device const float4 *)bias)[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float score = score_tg[0][tid];
    if (tid < n) {
        if (simd_any(isnan(score))) *unordered = 1;
        bool taken = false;
        for (int r = 0; r < K; ++r) {
            const float v = taken ? -INFINITY : score;
            const float m = simd_max(v);
            if (tiisg == simd_min(!taken && v == m ? tiisg : (ushort)32)) {
                cand_score[sgitg*K + r] = m;
                cand_idx[sgitg*K + r] = (int32_t)tid;
                taken = true;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float c[3];
    int32_t ci[3];
    bool taken[3];
    for (int q = 0; q < 3; ++q) {
        const int e = tiisg + 32*q;
        taken[q] = e >= (n/32)*K;
        c[q] = taken[q] ? -INFINITY : cand_score[e];
        ci[q] = taken[q] ? 0 : cand_idx[e];
    }
    int32_t idx = 0;
    bool ordered = *unordered == 0;
    float prev = INFINITY;
    for (int r = 0; r < K; ++r) {
        int bq = -1;
        for (int q = 0; q < 3; ++q)
            if (!taken[q] && (bq < 0 || c[q] > c[bq])) bq = q;
        const float best = bq < 0 ? -INFINITY : c[bq];
        const float m = simd_max(best);
        const ushort w = simd_min(bq >= 0 && best == m ? tiisg : (ushort)32);
        const int32_t widx = simd_shuffle(bq < 0 ? 0 : ci[bq], w);
        if (tiisg == w) taken[bq] = true;
        if (tiisg == r) idx = widx;
        ordered = ordered && m < prev;
        prev = m;
    }
    if (!ordered) {
        idx = (int32_t)tid;
        uint stage = 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 2; k <= 512; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                float peer_score;
                int32_t peer_idx;
                if (j < 32) {
                    peer_score = simd_shuffle_xor(score, (ushort)j);
                    peer_idx = simd_shuffle_xor(idx, (ushort)j);
                } else {
                    score_tg[stage & 1][tid] = score;
                    idx_tg[stage & 1][tid] = idx;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    peer_score = score_tg[stage & 1][tid ^ j];
                    peer_idx = idx_tg[stage & 1][tid ^ j];
                    stage++;
                }
                const bool lower = (tid & j) == 0;
                const int a = lower ? idx : peer_idx, b = lower ? peer_idx : idx;
                const float sa = lower ? score : peer_score, sb = lower ? peer_score : score;
                const bool swap = (tid & k) == 0 ?
                    a >= n || (b < n && sa < sb) :
                    b >= n || (a < n && sa > sb);
                if (swap) {
                    score = peer_score;
                    idx = peer_idx;
                }
            }
        }
    }
    if (sgitg == 0) {
        const float w = tiisg < 6 ? prob[idx] : 0.0f;
        const float sum = max(simd_sum(w), 6.103515625e-5f);
        if (tiisg < 6) {
            selected[tiisg] = idx;
            /* Round the quotient before scaling, as the separate div and
             * scale kernels do. */
            weights[tiisg] = w / sum;
            weights[tiisg] = ((device volatile float *)weights)[tiisg] * scale;
        }
    }
}

// V4.1 Engram KV projection from the load-time FP8 repack of its F16 weight:
// e4m3 bytes, row-major, and one power-of-two float scale per 32x32 block.
// Each weight decodes to exactly its F16 value; the K walk and reduction
// tree replicate kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2), then BF16 rounding.
// A threadgroup decodes its weights once for up to four input rows (dims.z
// rows in all), each with that same walk.
template<short NR1>
void kernel_dsv41_engram_kv_fp8_impl(
        constant uint3 &dims,
        device const uchar * src0,
        device const float * scales,
        device const float4 * y4,
        device       float * dst,
        threadgroup  char  * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NSG = 8;
    constexpr short NR0 = 2;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = dims.x/NB;
    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y*4;

    device const uint4 * ax[NR0];
    device const float * as[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax[row] = (device const uint4 *) (src0 + (uint64_t)(r0 + row)*dims.x);
        as[row] = scales + (uint64_t)((r0 + row)/32)*nb;
    }

    float sumf[NR1][NR0] = { { 0.f } };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;

    device const float4 * yb4 = y4 + ((uint64_t)r1*dims.x + ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 xs[NR0][NF4];
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const uint4 q = ax[row][(ib*NB + il*NF)/16];
            const float s = as[row][ib];
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                /* e4m3 into the top bits of a half: the value times 2^-8. */
                const ushort4 b = ushort4(as_type<uchar4>(q[i]));
                xs[row][i] = float4(as_type<half4>(((b & 0x7fu) << 7) | ((b & 0x80u) << 8)))*s;
            }
        }

        FOR_UNROLL (short j = 0; j < NR1; ++j) {
            float4 yl4[NF4];
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                yl4[i] = yb4[j*dims.x/4 + i];
            }

            FOR_UNROLL (short row = 0; row < NR0; ++row) {
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NF4; ++i) {
                    sumq += dot(xs[row][i], yl4[i]);
                }

                sumf[j][row] += sumq;
            }
        }

        yb4 += NSG*NF*NW/4;
    }

    FOR_UNROLL (short j = 0; j < NR1; ++j) {
        helper_mv_reduce_and_write<NR0, true>(dst + (uint64_t)(r1 + j)*dims.y, sumf[j],
                                              r0, dims.y, tiisg, sgitg, shmem);
        if (j + 1 < NR1) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void kernel_dsv41_engram_kv_fp8(
        constant uint3 &dims,
        device const uchar * src0,
        device const float * scales,
        device const float4 * y4,
        device       float * dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    switch (min(4, (int)dims.z - (int)tgpig.y*4)) {
        case 4: kernel_dsv41_engram_kv_fp8_impl<4>(dims, src0, scales, y4, dst, shmem, tgpig, tiisg, sgitg); break;
        case 3: kernel_dsv41_engram_kv_fp8_impl<3>(dims, src0, scales, y4, dst, shmem, tgpig, tiisg, sgitg); break;
        case 2: kernel_dsv41_engram_kv_fp8_impl<2>(dims, src0, scales, y4, dst, shmem, tgpig, tiisg, sgitg); break;
        default: kernel_dsv41_engram_kv_fp8_impl<1>(dims, src0, scales, y4, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}
