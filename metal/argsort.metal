struct ds4_metal_args_argsort {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    uint32_t causal_start;
    uint32_t causal_ratio;
};

struct ds4_metal_args_argsort_merge {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
    uint32_t causal_start;
    uint32_t causal_ratio;
    uint32_t block_width;
    uint32_t block_top_k;
};

typedef void (argsort_t)(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Sort one float row into an index row. DS4 only exports the descending
// instance because router and indexer selection both need top-k order.
template<ds4_sort_order order, bool causal = false, bool shuffle = false>
kernel void kernel_argsort_f32_i32(
        constant   ds4_metal_args_argsort & args,
        device   const char * src0,
        device      int32_t * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    // bitonic sort
    const int col = tpitg[0];
    const int ib  = tgpig[0] / args.ne01;

    const int i00 = ib*ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];
    const int width = causal ? min(args.ne00,
        int((args.causal_start + uint(i01) + 1u) / args.causal_ratio)) : args.ne00;
    if (i00 >= width) return;
    const int work_width = causal ?
        ((width - 1) / ntg.x) * args.top_k + min((width - 1) % ntg.x + 1, args.top_k) : args.ne0;

    device const float * src0_row = (device const float *) (src0 + args.nb01*i01 + args.nb02*i02 + args.nb03*i03);

    // initialize indices
    shmem_i32[col] = i00 + col;

    // Stage this block's score slice in threadgroup memory (indices stay in
    // [i00, i00+ntg.x), so shmem_f32[idx - i00] replaces the device gather).
    // The host allocates ntg.x extra floats after the index array.  Values and
    // the comparison network are unchanged, so the permutation is identical.
    threadgroup float * shmem_f32 = (threadgroup float *) (shmem_i32 + ntg.x);
    if (i00 + col < width) {
        shmem_f32[col] = src0_row[i00 + col];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int reg_idx = i00 + col;
    float reg_value = reg_idx < width ? shmem_f32[col] : 0.0f;
    for (int k = 2; k <= ntg.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            if (shuffle && j < 32) {
                if (k > 32 && j == 16) {
                    reg_idx = shmem_i32[col];
                    reg_value = reg_idx < width ? shmem_f32[reg_idx - i00] : 0.0f;
                }
                const int other = simd_shuffle_xor(reg_idx, j);
                const float value = simd_shuffle_xor(reg_value, j);
                const bool first = ((col & k) == 0) == ((col & j) == 0);
                const bool exchange = first ?
                    (reg_idx >= width || (other < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value > value : reg_value < value))) :
                    (other >= width || (reg_idx < width && (order == DS4_SORT_ORDER_ASC ?
                        reg_value < value : reg_value > value)));
                if (exchange) { reg_idx = other; reg_value = value; }
                continue;
            }
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                       (shmem_i32[ixj] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                       (shmem_i32[col] <  width && (order == DS4_SORT_ORDER_ASC ?
                            shmem_f32[shmem_i32[col] - i00] < shmem_f32[shmem_i32[ixj] - i00] :
                            shmem_f32[shmem_i32[col] - i00] > shmem_f32[shmem_i32[ixj] - i00]))
                    ) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (shuffle && k >= 32) {
            shmem_i32[col] = reg_idx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib*args.top_k;

    // copy the result to dst without the padding
    if (i0 + col < work_width && col < args.top_k) {
        dst += i0 + args.ne0*i01 + args.ne0*args.ne1*i02 + args.ne0*args.ne1*args.ne2*i03;

        dst[col] = shuffle ? reg_idx : shmem_i32[col];
    }
}

// Host-visible sort variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_f32_i32_desc_causal")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_f32_i32_desc_causal_shuffle")]] kernel argsort_t kernel_argsort_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

typedef void (argsort_merge_t)(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]);

// Merges sorted index runs produced by kernel_argsort_f32_i32. In the DS4 graph
// this finishes top-k over router or compressed-attention score rows.
template<ds4_sort_order order, bool causal = false, bool prefix = false>
kernel void kernel_argsort_merge_f32_i32(
        constant   ds4_metal_args_argsort_merge & args,
        device const char    * src0,
        device const int32_t * tmp,
        device       int32_t * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int im  = tgpig[0] / args.ne01;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    const int start = im * (2 * args.len);
    const uint width = causal ? min(uint(args.ne00),
        (args.causal_start + uint(i01) + 1u) / args.causal_ratio) : 0u;
    const int work_width = causal ? int(((width - 1u) / args.block_width) * args.block_top_k +
        min((width - 1u) % args.block_width + 1u, args.block_top_k)) : args.ne0;

    // A merged run can contribute at most 512 entries to the final result.
    // Keep the original run offsets so the comparison and tie order agree.
    const int read_limit = prefix ? min(args.len, 512) : args.len;
    const int len0 = MIN(read_limit, MAX(0, work_width - start));
    const int len1 = MIN(read_limit, MAX(0, work_width - (start + args.len)));

    const int total = len0 + len1;

    device const int32_t * tmp0 = tmp + start
        + i01*args.ne0
        + i02*args.ne0*args.ne01
        + i03*args.ne0*args.ne01*args.ne02;

    device const int32_t * tmp1 = tmp0 + args.len;

    dst += start
        + i01*args.top_k
        + i02*args.top_k*args.ne01
        + i03*args.top_k*args.ne01*args.ne02;

    device const float * src0_row = (device const float *)(src0
        + args.nb01*i01
        + args.nb02*i02
        + args.nb03*i03);

    if (total == 0) {
        return;
    }

    const int chunk = (total + ntg.x - 1) / ntg.x;

    const int k0 = tpitg.x * chunk;
    const int limit = prefix ? min(args.top_k, 512) : args.top_k;
    const int k1 = MIN(MIN(k0 + chunk, total), limit);

    if (k0 >= limit) {
        return;
    }

    if (k0 >= total) {
        return;
    }

    int low  = k0 > len1 ? k0 - len1 : 0;
    int high = MIN(k0, len0);

    // binary-search partition (i, j) such that i + j = k
    while (low < high) {
        const int mid = (low + high) >> 1;

        const int32_t idx0 = tmp0[mid];
        const int32_t idx1 = tmp1[k0 - mid - 1];

        const float val0 = src0_row[idx0];
        const float val1 = src0_row[idx1];

        bool take_left;
        if (order == DS4_SORT_ORDER_ASC) {
            take_left = (val0 <= val1);
        } else {
            take_left = (val0 >= val1);
        }

        if (take_left) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    int i = low;
    int j = k0 - i;

    // keep the merge fronts into registers
    int32_t idx0 = 0;
    float   val0 = 0.0f;
    if (i < len0) {
        idx0 = tmp0[i];
        val0 = src0_row[idx0];
    }

    int32_t idx1 = 0;
    float   val1 = 0.0f;
    if (j < len1) {
        idx1 = tmp1[j];
        val1 = src0_row[idx1];
    }

    for (int k = k0; k < k1; ++k) {
        int32_t out_idx;

        if (i >= len0) {
            while (k < k1) {
                dst[k++] = tmp1[j++];
            }
            break;
        } else if (j >= len1) {
            while (k < k1) {
                dst[k++] = tmp0[i++];
            }
            break;
        } else {
            bool take_left;

            if (order == DS4_SORT_ORDER_ASC) {
                take_left = (val0 <= val1);
            } else {
                take_left = (val0 >= val1);
            }

            if (take_left) {
                out_idx = idx0;
                ++i;
                if (i < len0) {
                    idx0 = tmp0[i];
                    val0 = src0_row[idx0];
                }
            } else {
                out_idx = idx1;
                ++j;
                if (j < len1) {
                    idx1 = tmp1[j];
                    val1 = src0_row[idx1];
                }
            }
        }

        dst[k] = out_idx;
    }
}

// Host-visible merge variant used by DS4 top-k selection.
template [[host_name("kernel_argsort_merge_f32_i32_desc")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true>;
template [[host_name("kernel_argsort_merge_f32_i32_desc_causal_prefix")]] kernel argsort_merge_t kernel_argsort_merge_f32_i32<DS4_SORT_ORDER_DESC, true, true>;

struct ds4_metal_args_topk_rank {
    int32_t n;
    int32_t top_k;
    int32_t block;
    int32_t groups;
};

// Float order as unsigned order, with -0 as +0 since they compare equal.
static inline uint topk_key(float x) {
    uint u = as_type<uint>(x);
    if ((u & 0x7fffffffu) == 0) u = 0;
    return u & 0x80000000u ? ~u : u | 0x80000000u;
}

// Prefilter of the counting top-k below for long rows: the maxima of
// args.groups disjoint chunks (at least top_k), one simdgroup per chunk.
// The first threadgroup also resets the candidate count and the indirect
// dispatch size of the rank pass.
kernel void kernel_topk_bound_f32_desc(
        constant ds4_metal_args_topk_rank & args,
        device const float * src,
        device    uint32_t * peaks,
        device    uint32_t * header,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort col[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort nsg[[simdgroups_per_threadgroup]]) {
    if (tgpig == 0 && col < 4) header[col] = col == 1 || col == 2;
    const int g = int(tgpig) * nsg + sgitg;
    if (g >= args.groups) return;
    const int j0 = int((long)g * args.n / args.groups);
    const int j1 = int((long)(g + 1) * args.n / args.groups);
    uint peak = 0;
    for (int j = j0 + tiisg; j < j1; j += 32) peak = max(peak, topk_key(src[j]));
    peak = simd_max(peak);
    if (tiisg == 0) peaks[g] = peak;
}

// Compacts the entries the selection can reach (in no particular order: the
// ranks do not depend on it) and grows the rank pass to cover them; the
// others get no rank. The chunk maxima are distinct entries, so the floor of
// the top_k-th largest maximum to 16 key bits, found by two radix passes, has
// at least top_k entries at or above it.
kernel void kernel_topk_compact_f32_desc(
        constant ds4_metal_args_topk_rank & args,
        device const    float * src,
        device const uint32_t * peaks,
        device    atomic_uint * header,
        device       uint32_t * base,
        device        int32_t * cidx,
        device          float * cval,
        threadgroup atomic_uint * hist [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort col[[thread_index_in_threadgroup]],
        ushort ntg[[threads_per_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup uint * pick = (threadgroup uint *)(hist + 256);
    uint floor = 0, remaining = uint(args.top_k);
    for (int shift = 24; shift >= 16; shift -= 8) {
        if (col < 256) atomic_store_explicit(&hist[col], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = col; g < args.groups; g += ntg) {
            const uint key = peaks[g];
            if (shift == 24 || (key >> 24) == (floor >> 24)) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            // Lane l sums digits 255 - 8l down to 248 - 8l.
            uint bins[8], sum = 0;
            FOR_UNROLL (int b = 0; b < 8; ++b) {
                bins[b] = atomic_load_explicit(&hist[255 - 8 * tiisg - b], memory_order_relaxed);
                sum += bins[b];
            }
            uint above = simd_prefix_exclusive_sum(sum);
            if (above < remaining && remaining <= above + sum) {
                int b = 0;
                while (above + bins[b] < remaining) above += bins[b++];
                pick[0] = uint(255 - 8 * tiisg - b);
                pick[1] = remaining - above;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        floor |= pick[0] << shift;
        remaining = pick[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int j = int(tgpig) * ntg + col;
    const float x = j < args.n ? src[j] : 0.0f;
    const bool keep = j < args.n && topk_key(x) >= floor;
    const uint kept = simd_sum(uint(keep));
    if (!kept) {
        if (j < args.n) base[j] = UINT_MAX;
        return;
    }
    uint slot = 0;
    if (tiisg == 0) {
        slot = atomic_fetch_add_explicit(&header[3], kept, memory_order_relaxed);
        atomic_fetch_max_explicit(&header[0], (slot + kept + 31u) / 32u, memory_order_relaxed);
    }
    slot = simd_broadcast_first(slot) + simd_prefix_exclusive_sum(uint(keep));
    if (keep) {
        cidx[slot] = j;
        cval[slot] = x;
    } else if (j < args.n) {
        base[j] = UINT_MAX;
    }
}

// Counts the entries of src[j0, j1) above and equal to v. Sixteen loads
// issued ahead of their compares keep the loads overlapped.
static inline void topk_rank_count(device const float * src, int j0, int j1, float v,
                                   thread uint & above, thread uint & equal) {
    int j = j0;
    for (; j + 16 <= j1; j += 16) {
        float x[16];
        FOR_UNROLL (int u = 0; u < 16; ++u) x[u] = src[j + u];
        FOR_UNROLL (int u = 0; u < 16; ++u) {
            above += x[u] > v;
            equal += x[u] == v;
        }
    }
    for (; j < j1; ++j) {
        above += src[j] > v;
        equal += src[j] == v;
    }
}

// Top-k of one row by counting, in the order of the kernel_argsort_f32_i32
// and kernel_argsort_merge_f32_i32 chain. That chain sorts blocks of
// args.block entries and merges them keeping the earlier block on ties, so
// an entry's rank is the number of larger entries plus the equal entries of
// earlier blocks, plus the equal entries its own block's sort put first. Only
// that last part needs the sort: a selected entry tied inside its block flags
// the block for kernel_topk_ties_f32_i32_desc. Long rows rank only the
// candidates of kernel_topk_compact_f32_desc. Each lane owns one entry and the
// simdgroups split the candidates, so the pass spreads over the GPU where the
// chain sorts each block on one core.
template<bool compact>
kernel void kernel_topk_rank_f32_i32(
        constant ds4_metal_args_topk_rank & args,
        device const   float * src,
        device const uint32_t * header,
        device const int32_t * cidx,
        device const   float * cval,
        device       int32_t * dst,
        device    atomic_uint * ties,
        device      uint32_t * base,
        threadgroup uint32_t * counts [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort   nsg[[simdgroups_per_threadgroup]]) {
    if (!compact) cval = src;
    const int m = compact ? int(header[3]) : args.n;
    const int c = int(tgpig) * 32 + tiisg;
    const float v = c < m ? cval[c] : 0.0f;
    const int i = !compact ? c : c < m ? cidx[c] : 0;
    const int lo = i / args.block * args.block;
    const int hi = lo + args.block;
    const int chunk = (m + 16 * nsg - 1) / (16 * nsg) * 16;
    const int j1 = MIN((sgitg + 1) * chunk, m);
    const int j0 = sgitg * chunk;
    uint above = 0, before = 0, same = 0, after = 0;
    if (!compact) {
        topk_rank_count(src, j0, MIN(j1, lo), v, above, before);
        topk_rank_count(src, MAX(j0, lo), MIN(j1, hi), v, above, same);
        topk_rank_count(src, MAX(j0, hi), j1, v, above, after);
    } else {
        // The candidates are in no index order: blocks come from the indices.
        int j = j0;
        for (; j + 16 <= j1; j += 16) {
            float x[16];
            int xi[16];
            FOR_UNROLL (int u = 0; u < 16; ++u) {
                x[u] = cval[j + u];
                xi[u] = cidx[j + u];
            }
            FOR_UNROLL (int u = 0; u < 16; ++u) {
                above += x[u] > v;
                before += x[u] == v && xi[u] < lo;
                same += x[u] == v && xi[u] >= lo && xi[u] < hi;
            }
        }
        for (; j < j1; ++j) {
            above += cval[j] > v;
            before += cval[j] == v && cidx[j] < lo;
            same += cval[j] == v && cidx[j] >= lo && cidx[j] < hi;
        }
    }
    counts[sgitg * 32 + tiisg] = above + before;
    counts[(nsg + sgitg) * 32 + tiisg] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg != 0 || c >= m) return;
    uint rank = above + before;
    for (int s = 1; s < nsg; ++s) {
        rank += counts[s * 32 + tiisg];
        same += counts[(nsg + s) * 32 + tiisg];
    }
    base[i] = rank;
    if (rank >= uint(args.top_k)) return;
    dst[rank] = i;
    if (same > 1) atomic_fetch_or_explicit(&ties[i / args.block], 1u, memory_order_relaxed);
}

typedef decltype(kernel_topk_rank_f32_i32<false>) topk_rank_t;
template [[host_name("kernel_topk_rank_f32_i32_desc")]] kernel topk_rank_t kernel_topk_rank_f32_i32<false>;
template [[host_name("kernel_topk_rank_f32_i32_desc_compact")]] kernel topk_rank_t kernel_topk_rank_f32_i32<true>;

// Redoes the entries of a block kernel_topk_rank_f32_i32 flagged, and clears
// the flag: the network of kernel_argsort_f32_i32 orders the block, and each
// entry moves past the equal entries the network put before it.
kernel void kernel_topk_ties_f32_i32_desc(
        constant ds4_metal_args_topk_rank & args,
        device const float * src,
        device     int32_t * dst,
        device    uint32_t * ties,
        device const uint32_t * base,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort col[[thread_index_in_threadgroup]]) {
    if (!ties[tgpig]) return;
    threadgroup_barrier(mem_flags::mem_device);
    if (col == 0) ties[tgpig] = 0;
    const int n = args.n;
    const int nth = args.block;
    const int i00 = int(tgpig) * nth;
    threadgroup float * shmem_f32 = (threadgroup float *)(shmem_i32 + nth);
    shmem_i32[col] = i00 + col;
    if (i00 + col < n) shmem_f32[col] = src[i00 + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= nth; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                const int a = shmem_i32[col], b = shmem_i32[ixj];
                const bool exchange = (col & k) == 0 ?
                    (a >= n || (b < n && shmem_f32[a - i00] < shmem_f32[b - i00])) :
                    (b >= n || (a < n && shmem_f32[a - i00] > shmem_f32[b - i00]));
                if (exchange) {
                    shmem_i32[col] = b;
                    shmem_i32[ixj] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int i = shmem_i32[col];
    if (i >= n || base[i] >= uint(args.top_k)) return;
    const float v = shmem_f32[i - i00];
    int first = col;
    while (first > 0 && shmem_f32[shmem_i32[first - 1] - i00] == v) --first;
    const uint rank = base[i] + uint(col - first);
    if (rank < uint(args.top_k)) dst[rank] = i;
}
