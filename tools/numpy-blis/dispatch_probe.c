#include "blis.h"
#include <dlfcn.h>

static gemm_ukr_ft original[4];
static cntx_t* context;
static unsigned calls[4];
static ind_t methods[2];
static const num_t datatypes[] = {BLIS_FLOAT, BLIS_DOUBLE, BLIS_SCOMPLEX, BLIS_DCOMPLEX};

#define COUNTED(dt) \
static void counted_##dt(dim_t m, dim_t n, dim_t k, const void* alpha, \
                         const void* a, const void* b, const void* beta, void* c, \
                         inc_t rs, inc_t cs, const auxinfo_t* aux, const cntx_t* ctx) \
{ \
    ++calls[dt]; \
    original[dt](m, n, k, alpha, a, b, beta, c, rs, cs, aux, ctx); \
}
COUNTED(0)
COUNTED(1)
COUNTED(2)
COUNTED(3)

int probe_install_all(int native)
{
    bli_init();
    context = (cntx_t*)bli_gks_query_cntx();
    void_fp counted[] = {(void_fp)counted_0, (void_fp)counted_1,
                         (void_fp)counted_2, (void_fp)counted_3};
    for (int dt = 0; dt < 4; ++dt) {
        original[dt] = (gemm_ukr_ft)bli_cntx_get_ukr_dt(datatypes[dt], BLIS_GEMM_UKR, context);
        if (!original[dt]) return 0;
        calls[dt] = 0;
        bli_cntx_set_ukr_dt(counted[dt], datatypes[dt], BLIS_GEMM_UKR, context);
    }
    for (int dt = 2; dt < 4; ++dt) {
        methods[dt - 2] = bli_ind_oper_find_avail(BLIS_GEMM, datatypes[dt]);
        if (native) bli_ind_oper_enable_only(BLIS_GEMM, BLIS_NAT, datatypes[dt]);
    }
    return 1;
}

void probe_restore_all(void)
{
    for (int dt = 0; dt < 4; ++dt)
        bli_cntx_set_ukr_dt((void_fp)original[dt], datatypes[dt], BLIS_GEMM_UKR, context);
    for (int dt = 2; dt < 4; ++dt)
        bli_ind_oper_enable_only(BLIS_GEMM, methods[dt - 2], datatypes[dt]);
}

void probe_reset_all(void) { for (int dt = 0; dt < 4; ++dt) calls[dt] = 0; }
unsigned probe_calls_dt(int dt) { return calls[dt]; }
int probe_selected_dt(int dt)
{
    const char* symbols[] = {
        "bli_sgemm_wasm32_simd128_4x4", "bli_dgemm_wasm32_simd128_4x4",
        "bli_cgemm_wasm32_simd128_4x2", "bli_zgemm_wasm32_simd128_2x2"
    };
    void* simd = dlsym(RTLD_DEFAULT, symbols[dt]);
    return simd && simd == (void*)original[dt];
}
int probe_mr_dt(int dt) { return bli_cntx_get_blksz_def_dt(datatypes[dt], BLIS_MR, context); }
int probe_nr_dt(int dt) { return bli_cntx_get_blksz_def_dt(datatypes[dt], BLIS_NR, context); }
int probe_method_dt(int dt) { return bli_ind_oper_find_avail(BLIS_GEMM, datatypes[dt]); }
void probe_set_native(int native)
{
    bli_init();
    for (int dt = 2; dt < 4; ++dt)
        bli_ind_oper_enable_only(BLIS_GEMM, native ? BLIS_NAT : BLIS_1M, datatypes[dt]);
}

void probe_set_methods(int c, int z)
{
    bli_ind_oper_enable_only(BLIS_GEMM, c, BLIS_SCOMPLEX);
    bli_ind_oper_enable_only(BLIS_GEMM, z, BLIS_DCOMPLEX);
}

// Preserve the original DGEMM-only probe API.
int probe_install(void) { return probe_install_all(0); }
unsigned probe_calls(void) { return calls[1]; }
void probe_reset(void) { probe_reset_all(); }
void probe_restore(void) { probe_restore_all(); }
int probe_selected_is_simd(void) { return probe_selected_dt(1); }
int probe_mr(void) { return probe_mr_dt(1); }
int probe_nr(void) { return probe_nr_dt(1); }
