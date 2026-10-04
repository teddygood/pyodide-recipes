#include "blis.h"
#include <dlfcn.h>

static gemm_ukr_ft original;
static cntx_t *context;
static unsigned calls;

static void counted(dim_t m, dim_t n, dim_t k, const void *alpha,
                    const void *a, const void *b, const void *beta, void *c,
                    inc_t rs, inc_t cs, const auxinfo_t *aux, const cntx_t *ctx)
{
    ++calls;
    original(m, n, k, alpha, a, b, beta, c, rs, cs, aux, ctx);
}

int probe_install(void)
{
    bli_init();
    context = (cntx_t *)bli_gks_query_cntx();
    original = (gemm_ukr_ft)bli_cntx_get_ukr_dt(BLIS_DOUBLE, BLIS_GEMM_UKR, context);
    bli_cntx_set_ukr_dt((void_fp)counted, BLIS_DOUBLE, BLIS_GEMM_UKR, context);
    calls = 0;
    return original != 0;
}

unsigned probe_calls(void) { return calls; }
void probe_reset(void) { calls = 0; }
void probe_restore(void)
{
    bli_cntx_set_ukr_dt((void_fp)original, BLIS_DOUBLE, BLIS_GEMM_UKR, context);
}
int probe_selected_is_simd(void)
{
    void *simd = dlsym(RTLD_DEFAULT, "bli_dgemm_wasm32_simd128_4x4");
    return simd && simd == (void *)original;
}
int probe_mr(void) { return bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, context); }
int probe_nr(void) { return bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, context); }
