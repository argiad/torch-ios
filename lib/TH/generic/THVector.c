#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THVector.c"
#else

#ifdef __NEON_HL__

#include <arm_neon.h>

static inline void THVector_(neon_axpy)(float *dst,
                const float *src,
                const float alpha,
                const long elemCnt)
{
    long i;
    for (i = 0; i <= elemCnt - 16; i += 16)
    {
        float32x4_t q0 = vld1q_f32(src + i);
        float32x4_t q1 = vld1q_f32(src + i + 4);
        float32x4_t q2 = vld1q_f32(src + i + 8);
        float32x4_t q3 = vld1q_f32(src + i + 12);
        float32x4_t q4 = vld1q_f32(dst + i);
        float32x4_t q5 = vld1q_f32(dst + i + 4);
        float32x4_t q6 = vld1q_f32(dst + i + 8);
        float32x4_t q7 = vld1q_f32(dst + i + 12);
        q0 = vmlaq_n_f32(q4, q0, alpha);
        q1 = vmlaq_n_f32(q5, q1, alpha);
        q2 = vmlaq_n_f32(q6, q2, alpha);
        q3 = vmlaq_n_f32(q7, q3, alpha);
        vst1q_f32(dst + i,      q0);
        vst1q_f32(dst + i + 4,  q1);
        vst1q_f32(dst + i + 8,  q2);
        vst1q_f32(dst + i + 12, q3);
    }
    for (; i < elemCnt; i++)
    {
        dst[i] = src[i] * alpha + dst[i];
    }
}
#endif

static inline void THVector_(fill)(real *x, const real c, const long n) {
  long i = 0;

  for(; i < n-4; i += 4)
  {
    x[i] = c;
    x[i+1] = c;
    x[i+2] = c;
    x[i+3] = c;
  }

  for(; i < n; i++)
    x[i] = c;
}

static inline void THVector_(add)(real *y, const real *x, const real c, const long n)
{
#if defined(__NEON_HL__) && defined(TH_REAL_IS_FLOAT)
  THVector_(neon_axpy)(y, x, c, n);
#else
  long i = 0;

  for(;i < n-4; i += 4)
  {
    y[i] += c * x[i];
    y[i+1] += c * x[i+1];
    y[i+2] += c * x[i+2];
    y[i+3] += c * x[i+3];
  }

  for(; i < n; i++)
    y[i] += c * x[i];
#endif
}

static inline void THVector_(diff)(real *z, const real *x, const real *y, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    z[i] = x[i] - y[i];
    z[i+1] = x[i+1] - y[i+1];
    z[i+2] = x[i+2] - y[i+2];
    z[i+3] = x[i+3] - y[i+3];
  }

  for(; i < n; i++)
    z[i] = x[i] - y[i];
}

static inline void THVector_(scale)(real *y, const real c, const long n)
{
  long i = 0;

  for(; i < n-4; i +=4)
  {
    y[i] *= c;
    y[i+1] *= c;
    y[i+2] *= c;
    y[i+3] *= c;
  }

  for(; i < n; i++)
    y[i] *= c;
}

static inline void THVector_(mul)(real *y, const real *x, const long n)
{
  long i = 0;

  for(; i < n-4; i += 4)
  {
    y[i] *= x[i];
    y[i+1] *= x[i+1];
    y[i+2] *= x[i+2];
    y[i+3] *= x[i+3];
  }

  for(; i < n; i++)
    y[i] *= x[i];
}

static inline void THVector_(conv1d)(real *y, real *x, real *c, real a, const long n, const long cn, unsigned char reverse){
    long i;
    if (reverse==0){
        for(i = 0; i < cn; i++)
            THVector_(add)(y, (x + i), (c[i]*a), n);
    }
    else{
        for(i = 0; i < cn; i++)
            THVector_(add)(y, (x + i), (c[-i]*a), n);
    }
}

#endif
