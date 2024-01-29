/*
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <http://unlicense.org/>
 */
#ifndef SVD_H
#define SVD_H
#ifndef NO_OSTREAM
#include <iostream>
#include <cuda_runtime.h>

#endif

namespace svd
{
    class SMat3
    {
    public:
        float m00, m01, m02, m11, m12, m22;
    public:
        __host__ __device__ SMat3();
        __host__ __device__ SMat3(const float m00, const float m01, const float m02,
              const float m11, const float m12, const float m22);
        __host__ __device__ void clear() ;
        __host__ __device__ void setSymmetric(const float m00, const float m01, const float m02,
                          const float m11,
                          const float m12, const float m22) ;
        __host__ __device__ void setSymmetric(const SMat3 &rhs) ;
    private:
        __host__ __device__ SMat3(const SMat3 &rhs);
        __host__ __device__ SMat3 &operator=(const SMat3 &rhs);
    };
    class Mat3
    {
    public:
        float m00, m01, m02, m10, m11, m12, m20, m21, m22;
    public:
        __host__ __device__  Mat3();
        __host__ __device__  Mat3(const float m00, const float m01, const float m02,
             const float m10, const float m11, const float m12,
             const float m20, const float m21, const float m22);
        __host__ __device__  void clear() ;
        __host__ __device__ void set(const float m00, const float m01, const float m02,
                 const float m10, const float m11, const float m12,
                 const float m20, const float m21, const float m22) ;
        __host__ __device__ void set(const Mat3 &rhs) ;
        __host__ __device__ void setSymmetric(const float a00, const float a01, const float a02,
                          const float a11, const float a12, const float a22) ;
        __host__ __device__ void setSymmetric(const SMat3 &rhs);
    private:
        __host__ __device__ Mat3(const Mat3 &rhs);
        __host__ __device__ Mat3 &operator=(const Mat3 &rhs);
    };
    class Vec3
    {
    public:
        float x, y, z;
    public:
        __host__ __device__ Vec3();
        __host__ __device__ Vec3(const float x, const float y, const float z);
        __host__ __device__ void clear();
        __host__ __device__ void set(const float x, const float y, const float z);
        __host__ __device__ void set(const Vec3 &rhs);
    private:
        __host__ __device__ Vec3(const Vec3 &rhs);
        __host__ __device__ Vec3 &operator=(const Vec3 &rhs);
    };
#ifndef NO_OSTREAM
    std::ostream &operator<<(std::ostream &os, const Mat3 &m) ;
    std::ostream &operator<<(std::ostream &os, const SMat3 &m) ;
    std::ostream &operator<<(std::ostream &os, const Vec3 &v) ;
#endif
    class MatUtils
    {
    public:
        __host__ __device__   static float fnorm(const Mat3 &a) ;
        __host__ __device__ static float fnorm(const SMat3 &a) ;
        __host__ __device__ static float off(const Mat3 &a) ;
        __host__ __device__ static float off(const SMat3 &a) ;

    public:
        __host__ __device__ static void mmul(Mat3 &out, const Mat3 &a, const Mat3 &b) ;
        __host__ __device__ static void mmul_ata(SMat3 &out, const Mat3 &a) ;
        __host__ __device__ static void transpose(Mat3 &out, const Mat3 &a);
        __host__ __device__ static void vmul(Vec3 &out, const Mat3 &a, const Vec3 &v) ;
        __host__ __device__ static void vmul_symmetric(Vec3 &out, const SMat3 &a, const Vec3 &v) ;
    };
    class VecUtils
    {
    public:
        __host__ __device__ static void addScaled(Vec3 &v, const float s, const Vec3 &rhs) ;
        __host__ __device__ static float dot(const Vec3 &a, const Vec3 &b) ;
        __host__ __device__ static void normalize(Vec3 &v) ;
        __host__ __device__ static void scale(Vec3 &v, const float s) ;
        __host__ __device__ static void sub(Vec3 &c, const Vec3 &a, const Vec3 &b) ;
    };
    class Givens
    {
    public:
        __host__ __device__ static void rot01_post(Mat3 &m, const float c, const float s);
        __host__ __device__ static void rot02_post(Mat3 &m, const float c, const float s);
        __host__ __device__ static void rot12_post(Mat3 &m, const float c, const float s);
    };
    class Schur2
    {
    public:
        __host__ __device__ static void rot01(SMat3 &out, float &c, float &s) ;
        __host__ __device__ static void rot02(SMat3 &out, float &c, float &s) ;
        __host__ __device__ static void rot12(SMat3 &out, float &c, float &s) ;
    };
    class Svd
    {
    public:
        __host__ __device__ static void getSymmetricSvd(const SMat3 &a, SMat3 &vtav, Mat3 &v,
                                    const float tol, const int max_sweeps);
        __host__ __device__ static void pseudoinverse(Mat3 &out, const SMat3 &d, const Mat3 &v,
                                  const float tol);
        __host__ __device__ static float solveSymmetric(const SMat3 &A, const Vec3 &b, Vec3 &x,
                                     const float svd_tol, const int svd_sweeps, const float pinv_tol);
    };
    class LeastSquares
    {
    public:
        __host__ __device__ static float solveLeastSquares(const Mat3 &a, const Vec3 &b, Vec3 &x,
                                        const float svd_tol, const int svd_sweeps, const float pinv_tol) ;
    };
};
#endif