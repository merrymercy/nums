# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import warnings

from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray
from nums.core.storage.storage import ArrayGrid
import numpy as np
import scipy.stats
from nums.numpy import numpy_utils as numpy_utils


def _not_implemented(fun):
    # From project JAX: https://github.com/google/jax/blob/master/jax/numpy/lax_numpy.py
    def wrapped(*args, **kwargs):
        # pylint: disable=unused-argument
        msg = "NumPy function {} not yet implemented."
        raise NotImplementedError(msg.format(fun))
    return wrapped


# We require app instance to initialize constants.
app = instance()


############################################
# Constants
############################################

pi = app.scalar(np.pi)
e = app.scalar(np.e)
euler_gamma = app.scalar(np.euler_gamma)
inf = infty = Inf = Infinity = PINF = app.scalar(np.inf)
NINF = app.scalar(np.NINF)
PZERO = app.scalar(np.PZERO)
NZERO = app.scalar(np.NZERO)
nan = NAN = NaN = app.scalar(np.nan)


############################################
# Creation and I/O Ops
############################################


def loadtxt(fname, dtype=float, comments='# ', delimiter=',',
            converters=None, skiprows=0, usecols=None, unpack=False,
            ndmin=0, encoding='bytes', max_rows=None) -> BlockArray:
    num_rows = numpy_utils.get_num_cores(app)
    try:
        ba: BlockArray = app.loadtxt(
            fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows,
            usecols=usecols, unpack=unpack, ndmin=ndmin,
            encoding=encoding, max_rows=max_rows, num_cpus=num_rows)
        shape = ba.shape
        block_shape = app.compute_block_shape(shape, dtype)
        return ba.reshape(block_shape=block_shape)
    except Exception as _:
        warnings.warn("Failed to load text data in parallel; using np.loadtxt locally.")
        np_arr = np.loadtxt(fname, dtype=dtype, comments=comments, delimiter=delimiter,
                            converters=converters, skiprows=skiprows,
                            usecols=usecols, unpack=unpack, ndmin=ndmin,
                            encoding=encoding, max_rows=max_rows)
        shape = np_arr.shape
        block_shape = app.compute_block_shape(shape, dtype)
        return app.array(np_arr, block_shape=block_shape)


def array(object, dtype=None, copy=True, order="K", ndmin=0, subok=False) -> BlockArray:
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    if ndmin != 0:
        raise NotImplementedError("Only ndmin=0 is currently supported.")
    if subok:
        raise ValueError("subok must be False.")
    if isinstance(object, BlockArray):
        if copy:
            object = object.copy()
        if dtype is not None:
            if dtype is not object.dtype:
                object = object.astype(dtype)
        return object
    result = np.array(object, dtype=dtype, copy=copy,
                      order=order, ndmin=ndmin, subok=subok)
    dtype = np.__getattribute__(str(result.dtype))
    shape = result.shape
    block_shape = app.compute_block_shape(shape, dtype)
    return app.array(result, block_shape)


def empty(shape, dtype=np.float):
    block_shape = app.compute_block_shape(shape, dtype)
    return app.empty(shape=shape, block_shape=block_shape, dtype=dtype)


def zeros(shape, dtype=np.float):
    block_shape = app.get_block_shape(shape, dtype)
    return app.zeros(shape=shape, block_shape=block_shape, dtype=dtype)


def ones(shape, dtype=np.float):
    block_shape = app.get_block_shape(shape, dtype)
    return app.ones(shape=shape, block_shape=block_shape, dtype=dtype)


def empty_like(prototype: BlockArray, dtype=None, order='K', shape=None):
    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    return empty(shape, dtype)


def zeros_like(prototype, dtype=None, order='K', shape=None):
    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    return zeros(shape, dtype)


def ones_like(prototype, dtype=None, order='K', shape=None):
    if shape is None:
        shape = prototype.shape
    if dtype is None:
        dtype = prototype.dtype
    if order is not None and order != "K":
        raise NotImplementedError("Only order='K' is supported.")
    return ones(shape, dtype)


def concatenate(arrays, axis=0, out=None):
    if out is not None:
        raise NotImplementedError("out is currently not supported for concatenate.")
    # Pick the mode along specified axis.
    axis_block_size = scipy.stats.mode(list(map(
        lambda arr: arr.block_shape[axis], arrays
    ))).mode.item()
    return app.concatenate(arrays, axis=axis, axis_block_size=axis_block_size)


def split(ary: BlockArray, indices_or_sections, axis=0):
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError("Split currently supports integers only.")
    dim_total = ary.shape[axis]
    # Splits into N equal arrays, and raise if this is not possible.
    if dim_total % indices_or_sections != 0:
        raise ValueError("ary axis %s cannot be split into %s equal arrays." % (
            axis,
            indices_or_sections))
    dim_partial = dim_total // indices_or_sections
    results = []
    ss_op = [slice(None, None, 1) for _ in ary.shape]
    for i in range(0, dim_total, dim_partial):
        start = i
        stop = i + dim_partial
        ss_op[axis] = slice(start, stop, 1)
        ary_part = ary[tuple(ss_op)]
        results.append(ary_part)
    return tuple(results)


def identity(n: int, dtype=np.float) -> BlockArray:
    return eye(n, n, dtype=dtype)


def eye(N, M=None, k=0, dtype=np.float):
    if k != 0:
        raise NotImplementedError("Only k==0 is currently supported.")
    if M is None:
        M = N
    shape = (N, M)
    block_shape = app.get_block_shape(shape, dtype)
    return app.eye(shape, block_shape, dtype)


def diag(v: BlockArray, k=0) -> BlockArray:
    if k != 0:
        raise NotImplementedError("Only k==0 is currently supported.")
    return app.diag(v)


############################################
# Manipulation Ops
############################################


def arange(start=None, stop=None, step=1, dtype=np.int64) -> BlockArray:
    if stop is None:
        stop = start
        start = 0
    if step != 1:
        raise NotImplementedError("Only step size of 1 is currently supported.")
    shape = (stop - start,)
    block_shape = app.get_block_shape(shape, dtype)
    return app.arange(shape, block_shape, step, dtype)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    shape = (num,)
    dtype = np.float64 if dtype is None else dtype
    block_shape = app.get_block_shape(shape, dtype)
    return app.linspace(start, stop, shape, block_shape, endpoint, retstep, dtype, axis)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    ba: BlockArray = linspace(start, stop, num, endpoint, dtype=None, axis=axis)
    ba = power(app.scalar(base), ba)
    if dtype is not None and dtype != ba.dtype:
        ba = ba.astype(dtype)
    return ba


############################################
# Reduction Ops
############################################


def min(a: BlockArray, axis=None, out=None,
        keepdims=False, initial=None, where=None) -> BlockArray:
    if initial is not None:
        raise NotImplementedError("'initial' is currently not supported.")
    if where is not None:
        raise NotImplementedError("'where' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.min(a, axis=axis, keepdims=keepdims)


amin = min


def max(a: BlockArray, axis=None, out=None,
        keepdims=False, initial=None, where=None) -> BlockArray:
    if initial is not None:
        raise NotImplementedError("'initial' is currently not supported.")
    if where is not None:
        raise NotImplementedError("'where' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.max(a, axis=axis, keepdims=keepdims)


amax = max


def argmin(a: BlockArray, axis=None, out=None):
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.argop("argmin", a, axis=axis)


def argmax(a, axis=None, out=None):
    if len(a.shape) > 1:
        raise NotImplementedError("argmax currently only supports one-dimensional arrays.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.argop("argmax", a, axis=axis)


def sum(a: BlockArray, axis=None, dtype=None, out=None,
        keepdims=False, initial=None, where=None) -> BlockArray:
    if initial is not None:
        raise NotImplementedError("'initial' is currently not supported.")
    if where is not None:
        raise NotImplementedError("'where' is currently not supported.")
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.sum(a, axis=axis, keepdims=keepdims, dtype=dtype)


def mean(a: BlockArray, axis=None, dtype=None, out=None, keepdims=False):
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.mean(a, axis=axis, keepdims=keepdims, dtype=dtype)


def var(a: BlockArray, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.var(a, axis=axis, ddof=ddof, keepdims=keepdims, dtype=dtype)


def std(a: BlockArray, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if out is not None:
        raise NotImplementedError("'out' is currently not supported.")
    return app.std(a, axis=axis, ddof=ddof, keepdims=keepdims, dtype=dtype)


############################################
# Utility Ops
############################################


def allclose(a: BlockArray, b: BlockArray, rtol=1.e-5, atol=1.e-8, equal_nan=False) -> BlockArray:
    if equal_nan is not False:
        raise NotImplementedError("equal_nan=True not supported.")
    return app.allclose(a, b, rtol, atol)


############################################
# Generated Ops (Unary, Binary)
############################################


def abs(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="abs",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def absolute(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="absolute",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def arccos(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="arccos",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def arccosh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="arccosh",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def arcsin(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="arcsin",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def arcsinh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="arcsinh",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def arctan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="arctan",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def arctanh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="arctanh",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def bitwise_not(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="bitwise_not",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def cbrt(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="cbrt",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def ceil(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="ceil",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def conj(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="conj",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def conjugate(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="conjugate",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def cos(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="cos",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def cosh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="cosh",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def deg2rad(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="deg2rad",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def degrees(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="degrees",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def exp(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="exp",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def exp2(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="exp2",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def expm1(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="expm1",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def fabs(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="fabs",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def floor(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="floor",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def invert(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="invert",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def isfinite(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="isfinite",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def isinf(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="isinf",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def isnan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="isnan",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def log(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="log",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def log10(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="log10",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def log1p(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="log1p",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def log2(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="log2",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def logical_not(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="logical_not",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def negative(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="negative",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def positive(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="positive",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def rad2deg(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="rad2deg",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def radians(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="radians",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def reciprocal(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="reciprocal",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def rint(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="rint",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def sign(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="sign",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def signbit(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="signbit",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def sin(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="sin",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def sinh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="sinh",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def spacing(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="spacing",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def sqrt(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="sqrt",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def square(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="square",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def tan(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="tan",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def tanh(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="tanh",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def trunc(x: BlockArray, out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_uop(op_name="trunc",
                       arr=x,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def add(x1: BlockArray, x2: BlockArray,
        out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="add",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def arctan2(x1: BlockArray, x2: BlockArray,
            out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="arctan2",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def bitwise_and(x1: BlockArray, x2: BlockArray,
                out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="bitwise_and",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def bitwise_or(x1: BlockArray, x2: BlockArray,
               out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="bitwise_or",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def bitwise_xor(x1: BlockArray, x2: BlockArray,
                out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="bitwise_xor",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def copysign(x1: BlockArray, x2: BlockArray,
             out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="copysign",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def divide(x1: BlockArray, x2: BlockArray,
           out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="divide",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def equal(x1: BlockArray, x2: BlockArray,
          out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="equal",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def float_power(x1: BlockArray, x2: BlockArray,
                out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="float_power",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def floor_divide(x1: BlockArray, x2: BlockArray,
                 out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="floor_divide",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def fmax(x1: BlockArray, x2: BlockArray,
         out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="fmax",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def fmin(x1: BlockArray, x2: BlockArray,
         out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="fmin",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def fmod(x1: BlockArray, x2: BlockArray,
         out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="fmod",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def gcd(x1: BlockArray, x2: BlockArray,
        out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="gcd",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def greater(x1: BlockArray, x2: BlockArray,
            out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="greater",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def greater_equal(x1: BlockArray, x2: BlockArray,
                  out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="greater_equal",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def heaviside(x1: BlockArray, x2: BlockArray,
              out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="heaviside",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def hypot(x1: BlockArray, x2: BlockArray,
          out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="hypot",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def lcm(x1: BlockArray, x2: BlockArray,
        out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="lcm",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def ldexp(x1: BlockArray, x2: BlockArray,
          out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="ldexp",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def left_shift(x1: BlockArray, x2: BlockArray,
               out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="left_shift",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def less(x1: BlockArray, x2: BlockArray,
         out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="less",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def less_equal(x1: BlockArray, x2: BlockArray,
               out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="less_equal",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def logaddexp(x1: BlockArray, x2: BlockArray,
              out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="logaddexp",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def logaddexp2(x1: BlockArray, x2: BlockArray,
               out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="logaddexp2",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def logical_and(x1: BlockArray, x2: BlockArray,
                out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="logical_and",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def logical_or(x1: BlockArray, x2: BlockArray,
               out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="logical_or",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def logical_xor(x1: BlockArray, x2: BlockArray,
                out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="logical_xor",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def maximum(x1: BlockArray, x2: BlockArray,
            out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="maximum",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def minimum(x1: BlockArray, x2: BlockArray,
            out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="minimum",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def mod(x1: BlockArray, x2: BlockArray,
        out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="mod",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def multiply(x1: BlockArray, x2: BlockArray,
             out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="multiply",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def nextafter(x1: BlockArray, x2: BlockArray,
              out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="nextafter",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def not_equal(x1: BlockArray, x2: BlockArray,
              out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="not_equal",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def power(x1: BlockArray, x2: BlockArray,
          out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="power",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def remainder(x1: BlockArray, x2: BlockArray,
              out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="remainder",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def right_shift(x1: BlockArray, x2: BlockArray,
                out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="right_shift",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def subtract(x1: BlockArray, x2: BlockArray,
             out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="subtract",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))


def true_divide(x1: BlockArray, x2: BlockArray,
                out: BlockArray = None, where=True, **kwargs) -> BlockArray:
    return app.map_bop(op_name="true_divide",
                       arr_1=x1,
                       arr_2=x2,
                       out=out,
                       where=where,
                       kwargs=numpy_utils.ufunc_kwargs(kwargs))
