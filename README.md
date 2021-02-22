# [MLKernels](http://github.com/wesselb/mlkernels)

[![CI](https://github.com/wesselb/mlkernels/workflows/CI/badge.svg)](https://github.com/wesselb/mlkernels/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/mlkernels/badge.svg?branch=main)](https://coveralls.io/github/wesselb/mlkernels?branch=main)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/mlkernels)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Kernels, the machine learning ones

Contents:

- [Installation](#installation)
- [Usage](#usage)
  - [Important Remark: Structured Matrix Types](#important-remark-structured-matrix-types)
- [Available Kernels](#available-kernels)
- [Compositional Design](#compositional-design)
- [Displaying Kernels](#displaying-kernels)
- [Properties of Kernels](#properties-of-kernels)


## Installation

```
pip install mlkernels
```

See also [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

## Usage

Inputs to kernels, henceforth referred to simply as _inputs_, 
must be of one of the following three forms:

* If the input `x` is a _rank 0 tensor_, i.e. a scalar, then `x` refers to a 
single input location. For example, `0` simply refers to the sole input 
location `0`.

* If the input `x` is a _rank 1 tensor_, then every element of `x` is 
interpreted as a separate input location. For example, `np.linspace(0, 1, 10)`
generates 10 different input locations ranging from `0` to `1`.

* If the input `x` is a _rank 2 tensor_, then every _row_ of `x` is 
interpreted as a separate input location. In this case inputs are 
multi-dimensional, and the columns correspond to the various input dimensions.

If `k` is a kernel, say `k = EQ()`, then `k(x, y)` constructs the _kernel 
matrix_ for all pairs of points between `x` and `y`. `k(x)` is shorthand for
`k(x, x)`. Furthermore, `k.elwise(x, y)` constructs the _kernel vector_ pairing
the points in `x` and `y` element wise, which will be a _rank 2 column vector_.
Instead of calling the kernel, one can also use the functions `pairwise` and `elwise`:
`pairwise(k, x, y)` and `elwise(k, x, y)`.

Example:

```python
>>> EQ()(np.linspace(0, 1, 3))
<dense matrix: shape=3x3, dtype=float64
 mat=[[1.    0.882 0.607]
      [0.882 1.    0.882]
      [0.607 0.882 1.   ]]>

>>> pairwise(EQ(), np.linspace(0, 1, 3))
<dense matrix: shape=3x3, dtype=float64
 mat=[[1.    0.882 0.607]
      [0.882 1.    0.882]
      [0.607 0.882 1.   ]]>
 
>>> EQ().elwise(np.linspace(0, 1, 3), 0)
array([[1.        ],
       [0.8824969 ],
       [0.60653066]])

>>> elwise(EQ(), np.linspace(0, 1, 3), 0)
array([[1.        ],
       [0.8824969 ],
       [0.60653066]])
```

### Important Remark: Structured Matrix Types

MLKernels uses [an extension of LAB](https://github.com/wesselb/matrix) to
accelerate linear algebra with structured linear algebra primitives.
You will encounter these primitives:

```python
>>> k = 2 * Delta()

>>> x = np.linspace(0, 5, 10)

>>> k(x)
<diagonal matrix: shape=10x10, dtype=float64
 diag=[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]>
```

If you're using [LAB](https://github.com/wesselb/lab) to further process these matrices,
then there is absolutely no need to worry:
these structured matrix types know how to add, multiply, and do other linear algebra
operations.

```python
>>> import lab as B

>>> B.matmul(k(x), k(x))
<diagonal matrix: shape=10x10, dtype=float64
 diag=[4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]>
```

If you're not using [LAB](https://github.com/wesselb/lab), you can convert these
structured primitives to regular NumPy/TensorFlow/PyTorch/JAX arrays by calling
`B.dense` (`B` is from [LAB](https://github.com/wesselb/lab)):

```python
>>> import lab as B

>>> B.dense(k(x))
array([[2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 2., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 2., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 2., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 2., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 2., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 2., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 2., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 2., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 2.]])
```

## Available Kernels

Constants function as constant kernels.
Besides that, the following kernels are available:

* `EQ()`, the exponentiated quadratic:

    $$ k(x, y) = \exp\left( -\frac{1}{2}\|x - y\|^2 \right); $$

* `RQ(alpha)`, the rational quadratic:

    $$ k(x, y) = \left( 1 + \frac{\|x - y\|^2}{2 \alpha} \right)^{-\alpha}; $$

* `Exp()` or `Matern12()`, the exponential kernel:

    $$ k(x, y) = \exp\left( -\|x - y\| \right); $$

* `Matern32()`, the Matern–3/2 kernel:

    $$ k(x, y) = \left(
        1 + \sqrt{3}\|x - y\|
        \right)\exp\left(-\sqrt{3}\|x - y\|\right); $$

* `Matern52()`, the Matern–5/2 kernel:

    $$ k(x, y) = \left(
        1 + \sqrt{5}\|x - y\| + \frac{5}{3} \|x - y\|^2
       \right)\exp\left(-\sqrt{3}\|x - y\|\right); $$

* `Delta()`, the Kronecker delta kernel:

    $$ k(x, y) = \begin{cases}
        1 & \text{if } x = y, \\
        0 & \text{otherwise};
       \end{cases} $$
       
* `DecayingKernel(alpha, beta)`:

    $$ k(x, y) = \frac{\|\beta\|^\alpha}{\|x + y + \beta\|^\alpha}; $$
    
* `LogKernel()`:

    $$ k(x, y) = \frac{\log(1 + \|x - y\|)}{\|x - y\|}; $$

* `TensorProductKernel(f)`:

    $$ k(x, y) = f(x)f(y). $$

    Adding or multiplying a `FunctionType` `f` to or with a kernel will 
    automatically translate `f` to `TensorProductKernel(f)`. For example,
    `f * k` will translate to `TensorProductKernel(f) * k`, and `f + k` will 
    translate to `TensorProductKernel(f) + k`.


## Compositional Design

* Add and subtract kernels.

    Example:
    
    ```python
    >>> EQ() + Exp()
    EQ() + Exp()

    >>> EQ() + EQ()
    2 * EQ()

    >>> EQ() + 1
    EQ() + 1

    >>> EQ() + 0
    EQ()

    >>> EQ() - Exp()
    EQ() - Exp()

    >>> EQ() - EQ()
    0
    ```

* Multiply kernels.
    
    Example:

    ```python
    >>> EQ() * Exp()
    EQ() * Exp()

    >>> 2 * EQ()
    2 * EQ()

    >>> 0 * EQ()
    0
    ```

* Shift kernels.

    Definition:
    
    ```python
    k.shift(c)(x, y) == k(x - c, y - c)

    k.shift(c1, c2)(x, y) == k(x - c1, y - c2)
    ```
    
    Example:
    
    ```python
    >>> Linear().shift(1)
    Linear() shift 1

    >>> EQ().shift(1, 2)
    EQ() shift (1, 2)
    ```

* Stretch kernels.

    Definition:
    
    ```python
    k.stretch(c)(x, y) == k(x / c, y / c)

    k.stretch(c1, c2)(x, y) == k(x / c1, y / c2)
    ```
  
    Example:    
    
    ```python
    >>> EQ().stretch(2)
    EQ() > 2

    >>> EQ().stretch(2, 3)
    EQ() > (2, 3)
    ```
    
    The `>` operator is implemented to provide a shorthand for stretching:
    
    ```python
    >>> EQ() > 2
    EQ() > 2
    ```

* Select particular input dimensions of kernels.

    Definition:

    ```python
    k.select([0])(x, y) == k(x[:, 0], y[:, 0])
  
    k.select([0, 1])(x, y) == k(x[:, [0, 1]], y[:, [0, 1]])

    k.select([0], [1])(x, y) == k(x[:, 0], y[:, 1])

    k.select(None, [1])(x, y) == k(x, y[:, 1])
    ```

    Example:

    ```python
    >>> EQ().select([0])
    EQ() : [0]
  
    >>> EQ().select([0, 1])
    EQ() : [0, 1]

    >>> EQ().select([0], [1])
    EQ() : ([0], [1])

    >>> EQ().select(None, [1])
    EQ() : (None, [1])
    ```

* Transform the inputs of kernels.

    Definition:

    ```python
    k.transform(f)(x, y) == k(f(x), f(y))

    k.transform(f1, f2)(x, y) == k(f1(x), f2(y))

    k.transform(None, f)(x, y) == k(x, f(y))
    ```
        
    Example:
        
    ```python
    >>> EQ().transform(f)
    EQ() transform f

    >>> EQ().transform(f1, f2)
    EQ() transform (f1, f2)

    >>> EQ().transform(None, f)
    EQ() transform (None, f)
    ```

* Numerically, but efficiently, take derivatives of kernels.
    This currently only works in TensorFlow.

    Definition:

    ```python
    k.diff(0)(x, y) == d/d(x[:, 0]) d/d(y[:, 0]) k(x, y)

    k.diff(0, 1)(x, y) == d/d(x[:, 0]) d/d(y[:, 1]) k(x, y)

    k.diff(None, 1)(x, y) == d/d(y[:, 1]) k(x, y)
    ```
        
    Example:

    ```python
    >>> EQ().diff(0)
    d(0) EQ()

    >>> EQ().diff(0, 1)
    d(0, 1) EQ()

    >>> EQ().diff(None, 1)
    d(None, 1) EQ()
    ```

*
    Make kernels periodic.

    Definition:

    ```python
    k.periodic(2 pi / w)(x, y) == k((sin(w * x), cos(w * x)), (sin(w * y), cos(w * y)))
    ```

    Example:
     
    ```python
    >>> EQ().periodic(1)
    EQ() per 1
    ```

* 
    Reverse the arguments of kernels.

    Definition:

    ```python
    reversed(k)(x, y) == k(y, x)
    ```

    Example:

    ```python
    >>> reversed(Linear())
    Reversed(Linear())
    ```
    
*
    Extract terms and factors from sums and products respectively of kernels.
    
    Example:
    
    ```python
    >>> (EQ() + RQ(0.1) + Linear()).term(1)
    RQ(0.1)

    >>> (2 * EQ() * Linear).factor(0)
    2
    ```
    
    Kernels and means "wrapping" others can be "unwrapped" by indexing `k[0]`
     or `m[0]`.
     
    Example:
    
    ```python
    >>> reversed(Linear())
    Reversed(Linear())
  
    >>> reversed(Linear())[0]
    Linear()

    >>> EQ().periodic(1)
    EQ() per 1

    >>> EQ().periodic(1)[0]
    EQ()
    ```

## Displaying Kernels

Kernels and means have a `display` method.
The `display` method accepts a callable formatter that will be applied before any value
is printed.
This comes in handy when pretty printing kernels.

Example:

```python
>>> print((2.12345 * EQ()).display(lambda x: f"{x:.2f}"))
2.12 * EQ(), 0
```

## Properties of Kernels

*
    Kernels can be equated to check for equality.
    This will attempt basic algebraic manipulations.
    If the means and kernels are not equal _or_ equality cannot be proved, `False` is
    returned.
    
    Example of equating kernels:

    ```python
    >>>  2 * EQ() == EQ() + EQ()
    True

    >>> EQ() + Exp() == Exp() + EQ()
    True

    >>> 2 * Exp() == EQ() + Exp()
    False

    >>> EQ() + Exp() + Linear()  == Linear() + Exp() + EQ()  # Too hard: cannot prove equality!
    False
    ```


*
    The stationarity of a kernel `k` can always be determined by querying
    `k.stationary`.

    Example of querying the stationarity:

    ```python
    >>> EQ().stationary
    True

    >>> (EQ() + Linear()).stationary
    False
    ```


