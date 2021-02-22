# [MLKernels](http://github.com/wesselb/mlkernels)

[![CI](https://github.com/wesselb/mlkernels/workflows/CI/badge.svg)](https://github.com/wesselb/mlkernels/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/mlkernels/badge.svg?branch=main)](https://coveralls.io/github/wesselb/mlkernels?branch=main)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/mlkernels)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Kernels, the machine learning ones

Contents:

- [Installation](#installation)
- [Usage](#usage)
    - [Structured Matrix Types](#structured-matrix-types)
    - [AutoGrad, TensorFlow, PyTorch, or JAX? Your Choice!](#autograd-tensorflow-pytorch-or-jax-your-choice)
- [Available Kernels](#available-kernels)
- [Compositional Design](#compositional-design)
- [Displaying Kernels](#displaying-kernels)
- [Properties of Kernels](#properties-of-kernels)

TLDR:

```python
>>> from mlkernels import EQ, Linear

>>> k1 = 2 * EQ()

>>> k1
2 * EQ()

>>> k2 = 2 + EQ() * Linear()

>>> k2
2 * 1 + EQ() * Linear()

>>> k1(np.linspace(0, 1, 3))
array([[2.        , 1.76499381, 1.21306132],
       [1.76499381, 2.        , 1.76499381],
       [1.21306132, 1.76499381, 2.        ]])

>>> k2(np.linspace(0, 1, 3))
array([[2.        , 2.        , 2.        ],
       [2.        , 2.25      , 2.44124845],
       [2.        , 2.44124845, 3.        ]])
```

## Installation

```
pip install mlkernels
```

See also [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

## Usage

Let `k` be a kernel, e.g. `k = EQ()`.

*
    `k(x, y)` constructs the _kernel matrix_ for all pairs of points between `x` and
    `y`.
*
    `k(x)` is shorthand for `k(x, x)`.
*
    `k.elwise(x, y)` constructs the _kernel vector_ pairing the points in `x` and 
    `y` element-wise, which will be a _rank-2 column vector_.

Example:

```python
>>> k = EQ()

>>> k(np.linspace(0, 1, 3))
array([[1.        , 0.8824969 , 0.60653066],
       [0.8824969 , 1.        , 0.8824969 ],
       [0.60653066, 0.8824969 , 1.        ]])
 
>>> k.elwise(np.linspace(0, 1, 3), 0)
array([[1.        ],
       [0.8824969 ],
       [0.60653066]])
```

Inputs to kernels must be of one of the following three forms:

*
    If the input `x` is a _rank-0 tensor_, i.e. a scalar, then `x` refers to a
    single input location.
    For example, `0` simply refers to the sole input location `0`.

*
    If the input `x` is a _rank-1 tensor_, i.e. a vector, then every element of `x` is
    interpreted as a separate input location.
    For example, `np.linspace(0, 1, 10)` generates 10 different input locations 
    ranging from `0` to `1`.

*
    If the input `x` is a _rank-2 tensor_, i.e. a matrix, then every _row_ of `x` is
    interpreted as a separate input location. In this case inputs are
    multi-dimensional, and the columns correspond to the various input dimensions.

### Structured Matrix Types

MLKernels uses [an extension of LAB](https://github.com/wesselb/matrix) to
accelerate linear algebra with structured linear algebra primitives.
By calling `k(x, y)` or `k.elwise(x, y)`, these structured matrix types are 
automatically converted regular NumPy/TensorFlow/PyTorch/JAX arrays, so they won't 
bother you.
Would you want to preserve matrix structure, then you can use the exported functions
`pairwise` and `elwise`.

Example:

```python
>>> k = 2 * Delta()

>>> x = np.linspace(0, 5, 10)

>>> from mlkernels import pairwise

>>> pairwise(k, x)  # Preserve structure.
<diagonal matrix: shape=10x10, dtype=float64
 diag=[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]>

>>> k(x)  # Do not preserve structure.
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

These structured matrices are compatible with [LAB](https://github.com/wesselb/lab):
they know how to efficiently add, multiply, and do other linear algebra operations.

```python
>>> import lab as B

>>> B.matmul(pairwise(k, x), pairwise(k, x))
<diagonal matrix: shape=10x10, dtype=float64
 diag=[4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]>
```

You can eventually convert structured primitives to regular 
NumPy/TensorFlow/PyTorch/JAX arrays by calling `B.dense`:

```python
>>> import lab as B

>>> B.dense(B.matmul(pairwise(k, x), pairwise(k, x)))
array([[4., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 4., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 4., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 4., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 4., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 4., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 4., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 4., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 4., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 4.]])
```

### AutoGrad, TensorFlow, PyTorch, or JAX? Your Choice!

```python
from mlkernels.autograd import EQ, Linear
```

```python
from mlkernels.tensorflow import EQ, Linear
```

```python
from mlkernels.torch import EQ, Linear
```

```python
from mlkernels.jax import EQ, Linear
```

## Available Kernels

Constants function as constant kernels.
Besides that, the following kernels are available:

* `EQ()`, the exponentiated quadratic:

    $$ k(x, y) = \exp\left( -\frac{1}{2}\|x - y\|^2 \right); $$

* `RQ(alpha)`, the rational quadratic:

    $$ k(x, y) = \left( 1 + \frac{\|x - y\|^2}{2 \alpha} \right)^{-\alpha}; $$

* `Matern12()` or `Exp()`, the Matern–1/2 kernel:

    $$ k(x, y) = \exp\left( -\|x - y\| \right); $$

* `Matern32()`, the Matern–3/2 kernel:

    $$ k(x, y) = \left(
        1 + \sqrt{3}\|x - y\|
        \right)\exp\left(-\sqrt{3}\|x - y\|\right); $$

* `Matern52()`, the Matern–5/2 kernel:

    $$ k(x, y) = \left(
        1 + \sqrt{5}\|x - y\| + \frac{5}{3} \|x - y\|^2
       \right)\exp\left(-\sqrt{3}\|x - y\|\right); $$

* `Linear()`, the linear kernel:

  $$ k(x, y) = \langle x, y \rangle; $$

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

*
    Add and subtract kernels.

    Example:
    
    ```python
    >>> EQ() + Matern12()
    EQ() + Matern12()

    >>> EQ() + EQ()
    2 * EQ()

    >>> EQ() + 1
    EQ() + 1

    >>> EQ() + 0
    EQ()

    >>> EQ() - Matern12()
    EQ() - Matern12()

    >>> EQ() - EQ()
    0
    ```

*
    Multiply kernels.
    
    Example:

    ```python
    >>> EQ() * Matern12()
    EQ() * Matern12()

    >>> 2 * EQ()
    2 * EQ()

    >>> 0 * EQ()
    0
    ```

*
    Shift kernels.

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

*
    Stretch kernels.

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

*
    Select particular input dimensions of kernels.

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

*
    Transform the inputs of kernels.

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

*
    Numerically, but efficiently, take derivatives of kernels.
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
    
    Kernels "wrapping" others can be "unwrapped" by indexing `k[0]`.
     
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
2.12 * EQ()
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

    >>> EQ() + Matern12() == Matern12() + EQ()
    True

    >>> 2 * Matern12() == EQ() + Matern12()
    False

    >>> EQ() + Matern12() + Linear()  == Linear() + Matern12() + EQ()  # Too hard: cannot prove equality!
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


