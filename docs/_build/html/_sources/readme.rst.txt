`MLKernels <http://github.com/wesselb/mlkernels>`__
===================================================

|CI| |Coverage Status| |Latest Docs| |Code style: black|

Kernels, the machine learning ones (also, mean functions)

Contents:

-  `Installation <#installation>`__
-  `Usage <#usage>`__
-  `AutoGrad, TensorFlow, PyTorch, or JAX? Your
   Choice! <#autograd-tensorflow-pytorch-or-jax-your-choice>`__
-  `Available Kernels <#available-kernels>`__
-  `Available Means <#available-means>`__
-  `Compositional Design <#compositional-design>`__
-  `Displaying Kernels and Means <#displaying-kernels-and-means>`__
-  `Properties of Kernels and
   Means <#properties-of-kernels-and-means>`__
-  `Structured Matrix Types <#structured-matrix-types>`__
-  `Implementing Your Own Kernel <#implementing-your-own-kernel>`__

TLDR:

.. code:: python

   >>> from mlkernels import EQ, Linear

   >>> k1 = 2 * Linear() + 1

   >>> k1
   2 * Linear() + 1

   >>> k1(np.linspace(0, 1, 3))  # Structured matrices enable efficiency.
   <low-rank matrix: shape=3x3, dtype=float64, rank=2
    left=[[0.  1. ]
          [0.5 1. ]
          [1.  1. ]]
    middle=[[2. 0.]
            [0. 1.]]
    right=[[0.  1. ]
           [0.5 1. ]
           [1.  1. ]]>

   >>> import lab as B

   >>> B.dense(k1(np.linspace(0, 1, 3)))  # Discard structure: get a regular NumPy array.
   array([[1. , 1. , 1. ],
          [1. , 1.5, 2. ],
          [1. , 2. , 3. ]])

   >>> k2 = 2 + EQ() * Linear()

   >>> k2
   2 * 1 + EQ() * Linear()

   >>> k2(np.linspace(0, 1, 3))
   <dense matrix: shape=3x3, dtype=float64
    mat=[[2.    2.    2.   ]
         [2.    2.25  2.441]
         [2.    2.441 3.   ]]>

   >>> B.dense(k1(np.linspace(0, 1, 3)))
   array([[2.        , 2.        , 2.        ],
          [2.        , 2.25      , 2.44124845],
          [2.        , 2.44124845, 3.        ]])

Installation
------------

::

   pip install mlkernels

See also `the instructions
here <https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc>`__.

Usage
-----

Let ``k`` be a kernel, e.g. ``k = EQ()``.

-  ``k(x, y)`` constructs the *kernel matrix* for all pairs of points
   between ``x`` and ``y``.
-  ``k(x)`` is shorthand for ``k(x, x)``.
-  ``k.elwise(x, y)`` constructs the *kernel vector* pairing the points
   in ``x`` and ``y`` element-wise, which will be a *rank-2 column
   vector*.

Example:

.. code:: python

   >>> k = EQ()

   >>> k(np.linspace(0, 1, 3))
   <dense matrix: shape=3x3, dtype=float64
    mat=[[1.    0.882 0.607]
         [0.882 1.    0.882]
         [0.607 0.882 1.   ]]>
    
   >>> k.elwise(np.linspace(0, 1, 3), 0)
   array([[1.        ],
          [0.8824969 ],
          [0.60653066]])

Let ``m`` be a mean, e.g. ``m = TensorProductMean(lambda x: x ** 2)``.

-  ``m(x)`` constructs the *mean vector* for the points in ``x``, which
   will be a *rank-2 column vector*.

Example:

.. code:: python

   >>> m = TensorProductMean(lambda x: x ** 2)

   >>> m(np.linspace(0, 1, 3))
   array([[0.  ],
          [0.25],
          [1.  ]])

Inputs to kernels and means are interpreted in the following way:

-  If the input ``x`` is a *rank-0 tensor*, i.e. a scalar, then ``x``
   refers to a single input location. For example, ``0`` simply refers
   to the sole input location ``0``.

-  If the input ``x`` is a *rank-1 tensor*, i.e. a vector, then every
   element of ``x`` is interpreted as a separate input location. For
   example, ``np.linspace(0, 1, 10)`` generates 10 different input
   locations ranging from ``0`` to ``1``.

-  If the input ``x`` is a *rank-2 tensor*, i.e. a matrix, then every
   *row* of ``x`` is interpreted as a separate input location. In this
   case inputs are multi-dimensional, and the columns correspond to the
   various input dimensions.

-  If the input ``x`` is a *tensor of rank 3 or higher*, then the input
   is interpreted as a batch of matrices where the matrix dimensions are
   the two outermost dimensions.

Example:

.. code:: python

   >>> k = EQ()

   >>> k(0)  # One scalar input
   <dense matrix: batch=(), shape=(1, 1), dtype=float64
    mat=[[1.]]>

   >>> k(np.linspace(0, 1, 3))  # Three single-dimensional inputs
   <dense matrix: batch=(), shape=(3, 3), dtype=float64
    mat=[[1.    0.882 0.607]
         [0.882 1.    0.882]
         [0.607 0.882 1.   ]]>

   >>> k(np.random.randn(3, 2))  # Three two-dimensional inputs
   <dense matrix: batch=(), shape=(3, 3), dtype=float64
    mat=[[1.    0.606 0.399]
         [0.606 1.    0.931]
         [0.399 0.931 1.   ]]>

   >>> k(np.random.randn(2, 3, 2))  # A batch of two times three two-dimensional inputs
   <dense matrix: batch=(2,), shape=(3, 3), dtype=float64
    mat=[[[1.    0.15  0.559]
          [0.15  1.    0.678]
          [0.559 0.678 1.   ]]

         [[1.    0.891 0.627]
          [0.891 1.    0.638]
          [0.627 0.638 1.   ]]]>

AutoGrad, TensorFlow, PyTorch, or JAX? Your Choice!
---------------------------------------------------

.. code:: python

   from mlkernels.autograd import EQ, Linear

.. code:: python

   from mlkernels.tensorflow import EQ, Linear

.. code:: python

   from mlkernels.torch import EQ, Linear

.. code:: python

   from mlkernels.jax import EQ, Linear

Available Kernels
-----------------

See
`here <https://wesselb.github.io/mlkernels/docs/_build/html/readme.html#available-kernels>`__
for a nicely rendered version of the list below.

Constants function as constant kernels. Besides that, the following
kernels are available:

-  ``EQ()``, the exponentiated quadratic:

   .. math::  k(x, y) = \exp\left( -\frac{1}{2}\|x - y\|^2 \right); 

-  ``CEQ(alpha)``, the causal exponentiated quadratic:

   .. math::  k(x, y) = \left(1 - \erf\left( \frac{\alpha}{4}\|x - y\| \right)\right)\exp\left( -\frac{1}{2}\|x - y\|^2 \right); 

-  ``RQ(alpha)``, the rational quadratic:

   .. math::  k(x, y) = \left( 1 + \frac{\|x - y\|^2}{2 \alpha} \right)^{-\alpha}; 

-  ``Matern12()`` or ``Exp()``, the Matern–1/2 kernel:

   .. math::  k(x, y) = \exp\left( -\|x - y\| \right); 

-  ``Matern32()``, the Matern–3/2 kernel:

   .. math::

       k(x, y) = \left(
            1 + \sqrt{3}\|x - y\|
            \right)\exp\left(-\sqrt{3}\|x - y\|\right); 

-  ``Matern52()``, the Matern–5/2 kernel:

   .. math::

       k(x, y) = \left(
            1 + \sqrt{5}\|x - y\| + \frac{5}{3} \|x - y\|^2
           \right)\exp\left(-\sqrt{3}\|x - y\|\right); 

-  ``Linear()``, the linear kernel:

   .. math::  k(x, y) = \langle x, y \rangle; 

-  ``Delta(epsilon=1e-6)``, the Kronecker delta kernel:

   .. math::

       k(x, y) = \begin{cases}
            1 & \text{if } \|x - y\| < \varepsilon, \\
            0 & \text{otherwise};
           \end{cases} 

-  ``DecayingKernel(alpha, beta)``:

   .. math::  k(x, y) = \frac{\|\beta\|^\alpha}{\|x + y + \beta\|^\alpha}; 

-  ``LogKernel()``:

   .. math::  k(x, y) = \frac{\log(1 + \|x - y\|)}{\|x - y\|}; 

-  ``PosteriorKernel(k_ij, k_zi, k_zj, z, K_z)``:

   .. math::  k(x, y) = k_{ij}(x, y) - k_{iz}(x, z) K_z^{-1} k_{zj}(x, y); 

-  ``SubspaceKernel(k_zi, k_zj, z, A)``:

   .. math::  k(x, y) = k_{iz}(x, z) A^{-1} k_{zj}(x, y); 

-  ``TensorProductKernel(f)``:

   .. math::  k(x, y) = f(x)f(y). 

   Adding or multiplying a ``FunctionType`` ``f`` to or with a kernel
   will automatically translate ``f`` to ``TensorProductKernel(f)``. For
   example, ``f * k`` will translate to ``TensorProductKernel(f) * k``,
   and ``f + k`` will translate to ``TensorProductKernel(f) + k``.

Available Means
---------------

Constants function as constant means. Besides that, the following means
are available:

-  ``TensorProductMean(f)``:

   .. math::  m(x) = f(x). 

   Adding or multiplying a ``FunctionType`` ``f`` to or with a mean will
   automatically translate ``f`` to ``TensorProductMean(f)``. For
   example, ``f * m`` will translate to ``TensorProductMean(f) * m``,
   and ``f + m`` will translate to ``TensorProductMean(f) + m``.

Compositional Design
--------------------

-  Add and subtract kernels and means.

   Example:

   .. code:: python

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

-  Multiply kernels and means.

   Example:

   .. code:: python

      >>> EQ() * Matern12()
      EQ() * Matern12()

      >>> 2 * EQ()
      2 * EQ()

      >>> 0 * EQ()
      0

-  Shift kernels and means.

   Definition:

   .. code:: python

      k.shift(c)(x, y) == k(x - c, y - c)

      k.shift(c1, c2)(x, y) == k(x - c1, y - c2)

   Example:

   .. code:: python

      >>> Linear().shift(1)
      Linear() shift 1

      >>> EQ().shift(1, 2)
      EQ() shift (1, 2)

-  Stretch kernels and means.

   Definition:

   .. code:: python

      k.stretch(c)(x, y) == k(x / c, y / c)

      k.stretch(c1, c2)(x, y) == k(x / c1, y / c2)

   Example:

   .. code:: python

      >>> EQ().stretch(2)
      EQ() > 2

      >>> EQ().stretch(2, 3)
      EQ() > (2, 3)

-  Select particular input dimensions of kernels and means.

   Definition:

   .. code:: python

      k.select([0])(x, y) == k(x[:, 0], y[:, 0])

      k.select([0, 1])(x, y) == k(x[:, [0, 1]], y[:, [0, 1]])

      k.select([0], [1])(x, y) == k(x[:, 0], y[:, 1])

      k.select(None, [1])(x, y) == k(x, y[:, 1])

   Example:

   .. code:: python

      >>> EQ().select([0])
      EQ() : [0]

      >>> EQ().select([0, 1])
      EQ() : [0, 1]

      >>> EQ().select([0], [1])
      EQ() : ([0], [1])

      >>> EQ().select(None, [1])
      EQ() : (None, [1])

-  Transform the inputs of kernels and means.

   Definition:

   .. code:: python

      k.transform(f)(x, y) == k(f(x), f(y))

      k.transform(f1, f2)(x, y) == k(f1(x), f2(y))

      k.transform(None, f)(x, y) == k(x, f(y))

   Example:

   .. code:: python

      >>> EQ().transform(f)
      EQ() transform f

      >>> EQ().transform(f1, f2)
      EQ() transform (f1, f2)

      >>> EQ().transform(None, f)
      EQ() transform (None, f)

-  Numerically, but efficiently, take derivatives of kernels and means.
   This currently only works in TensorFlow and does not yet support
   batched inputs.

   Definition:

   .. code:: python

      k.diff(0)(x, y) == d/d(x[:, 0]) d/d(y[:, 0]) k(x, y)

      k.diff(0, 1)(x, y) == d/d(x[:, 0]) d/d(y[:, 1]) k(x, y)

      k.diff(None, 1)(x, y) == d/d(y[:, 1]) k(x, y)

   Example:

   .. code:: python

      >>> EQ().diff(0)
      d(0) EQ()

      >>> EQ().diff(0, 1)
      d(0, 1) EQ()

      >>> EQ().diff(None, 1)
      d(None, 1) EQ()

-  Make kernels periodic. This is not implemented for means.

   Definition:

   .. code:: python

      k.periodic(2 pi / w)(x, y) == k((sin(w * x), cos(w * x)), (sin(w * y), cos(w * y)))

   Example:

   .. code:: python

      >>> EQ().periodic(1)
      EQ() per 1

-  Reverse the arguments of kernels. This does not apply to means.

   Definition:

   .. code:: python

      reversed(k)(x, y) == k(y, x)

   Example:

   .. code:: python

      >>> reversed(Linear())
      Reversed(Linear())

-  Extract terms and factors from sums and products respectively.

   Example:

   .. code:: python

      >>> (EQ() + RQ(0.1) + Linear()).term(1)
      RQ(0.1)

      >>> (2 * EQ() * Linear).factor(0)
      2

   Kernels and means “wrapping” others can be “unwrapped” by indexing
   ``k[0]`` or ``m[0]``.

   Example:

   .. code:: python

      >>> reversed(Linear())
      Reversed(Linear())

      >>> reversed(Linear())[0]
      Linear()

      >>> EQ().periodic(1)
      EQ() per 1

      >>> EQ().periodic(1)[0]
      EQ()

Displaying Kernels and Means
----------------------------

Kernels and means have a ``display`` method. The ``display`` method
accepts a callable formatter that will be applied before any value is
printed. This comes in handy when pretty printing kernels.

Example:

.. code:: python

   >>> print((2.12345 * EQ()).display(lambda x: f"{x:.2f}"))
   2.12 * EQ()

Properties of Kernels and Means
-------------------------------

-  Kernels and means can be equated to check for equality. This will
   attempt basic algebraic manipulations. If the kernels and means are
   not equal *or* equality cannot be proved, then ``False`` is returned.

   Example of equating kernels:

   .. code:: python

      >>>  2 * EQ() == EQ() + EQ()
      True

      >>> EQ() + Matern12() == Matern12() + EQ()
      True

      >>> 2 * Matern12() == EQ() + Matern12()
      False

      >>> EQ() + Matern12() + Linear()  == Linear() + Matern12() + EQ()  # Too hard: cannot prove equality!
      False

-  The stationarity of a kernel ``k`` can always be determined by
   querying ``k.stationary``.

   Example of querying the stationarity:

   .. code:: python

      >>> EQ().stationary
      True

      >>> (EQ() + Linear()).stationary
      False

Structured Matrix Types
-----------------------

MLKernels uses `an extension of
LAB <https://github.com/wesselb/matrix>`__ to accelerate linear algebra
with structured linear algebra primitives.

Example:

.. code:: python

   >>> import lab as B

   >>> k = 2 * Delta()

   >>> x = np.linspace(0, 5, 10)

   >>> from mlkernels import pairwise

   >>> k(x)  # Preserve structure.
   <diagonal matrix: shape=10x10, dtype=float64
    diag=[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]>

   >>> B.dense(k(x))  # Do not preserve structure.
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

These structured matrices are compatible with
`LAB <https://github.com/wesselb/lab>`__: they know how to efficiently
add, multiply, and do other linear algebra operations.

.. code:: python

   >>> import lab as B

   >>> B.matmul(pairwise(k, x), pairwise(k, x))
   <diagonal matrix: shape=10x10, dtype=float64
    diag=[4. 4. 4. 4. 4. 4. 4. 4. 4. 4.]>

As in the above exmaple, you can convert structured primitives to
regular NumPy/TensorFlow/PyTorch/JAX arrays by calling ``B.dense``:

.. code:: python

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

Implementing Your Own Kernel
----------------------------

An example is most helpful:

.. code:: python

   import lab as B
   from algebra.util import identical
   from matrix import Dense
   from plum import dispatch

   from mlkernels import Kernel, pairwise, elwise


   class EQWithLengthScale(Kernel):
       """Exponentiated quadratic kernel with a length scale.

       Args:
           scale (scalar): Length scale of the kernel.
       """

       def __init__(self, scale):
           self.scale = scale

       def _compute(self, dists2):
           # This computes the kernel given squared distances. We use `B` to provide a
           # backend-agnostic implementation.
           return B.exp(-0.5 * dists2 / self.scale ** 2)

       def render(self, formatter):
           # This method determines how the kernel is displayed.
           return f"EQWithLengthScale({formatter(self.scale)})"

       @property
       def _stationary(self):
           # This method can be defined to return `True` to indicate that the kernel is
           # stationary. By default, kernels are assumed to not be stationary.
           return True

       @dispatch
       def __eq__(self, other: "EQWithLengthScale"):
           # If `other` is also a `EQWithLengthScale`, then this method checks whether 
           # `self` and `other` can be treated as identical for the purpose of 
           # algebraic simplifications. In this case, `self` and `other` are identical 
           # for the purpose of algebraic simplification if `self.scale` and
           # `other.scale` are. We use `algebra.util.identical` to check this condition.
           return identical(self.scale, other.scale)


   # It remains to implement pairwise and element-wise computation of the kernel.


   @pairwise.dispatch
   def pairwise(k: EQWithLengthScale, x: B.Numeric, y: B.Numeric):
       return Dense(k._compute(B.pw_dists2(x, y)))


   @elwise.dispatch
   def elwise(k: EQWithLengthScale, x: B.Numeric, y: B.Numeric):
       return k._compute(B.ew_dists2(x, y))

.. code:: python

   >>> k = EQWithLengthScale(2)

   >>> k
   EQWithLengthScale(2)

   >>> k == EQWithLengthScale(2)
   True

   >>> 2 * k == k + EQWithLengthScale(2)
   True

   >>> k == Linear()
   False

   >>> k_composite = (2 * k + Linear()) * RQ(2.0)

   >>> k_composite
   (2 * EQWithLengthScale(2) + Linear()) * RQ(2.0)

   >>> k_composite(np.linspace(0, 1, 3))
   <dense matrix: shape=3x3, dtype=float64
    mat=[[2.    1.717 1.13 ]
         [1.717 2.25  2.16 ]
         [1.13  2.16  3.   ]]>

Of course, in practice we do not need to implement variants of kernels
which include length scales, because we always adjust the length scale
by stretching a base kernel:

.. code:: python

   >>> EQ().stretch(2)(np.linspace(0, 1, 3))
   <dense matrix: shape=3x3, dtype=float64
    mat=[[1.    0.969 0.882]
         [0.969 1.    0.969]
         [0.882 0.969 1.   ]]>

   >>> EQWithLengthScale(2)(np.linspace(0, 1, 3))
   <dense matrix: shape=3x3, dtype=float64
    mat=[[1.    0.969 0.882]
         [0.969 1.    0.969]
         [0.882 0.969 1.   ]]>

.. |CI| image:: https://github.com/wesselb/mlkernels/workflows/CI/badge.svg
   :target: https://github.com/wesselb/mlkernels/actions?query=workflow%3ACI
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/mlkernels/badge.svg?branch=main
   :target: https://coveralls.io/github/wesselb/mlkernels?branch=main
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://wesselb.github.io/mlkernels
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
