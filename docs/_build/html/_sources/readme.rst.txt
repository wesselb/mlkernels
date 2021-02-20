`MLKernels <http://github.com/wesselb/mlkernels>`__
===================================================

|CI| |Coverage Status| |Latest Docs| |Code style: black|

Kernels, the machine learning ones

Installation
------------

See `the instructions
here <https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc>`__.
Then simply

::

    pip install mlkernels

Usage
-----

Inputs to kernels, henceforth referred to simply as *inputs*, must be of
one of the following three forms:

-  If the input ``x`` is a *rank 0 tensor*, i.e. a scalar, then ``x``
   refers to a single input location. For example, ``0`` simply refers
   to the sole input location ``0``.

-  If the input ``x`` is a *rank 1 tensor*, then every element of ``x``
   is interpreted as a separate input location. For example,
   ``np.linspace(0, 1, 10)`` generates 10 different input locations
   ranging from ``0`` to ``1``.

-  If the input ``x`` is a *rank 2 tensor*, then every *row* of ``x`` is
   interpreted as a separate input location. In this case inputs are
   multi-dimensional, and the columns correspond to the various input
   dimensions.

If ``k`` is a kernel, say ``k = EQ()``, then ``k(x, y)`` constructs the
*kernel matrix* for all pairs of points between ``x`` and ``y``.
``k(x)`` is shorthand for ``k(x, x)``. Furthermore, ``k.elwise(x, y)``
constructs the *kernel vector* pairing the points in ``x`` and ``y``
element wise, which will be a *rank 2 column vector*.

Example:

.. code:: python

    >>> EQ()(np.linspace(0, 1, 3))
    array([[1.        , 0.8824969 , 0.60653066],
           [0.8824969 , 1.        , 0.8824969 ],
           [0.60653066, 0.8824969 , 1.        ]])
     
    >>> EQ().elwise(np.linspace(0, 1, 3), 0)
    array([[1.        ],
           [0.8824969 ],
           [0.60653066]])

Available Kernels
~~~~~~~~~~~~~~~~~

Constants function as constant kernels. Besides that, the following
kernels are available:

-  ``EQ()``, the exponentiated quadratic:

   .. math::  k(x, y) = \exp\left( -\frac{1}{2}\|x - y\|^2 \right); 

-  ``RQ(alpha)``, the rational quadratic:

   .. math::  k(x, y) = \left( 1 + \frac{\|x - y\|^2}{2 \alpha} \right)^{-\alpha}; 

-  ``Exp()`` or ``Matern12()``, the exponential kernel:

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

-  ``Delta()``, the Kronecker delta kernel:

   .. math::

       k(x, y) = \begin{cases}
          1 & \text{if } x = y, \\
          0 & \text{otherwise};
         \end{cases} 

-  ``DecayingKernel(alpha, beta)``:

   .. math::  k(x, y) = \frac{\|\beta\|^\alpha}{\|x + y + \beta\|^\alpha}; 

-  ``LogKernel()``:

   .. math::  k(x, y) = \frac{\log(1 + \|x - y\|)}{\|x - y\|}; 

-  ``TensorProductKernel(f)``:

   .. math::  k(x, y) = f(x)f(y). 

   Adding or multiplying a ``FunctionType`` ``f`` to or with a kernel
   will automatically translate ``f`` to ``TensorProductKernel(f)``. For
   example, ``f * k`` will translate to ``TensorProductKernel(f) * k``,
   and ``f + k`` will translate to ``TensorProductKernel(f) + k``.

Compositional Design
~~~~~~~~~~~~~~~~~~~~

-  Add and subtract kernels.

   Example:

   .. code:: python

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

-  Multiply kernels.

   Example:

   .. code:: python

       >>> EQ() * Exp()
       EQ() * Exp()

       >>> 2 * EQ()
       2 * EQ()

       >>> 0 * EQ()
       0

-  Shift kernels.

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

-  Stretch kernels.

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

   The ``>`` operator is implemented to provide a shorthand for
   stretching:

   .. code:: python

       >>> EQ() > 2
       EQ() > 2

-  Select particular input dimensions of kernels.

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

-  Transform the inputs of kernels.

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

-  Numerically, but efficiently, take derivatives of kernels. This
   currently only works in TensorFlow.

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

-  Make kernels periodic.

   Definition:

   .. code:: python

       k.periodic(2 pi / w)(x, y) == k((sin(w * x), cos(w * x)), (sin(w * y), cos(w * y)))

   Example:

   .. code:: python

       >>> EQ().periodic(1)
       EQ() per 1

-  Reverse the arguments of kernels.

   Definition:

   .. code:: python

       reversed(k)(x, y) == k(y, x)

   Example:

   .. code:: python

       >>> reversed(Linear())
       Reversed(Linear())

-  Extract terms and factors from sums and products respectively of
   kernels.

   Example:

   .. code:: python

       >>> (EQ() + RQ(0.1) + Linear()).term(1)
       RQ(0.1)

       >>> (2 * EQ() * Linear).factor(0)
       2

   Kernels and means "wrapping" others can be "unwrapped" by indexing
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

Displaying Kernels
^^^^^^^^^^^^^^^^^^

Kernels and means have a ``display`` method. The ``display`` method
accepts a callable formatter that will be applied before any value is
printed. This comes in handy when pretty printing kernels.

Example:

.. code:: python

    >>> print((2.12345 * EQ()).display(lambda x: f"{x:.2f}"))
    2.12 * EQ(), 0

Properties of Kernels
~~~~~~~~~~~~~~~~~~~~~

-  Kernels can be equated to check for equality. This will attempt basic
   algebraic manipulations. If the means and kernels are not equal *or*
   equality cannot be proved, ``False`` is returned.

   Example of equating kernels:

   .. code:: python

       >>>  2 * EQ() == EQ() + EQ()
       True

       >>> EQ() + Exp() == Exp() + EQ()
       True

       >>> 2 * Exp() == EQ() + Exp()
       False

       >>> EQ() + Exp() + Linear()  == Linear() + Exp() + EQ()  # Too hard: cannot prove equality!
       False

-  The stationarity of a kernel ``k`` can always be determined by
   querying ``k.stationary``.

   Example of querying the stationarity:

   .. code:: python

       >>> EQ().stationary
       True

       >>> (EQ() + Linear()).stationary
       False

.. |CI| image:: https://github.com/wesselb/mlkernels/workflows/CI/badge.svg
   :target: https://github.com/wesselb/mlkernels/actions?query=workflow%3ACI
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/mlkernels/badge.svg?branch=main
   :target: https://coveralls.io/github/wesselb/mlkernels?branch=main
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://wesselb.github.io/mlkernels
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
