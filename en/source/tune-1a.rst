.. _tune-1a:

******************************
Alternative solution (tune-1a)
******************************

An equivalent SIMD conversion for the i-loop without using
a compiler-specific directive is as follows.
We split the loop into several parts and change the order of loops so that the i-loop comes to the innermost.
Although the editing of the code is extensive, similar methods can be used in other applications.

.. literalinclude:: ../../cpp/axhelm-1a.cpp
   :language: cpp
   :tab-width: 4
   :lines: 79-
   :linenos:
   :emphasize-lines: 3,9,16,34,40,46
   :caption: axhelm-1a.cpp
   :lineno-match:

For both i-loops, the loop body is split into three parts, 
and the middle part comes to the inner of the m-loop.
Note that the datatype of intermediate variables ``qr, qs, qt``
is changed from a scaler to an array with size 8.

Not the same as :ref:`tune-1<tune-1>`,
but this version yields a performance of **321 GFLOP/s**.
