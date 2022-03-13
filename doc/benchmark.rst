.. include:: bibliography.txt

Benchmarks
==========

Cost function benchmark
-----------------------

iminuit comes with a couple of builtin cost functions, but also makes it easy for users to implement their own. One motivation to use your own cost function is to get the best possible performance. If your fit takes more than a minute (possible if you do an unbinned fit on a large sample or if you have a model with many parameters), you may be looking for more performance.

Easy steps to make a fit faster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
These steps require little effort in terms of changing your code and should be tried first.

- If you have a large sample, use a binned fit instead of an unbinned fit.
- Use a faster implementation of the probability density, e.g. from `numba_stats`_.
- For an unbinned fit, use the ``logpdf`` instead of the ``pdf``, if possible.
- Use `numba`_ to compile the cost function; perhaps with options `parallel=True` and `fastmath=True` to parallelize computation (but see benchmark results).

Benchmark of unbinned maximum-likelihood fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/cost_bench.svg

We fitted a normal distribution to a sample of varying size. The builtin :class:`iminuit.cost.UnbinnedNLL` cost function is compared with a simple direct implementation

.. code-block:: python

    import numpy as np
    from numba_stats import norm

    x = ... # data

    def cost_log_pdf(mu, sigma):
        return -np.sum(np.log(norm.pdf(x, mu, sigma)))

    def cost_logpdf(mu, sigma):
        return -np.sum(norm.logpdf(x, mu, sigma))

We need to compute the log-probability for maximum-likelihood. For many common distributions (those of the exponential family), it is possible to implement ``logpdf`` more efficiently than ``log(pdf)``, which means that the second form can be computed faster.

Conclusions
^^^^^^^^^^^

- :class:`iminuit.cost.UnbinnedNLL` is as fast as a simple direct implementation.
- Using ``logpdf`` instead of ``log(pdf)`` is a big gain for a normal distribution, it is about five times faster. This is because the compiler can use SIMD instructions more efficently and omit the extra calls to the ``log`` and ``exp`` functions. Function calls that cannot be inlined in a hot loop are detrimental for performance.
- Using `numba`_ is not a huge gain if the ``pdf`` or ``logpdf`` is already compiled.
- Using `numba`_ to generate a C function pointer for the cost function is a moderate speed-up for small samples, but negligible for large samples.
- Using `numba`_ with the options ``parallel`` and ``fastmath`` individually does not gain anything, while using both at the same time is a moderate gain of about 30 %.
- Hand-tuning the cost function can give better parallel performance than using the ``logpdf`` implementation from `numba_stats`_.

Minuit2 vs other optimisers
---------------------------

We compare the performance of Minuit2 (the code that is wrapped by iminuit) with other minimizers available in Python. We compare Minuit with the strategy settings 0 to 2 with several algorithms implemented in the `nlopt`_ library and `scipy.optimize`_. .

All algorithms minimize a dummy cost function

.. code-block:: python

    def cost_function(par):
        z = (y - par)
        return sum(z ** 2 + 0.1 * z ** 4)

where ``y`` are samples from a normal distribution scaled by a factor of 5. The second term in the sum assures that the cost function is non-linear in the parameters and not too easy to minimize. No analytical gradient is provided, since this is the most common way how minimizers are used for small problems.

The cost function is minimized for a variable number of parameters from 1 to 100. The number of function calls is recorded and the largest absolute deviation of the solution from the truth. The fit is repeated 100 times for each configuration to reduce the scatter of the results, and the medians of these trails are computed.

The scipy algorithms are run with default settings. For nlopt, a stopping criterion must be selected. We stop when the absolute variation in the parameters is becomes less than 1e-3.

The results are shown in the following three plots. The best algorithms require the fewest function calls to achieve the highest accuracy.

.. image:: _static/bench.svg

.. image:: _static/bench2d.svg

Shown in the first plot is the number of calls to the cost function divided by the number of parameters. Smaller is better. Note that the algorithms achieve varying levels of accuracy, therefore this plot alone cannot show which algorithm is best. Shown in the second plot is the accuracy of  the solution when the minimizer is stopped. The stopping criteria vary from algorithm to algorithm.

The third plot combines both and shows accuracy vs. number of function calls per parameter for fits with 2, 10, and 100 parameters, as indicated by the marker size. Since higher accuracy can be achieved with more function evaluations, the most efficient algorithms follow diagonal lines from the top left to the bottom right in the lower left edge of the plot.

Discussion
^^^^^^^^^^

The following discussion should be taken with a grain of salt, since experiments have shown that the results depend on the minimisation problem. Also not tested here is the robustness of these algorithms when the cost function is more complicated or not perfectly analytical.

* Minuit2 is average in terms of accuracy vs. efficiency. Strategy 0 is pretty efficient for fits with less than 10 parameters. The typical accuracy achieved in this problem is about 0.1 to 1 %. Experiments with other cost functions have shown that the accuracy strongly depends on how parabolic the function is near the minimum. Minuit2 seems to stop earlier when the function is not parabolic, achieving lower accuracy.

* The scipy methods Powell and CG are the most efficient algorithms on this problem. Both are more accurate than Minuit2 and CG uses much fewer function evaluations, especially in fits with many parameters. Powell uses a similar amount of function calls as Minuit2, but achieves accuracy at the level of 1e-12, while Minuit2 achieves 1e-3 to 1e-6.

* An algorithm with a constant curve in the first plot has a computation time which scales linearly in the number of parameters. This is the case for the Powell and CG methods, but Minuit2 and others that compute an approximation to the Hesse matrix scale quadratically.

* The Nelder-Mead algorithm shows very bad performance with weird features. It should not be used. On the other hand, the SBPLX algorithm does fairly well although it is a variant of the same idea.

In summary, Minuit2 (and therefore iminuit) is a good allrounder, but it is not outstanding in terms of convergence rate or accuracy. Using strategy 0 seem safe to use: it speeds up the convergence without reducing the accuracy of the result. If fast convergence is critical, it can be useful to try others minimisers. The scipy minimisers are accessible through :meth:`iminuit.Minuit.scipy`.
