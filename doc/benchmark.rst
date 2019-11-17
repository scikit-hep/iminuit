.. include:: references.txt

.. _benchmark:

Benchmark
=========

We compare the performane of Minuit2 (which is wrapped by iminuit) with other minimizers available in Python. We compare Minuit with the strategy settings 0 to 2 with several algorithms implemented in the `nlopt`_ library and `scipy.optimize`_. . 

Setup
-----

All algorithms minimize a dummy score function

.. code-block:: python

    def score_function(par):
        return log(sum((y - par) ** 2) + 1)

where `y` are samples from a normal distribution. The logarithm makes sure that the score function is non-linear in the parameters and not too easy to minimize. No analytical gradient is provided for the algorithms, since this is the most common way how minimizers are used.

The score function is minimized for a variable number of parameters from 2 to 100. The number of function calls is recorded and the largest absolute deviation of the solution from the truth. The fit is repeated 100 times for each configuration to reduce the scatter of the results, and the medians of these trails are computed.

The scipy algorithms are run with default settings. For nlopt, a stopping criterion must be selected. We stop when the absolute variation in the parameters is becomes less than 1e-3. The criterion was tuned to offer an accuracy comparable to Minuit2, which simplifies the comparison.

Results
-------
The results are shown in the following plot.

.. image:: bench.svg

Convergence rate
~~~~~~~~~~~~~~~~

Shown on the left is the number of calls to the score function divided by the number of parameters. Smaller is better.

* Minuit2 is not the fastest algorithm, but hopefully makes up for that in robustness. Four algorithms converge twice as fast as Minuit with strategy 0: the nlopt algorithms BOBYQA and NEWUOA, and the scipy algorithms BFGS und CG.
* An algorithm with a constant curve has a computation time which scales linearly in the number of parameters. This is the case for most algorithms, while iminuit with strategy 0 seems to rise and then reach a plateau, while strategy 1 and 2 and nlopt's PRAXIS algorithm scale quadratically. This difference in scaling becomes very important when the number of parameters is larger than 10. In this case, using strategy 0 is recommended.
* The Nelder-Mead algorithm shows very bad performance and has a weird step when the number of fitted parameters is 5 and higher. This is somewhat surprising since nlopt's SBPLX is a variant of the same algorithm with much better performance, which even rivals that of Minuit2 with strategy 1 when the number of parameters is larger than 50.

Accuracy
~~~~~~~~

Shown on the right is the other performance parameter of interest: how accurate is the solution when the minimizer is stopped.

* Minuit2 shows an accuracy of 1e-4 to 1e-5 of a standard deviation, which is more than enough. Strategy 0 and strategy 1 are identical in accuracy. Strategy 2 is a factor 5 to 10 more accurate.
* Several algorithms provide more accurate solutions. The two most accurate are scipy's CG and Powell. PRAXIS from nlopt is very accurate when the number of parameters is less than 4 (and in this case it also is competitively fast).
* Out of the four algorithms which converge faster than Minuit2 with strategy 0, BOBYQA and NEWUOA from nlopt are equally accurate. This is by design of the benchmark, the stopping criterion in nlopt was chosen to mimic Minuit2's accuracy. Scipy's BFGS and CG methods are several orders more accurate than Minuit2's solution.
* The solution of scipy's Powell algorithm is also very accurate, but the convergence rate is not competitive.
* Scipy's Nelder-Mead implementation not only has a poor converge rate, its accuracy is also terrible. This is not a general problem with the SIMPLEX algorithm, since nlopt's SBPLX shows good converge at an accuracy that is comparable to Minuit2. It looks like the implementation used by scipy is far from optimal.

Conclusions
-----------

Scipy's BFGS and CG and nlopt's BOBYQA and NEWUOA algorithms converge faster than Minuit2 when no analytical gradients are provided, at equal or better accuracy. Not tested here is the robustness of these algorithms when the score function is more complicated or not perfectly analytical. This may turn the tables in favor of Minuit2, since robustness was very important for the authors, while the other algorithms are generally optimized to give the best convergence rates for ideal score functions.