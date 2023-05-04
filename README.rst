.. |iminuit| image:: doc/_static/iminuit_logo.svg
   :alt: iminuit

|iminuit|
=========

.. version-marker-do-not-remove

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :target: https://scikit-hep.org
.. image:: https://img.shields.io/pypi/v/iminuit.svg
   :target: https://pypi.org/project/iminuit
.. image:: https://img.shields.io/conda/vn/conda-forge/iminuit.svg
   :target: https://github.com/conda-forge/iminuit-feedstock
.. image:: https://coveralls.io/repos/github/scikit-hep/iminuit/badge.svg?branch=develop
   :target: https://coveralls.io/github/scikit-hep/iminuit?branch=develop
.. image:: https://readthedocs.org/projects/iminuit/badge/?version=latest
   :target: https://iminuit.readthedocs.io/en/stable
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3949207.svg
   :target: https://doi.org/10.5281/zenodo.3949207
.. image:: https://img.shields.io/badge/ascl-2108.024-blue.svg?colorB=262255
   :target: https://ascl.net/2108.024
   :alt: ascl:2108.024
.. image:: https://img.shields.io/gitter/room/Scikit-HEP/iminuit
   :target: https://gitter.im/Scikit-HEP/iminuit
.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/scikit-hep/iminuit/develop?filepath=doc%2Ftutorial

*iminuit* is a Jupyter-friendly Python interface for the *Minuit2* C++ library maintained by CERN's ROOT team.

Minuit was designed to minimise statistical cost functions, for likelihood and least-squares fits of parametric models to data. It provides the best-fit parameters and error estimates from likelihood profile analysis.

The iminuit package comes with additional features:

- Builtin cost functions for statistical fits

  - Binned and unbinned maximum-likelihood
  - `Template fits with error propagation <https://doi.org/10.1140/epjc/s10052-022-11019-z>`_
  - Least-squares (optionally robust to outliers)
  - Gaussian penalty terms for parameters
  - Cost functions can be combined by adding them: ``total_cost = cost_1 + cost_2``
  - Visualization of the fit in Jupyter notebooks
- Support for SciPy minimisers as alternatives to Minuit's Migrad algorithm (optional)
- Support for Numba accelerated functions (optional)

Dependencies
------------

*iminuit* is (and always will be) a lean package which only depends on ``numpy``, but additional features are enabled if the following optional packages are installed.

- ``matplotlib``: Visualization of fitted model for builtin cost functions
- ``ipywidgets``: Interactive fitting, see example below (also requires ``matplotlib``)
- ``scipy``: Compute Minos intervals for arbitrary confidence levels
- ``unicodeitplus``: Render names of model parameters in simple LaTeX as Unicode

Documentation
-------------

Checkout our large and comprehensive list of `tutorials`_ that take you all the way from beginner to power user. For help and how-to questions, please use the `discussions`_ on GitHub or `gitter`_.

**Lecture by Glen Cowan**

`In the exercises to his lecture for the KMISchool 2022 <https://github.com/KMISchool2022>`_, Glen Cowan shows how to solve statistical problems in Python with iminuit. You can find the lectures and exercises on the Github page, which covers both frequentist and Bayesian methods.

`Glen Cowan <https://scholar.google.com/citations?hl=en&user=ljQwt8QAAAAJ&view_op=list_works>`_ is a known for his papers and international lectures on statistics in particle physics, as a member of the Particle Data Group, and as author of the popular book `Statistical Data Analysis <https://www.pp.rhul.ac.uk/~cowan/sda/>`_.

In a nutshell
-------------

iminuit can be used with a user-provided cost functions in form of a negative log-likelihood function or least-squares function. Standard functions are included in ``iminuit.cost``, so you don't have to write them yourself. The following example shows how to perform an unbinned maximum likelihood fit.

.. code:: ipython3

    import numpy as np
    from iminuit import Minuit
    from iminuit.cost import UnbinnedNLL
    from scipy.stats import norm

    x = norm.rvs(size=1000, random_state=1)

    def pdf(x, mu, sigma):
        return norm.pdf(x, mu, sigma)

    # Negative unbinned log-likelihood, you can write your own
    cost = UnbinnedNLL(x, pdf)

    m = Minuit(cost, mu=0, sigma=1)
    m.limits["sigma"] = (0, np.inf)
    m.migrad()  # find minimum
    m.hesse()   # compute uncertainties




.. raw:: html

    <table>
        <tr>
            <th colspan="5" style="text-align:center" title="Minimizer"> Migrad </th>
        </tr>
        <tr>
            <td colspan="2" style="text-align:left" title="Minimum value of function"> FCN = 2800 </td>
            <td colspan="3" style="text-align:center" title="Total number of function and (optional) gradient evaluations"> Nfcn = 34 </td>
        </tr>
        <tr>
            <td colspan="2" style="text-align:left" title="Estimated distance to minimum and goal"> EDM = 2.01e-07 (Goal: 0.0002) </td>
            <td colspan="3" style="text-align:center" title="Total run time of algorithms">  </td>
        </tr>
        <tr>
            <td colspan="2" style="text-align:center;background-color:#92CCA6;color:black"> Valid Minimum </td>
            <td colspan="3" style="text-align:center;background-color:#92CCA6;color:black"> No Parameters at limit </td>
        </tr>
        <tr>
            <td colspan="2" style="text-align:center;background-color:#92CCA6;color:black"> Below EDM threshold (goal x 10) </td>
            <td colspan="3" style="text-align:center;background-color:#92CCA6;color:black"> Below call limit </td>
        </tr>
        <tr>
            <td style="text-align:center;background-color:#92CCA6;color:black"> Covariance </td>
            <td style="text-align:center;background-color:#92CCA6;color:black"> Hesse ok </td>
            <td style="text-align:center;background-color:#92CCA6;color:black" title="Is covariance matrix accurate?"> Accurate </td>
            <td style="text-align:center;background-color:#92CCA6;color:black" title="Is covariance matrix positive definite?"> Pos. def. </td>
            <td style="text-align:center;background-color:#92CCA6;color:black" title="Was positive definiteness enforced by Minuit?"> Not forced </td>
        </tr>
    </table><table>
        <tr>
            <td></td>
            <th title="Variable name"> Name </th>
            <th title="Value of parameter"> Value </th>
            <th title="Hesse error"> Hesse Error </th>
            <th title="Minos lower error"> Minos Error- </th>
            <th title="Minos upper error"> Minos Error+ </th>
            <th title="Lower limit of the parameter"> Limit- </th>
            <th title="Upper limit of the parameter"> Limit+ </th>
            <th title="Is the parameter fixed in the fit"> Fixed </th>
        </tr>
        <tr>
            <th> 0 </th>
            <td> mu </td>
            <td> 0.039 </td>
            <td> 0.031 </td>
            <td>  </td>
            <td>  </td>
            <td>  </td>
            <td>  </td>
            <td>  </td>
        </tr>
        <tr>
            <th> 1 </th>
            <td> sigma </td>
            <td> 0.981 </td>
            <td> 0.022 </td>
            <td>  </td>
            <td>  </td>
            <td> 0 </td>
            <td>  </td>
            <td>  </td>
        </tr>
    </table><table>
        <tr>
            <td></td>
            <th> mu </th>
            <th> sigma </th>
        </tr>
        <tr>
            <th> mu </th>
            <td> 0.000962 </td>
            <td style="background-color:rgb(250,250,250);color:black"> 0 </td>
        </tr>
        <tr>
            <th> sigma </th>
            <td style="background-color:rgb(250,250,250);color:black"> 0 </td>
            <td> 0.000481 </td>
        </tr>
    </table><?xml version="1.0" encoding="utf-8" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
      "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
    <svg xmlns:xlink="http://www.w3.org/1999/xlink" width="360pt" height="288pt" viewBox="0 0 360 288" xmlns="http://www.w3.org/2000/svg" version="1.1">
     <metadata>
      <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
       <cc:Work>
        <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
        <dc:date>2023-05-04T12:00:21.519826</dc:date>
        <dc:format>image/svg+xml</dc:format>
        <dc:creator>
         <cc:Agent>
          <dc:title>Matplotlib v3.7.1, https://matplotlib.org/</dc:title>
         </cc:Agent>
        </dc:creator>
       </cc:Work>
      </rdf:RDF>
     </metadata>
     <defs>
      <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
     </defs>
     <g id="figure_1">
      <g id="patch_1">
       <path d="M 0 288
    L 360 288
    L 360 0
    L 0 0
    z
    " style="fill: #ffffff"/>
      </g>
      <g id="axes_1">
       <g id="patch_2">
        <path d="M 22.72524 268.321635
    L 356.99976 268.321635
    L 356.99976 3.153129
    L 22.72524 3.153129
    z
    " style="fill: #ffffff"/>
       </g>
       <g id="PolyCollection_1">
        <path d="M 37.919536 255.032583
    L 37.919536 256.268521
    L 47.415972 256.268521
    L 56.912407 256.268521
    L 61.660624 256.268521
    L 66.408842 256.268521
    L 71.15706 256.268521
    L 75.905277 256.268521
    L 80.653495 256.268521
    L 85.401713 256.268521
    L 90.14993 256.268521
    L 94.898148 256.268521
    L 99.646365 256.268521
    L 104.394583 256.268521
    L 109.142801 256.268521
    L 113.891018 256.268521
    L 123.387453 256.268521
    L 132.883889 256.268521
    L 137.632106 256.268521
    L 142.380324 256.268521
    L 147.128541 256.268521
    L 151.876759 256.268521
    L 154.250868 256.268521
    L 156.624977 256.268521
    L 158.999086 256.268521
    L 161.373194 256.268521
    L 163.747303 256.268521
    L 166.121412 256.268521
    L 168.495521 256.268521
    L 170.86963 256.268521
    L 173.243738 256.268521
    L 175.617847 256.268521
    L 177.991956 256.268521
    L 180.366065 256.268521
    L 182.740174 256.268521
    L 185.114282 256.268521
    L 187.488391 256.268521
    L 189.8625 256.268521
    L 194.610718 256.268521
    L 199.358935 256.268521
    L 204.107153 256.268521
    L 208.85537 256.268521
    L 218.351806 256.268521
    L 227.848241 256.268521
    L 232.596459 256.268521
    L 237.344676 256.268521
    L 242.092894 256.268521
    L 246.841111 256.268521
    L 251.589329 256.268521
    L 256.337547 256.268521
    L 261.085764 256.268521
    L 265.833982 256.268521
    L 270.582199 256.268521
    L 275.330417 256.268521
    L 280.078635 256.268521
    L 284.826852 256.268521
    L 294.323287 256.268521
    L 303.819723 256.268521
    L 322.812593 256.268521
    L 341.805464 256.268521
    L 341.805464 256.207833
    L 341.805464 256.207833
    L 322.812593 255.941147
    L 303.819723 254.82209
    L 294.323287 253.447424
    L 284.826852 251.034118
    L 280.078635 249.270662
    L 275.330417 247.029087
    L 270.582199 244.22072
    L 265.833982 240.753498
    L 261.085764 236.536148
    L 256.337547 231.48355
    L 251.589329 225.523148
    L 246.841111 218.602189
    L 242.092894 210.695363
    L 237.344676 201.812345
    L 232.596459 192.004574
    L 227.848241 181.37058
    L 218.351806 158.269548
    L 208.85537 134.284741
    L 204.107153 122.696435
    L 199.358935 111.820511
    L 194.610718 101.995621
    L 189.8625 93.545142
    L 187.488391 89.927852
    L 185.114282 86.759627
    L 182.740174 84.068938
    L 180.366065 81.880255
    L 177.991956 80.213677
    L 175.617847 79.084622
    L 173.243738 78.50359
    L 170.86963 78.476004
    L 168.495521 79.00212
    L 166.121412 80.07703
    L 163.747303 81.69073
    L 161.373194 83.828283
    L 158.999086 86.470044
    L 156.624977 89.591963
    L 154.250868 93.165947
    L 151.876759 97.160284
    L 147.128541 106.267909
    L 142.380324 116.607367
    L 137.632106 127.846219
    L 132.883889 139.644636
    L 123.387453 163.622174
    L 113.891018 186.252407
    L 109.142801 196.529761
    L 104.394583 205.930443
    L 99.646365 214.37766
    L 94.898148 221.83954
    L 90.14993 228.32306
    L 85.401713 233.86684
    L 80.653495 238.533481
    L 75.905277 242.402049
    L 71.15706 245.561171
    L 66.408842 248.103068
    L 61.660624 250.118725
    L 56.912407 251.694234
    L 47.415972 253.830735
    L 37.919536 255.032583
    z
    " clip-path="url(#pb98193418b)" style="fill: #1f77b4"/>
       </g>
       <g id="matplotlib.axis_1">
        <g id="xtick_1">
         <g id="line2d_1">
          <defs>
           <path id="mcdbe50f481" d="M 0 0
    L 0 3.5
    " style="stroke: #000000; stroke-width: 0.8"/>
          </defs>
          <g>
           <use xlink:href="#mcdbe50f481" x="40.249454" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_1">
          <!-- −3 -->
          <g transform="translate(32.87836 282.920073) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-2212" d="M 678 2272
    L 4684 2272
    L 4684 1741
    L 678 1741
    L 678 2272
    z
    " transform="scale(0.015625)"/>
            <path id="DejaVuSans-33" d="M 2597 2516
    Q 3050 2419 3304 2112
    Q 3559 1806 3559 1356
    Q 3559 666 3084 287
    Q 2609 -91 1734 -91
    Q 1441 -91 1130 -33
    Q 819 25 488 141
    L 488 750
    Q 750 597 1062 519
    Q 1375 441 1716 441
    Q 2309 441 2620 675
    Q 2931 909 2931 1356
    Q 2931 1769 2642 2001
    Q 2353 2234 1838 2234
    L 1294 2234
    L 1294 2753
    L 1863 2753
    Q 2328 2753 2575 2939
    Q 2822 3125 2822 3475
    Q 2822 3834 2567 4026
    Q 2313 4219 1838 4219
    Q 1578 4219 1281 4162
    Q 984 4106 628 3988
    L 628 4550
    Q 988 4650 1302 4700
    Q 1616 4750 1894 4750
    Q 2613 4750 3031 4423
    Q 3450 4097 3450 3541
    Q 3450 3153 3228 2886
    Q 3006 2619 2597 2516
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-2212"/>
           <use xlink:href="#DejaVuSans-33" x="83.789062"/>
          </g>
         </g>
        </g>
        <g id="xtick_2">
         <g id="line2d_2">
          <g>
           <use xlink:href="#mcdbe50f481" x="83.585167" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_2">
          <!-- −2 -->
          <g transform="translate(76.214073 282.920073) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-32" d="M 1228 531
    L 3431 531
    L 3431 0
    L 469 0
    L 469 531
    Q 828 903 1448 1529
    Q 2069 2156 2228 2338
    Q 2531 2678 2651 2914
    Q 2772 3150 2772 3378
    Q 2772 3750 2511 3984
    Q 2250 4219 1831 4219
    Q 1534 4219 1204 4116
    Q 875 4013 500 3803
    L 500 4441
    Q 881 4594 1212 4672
    Q 1544 4750 1819 4750
    Q 2544 4750 2975 4387
    Q 3406 4025 3406 3419
    Q 3406 3131 3298 2873
    Q 3191 2616 2906 2266
    Q 2828 2175 2409 1742
    Q 1991 1309 1228 531
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-2212"/>
           <use xlink:href="#DejaVuSans-32" x="83.789062"/>
          </g>
         </g>
        </g>
        <g id="xtick_3">
         <g id="line2d_3">
          <g>
           <use xlink:href="#mcdbe50f481" x="126.92088" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_3">
          <!-- −1 -->
          <g transform="translate(119.549786 282.920073) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-31" d="M 794 531
    L 1825 531
    L 1825 4091
    L 703 3866
    L 703 4441
    L 1819 4666
    L 2450 4666
    L 2450 531
    L 3481 531
    L 3481 0
    L 794 0
    L 794 531
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-2212"/>
           <use xlink:href="#DejaVuSans-31" x="83.789062"/>
          </g>
         </g>
        </g>
        <g id="xtick_4">
         <g id="line2d_4">
          <g>
           <use xlink:href="#mcdbe50f481" x="170.256593" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_4">
          <!-- 0 -->
          <g transform="translate(167.075343 282.920073) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-30" d="M 2034 4250
    Q 1547 4250 1301 3770
    Q 1056 3291 1056 2328
    Q 1056 1369 1301 889
    Q 1547 409 2034 409
    Q 2525 409 2770 889
    Q 3016 1369 3016 2328
    Q 3016 3291 2770 3770
    Q 2525 4250 2034 4250
    z
    M 2034 4750
    Q 2819 4750 3233 4129
    Q 3647 3509 3647 2328
    Q 3647 1150 3233 529
    Q 2819 -91 2034 -91
    Q 1250 -91 836 529
    Q 422 1150 422 2328
    Q 422 3509 836 4129
    Q 1250 4750 2034 4750
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-30"/>
          </g>
         </g>
        </g>
        <g id="xtick_5">
         <g id="line2d_5">
          <g>
           <use xlink:href="#mcdbe50f481" x="213.592306" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_5">
          <!-- 1 -->
          <g transform="translate(210.411056 282.920073) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-31"/>
          </g>
         </g>
        </g>
        <g id="xtick_6">
         <g id="line2d_6">
          <g>
           <use xlink:href="#mcdbe50f481" x="256.928019" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_6">
          <!-- 2 -->
          <g transform="translate(253.746769 282.920073) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-32"/>
          </g>
         </g>
        </g>
        <g id="xtick_7">
         <g id="line2d_7">
          <g>
           <use xlink:href="#mcdbe50f481" x="300.263732" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_7">
          <!-- 3 -->
          <g transform="translate(297.082482 282.920073) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-33"/>
          </g>
         </g>
        </g>
        <g id="xtick_8">
         <g id="line2d_8">
          <g>
           <use xlink:href="#mcdbe50f481" x="343.599445" y="268.321635" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_8">
          <!-- 4 -->
          <g transform="translate(340.418195 282.920073) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-34" d="M 2419 4116
    L 825 1625
    L 2419 1625
    L 2419 4116
    z
    M 2253 4666
    L 3047 4666
    L 3047 1625
    L 3713 1625
    L 3713 1100
    L 3047 1100
    L 3047 0
    L 2419 0
    L 2419 1100
    L 313 1100
    L 313 1709
    L 2253 4666
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-34"/>
          </g>
         </g>
        </g>
       </g>
       <g id="matplotlib.axis_2">
        <g id="ytick_1">
         <g id="line2d_9">
          <defs>
           <path id="m48968135c9" d="M 0 0
    L -3.5 0
    " style="stroke: #000000; stroke-width: 0.8"/>
          </defs>
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="256.268521" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_9">
          <!-- 0 -->
          <g transform="translate(9.36274 260.06774) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-30"/>
          </g>
         </g>
        </g>
        <g id="ytick_2">
         <g id="line2d_10">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="225.085905" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_10">
          <!-- 10 -->
          <g transform="translate(3.00024 228.885124) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-31"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
        <g id="ytick_3">
         <g id="line2d_11">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="193.903289" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_11">
          <!-- 20 -->
          <g transform="translate(3.00024 197.702507) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-32"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
        <g id="ytick_4">
         <g id="line2d_12">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="162.720672" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_12">
          <!-- 30 -->
          <g transform="translate(3.00024 166.519891) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-33"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
        <g id="ytick_5">
         <g id="line2d_13">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="131.538056" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_13">
          <!-- 40 -->
          <g transform="translate(3.00024 135.337275) scale(0.1 -0.1)">
           <use xlink:href="#DejaVuSans-34"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
        <g id="ytick_6">
         <g id="line2d_14">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="100.35544" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_14">
          <!-- 50 -->
          <g transform="translate(3.00024 104.154659) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-35" d="M 691 4666
    L 3169 4666
    L 3169 4134
    L 1269 4134
    L 1269 2991
    Q 1406 3038 1543 3061
    Q 1681 3084 1819 3084
    Q 2600 3084 3056 2656
    Q 3513 2228 3513 1497
    Q 3513 744 3044 326
    Q 2575 -91 1722 -91
    Q 1428 -91 1123 -41
    Q 819 9 494 109
    L 494 744
    Q 775 591 1075 516
    Q 1375 441 1709 441
    Q 2250 441 2565 725
    Q 2881 1009 2881 1497
    Q 2881 1984 2565 2268
    Q 2250 2553 1709 2553
    Q 1456 2553 1204 2497
    Q 953 2441 691 2322
    L 691 4666
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-35"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
        <g id="ytick_7">
         <g id="line2d_15">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="69.172824" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_15">
          <!-- 60 -->
          <g transform="translate(3.00024 72.972043) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-36" d="M 2113 2584
    Q 1688 2584 1439 2293
    Q 1191 2003 1191 1497
    Q 1191 994 1439 701
    Q 1688 409 2113 409
    Q 2538 409 2786 701
    Q 3034 994 3034 1497
    Q 3034 2003 2786 2293
    Q 2538 2584 2113 2584
    z
    M 3366 4563
    L 3366 3988
    Q 3128 4100 2886 4159
    Q 2644 4219 2406 4219
    Q 1781 4219 1451 3797
    Q 1122 3375 1075 2522
    Q 1259 2794 1537 2939
    Q 1816 3084 2150 3084
    Q 2853 3084 3261 2657
    Q 3669 2231 3669 1497
    Q 3669 778 3244 343
    Q 2819 -91 2113 -91
    Q 1303 -91 875 529
    Q 447 1150 447 2328
    Q 447 3434 972 4092
    Q 1497 4750 2381 4750
    Q 2619 4750 2861 4703
    Q 3103 4656 3366 4563
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-36"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
        <g id="ytick_8">
         <g id="line2d_16">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="37.990208" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_16">
          <!-- 70 -->
          <g transform="translate(3.00024 41.789426) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-37" d="M 525 4666
    L 3525 4666
    L 3525 4397
    L 1831 0
    L 1172 0
    L 2766 4134
    L 525 4134
    L 525 4666
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-37"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
        <g id="ytick_9">
         <g id="line2d_17">
          <g>
           <use xlink:href="#m48968135c9" x="22.72524" y="6.807591" style="stroke: #000000; stroke-width: 0.8"/>
          </g>
         </g>
         <g id="text_17">
          <!-- 80 -->
          <g transform="translate(3.00024 10.60681) scale(0.1 -0.1)">
           <defs>
            <path id="DejaVuSans-38" d="M 2034 2216
    Q 1584 2216 1326 1975
    Q 1069 1734 1069 1313
    Q 1069 891 1326 650
    Q 1584 409 2034 409
    Q 2484 409 2743 651
    Q 3003 894 3003 1313
    Q 3003 1734 2745 1975
    Q 2488 2216 2034 2216
    z
    M 1403 2484
    Q 997 2584 770 2862
    Q 544 3141 544 3541
    Q 544 4100 942 4425
    Q 1341 4750 2034 4750
    Q 2731 4750 3128 4425
    Q 3525 4100 3525 3541
    Q 3525 3141 3298 2862
    Q 3072 2584 2669 2484
    Q 3125 2378 3379 2068
    Q 3634 1759 3634 1313
    Q 3634 634 3220 271
    Q 2806 -91 2034 -91
    Q 1263 -91 848 271
    Q 434 634 434 1313
    Q 434 1759 690 2068
    Q 947 2378 1403 2484
    z
    M 1172 3481
    Q 1172 3119 1398 2916
    Q 1625 2713 2034 2713
    Q 2441 2713 2670 2916
    Q 2900 3119 2900 3481
    Q 2900 3844 2670 4047
    Q 2441 4250 2034 4250
    Q 1625 4250 1398 4047
    Q 1172 3844 1172 3481
    z
    " transform="scale(0.015625)"/>
           </defs>
           <use xlink:href="#DejaVuSans-38"/>
           <use xlink:href="#DejaVuSans-30" x="63.623047"/>
          </g>
         </g>
        </g>
       </g>
       <g id="LineCollection_1">
        <path d="M 40.958396 256.268521
    L 40.958396 250.031998
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 47.036114 252.314724
    L 47.036114 241.512749
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 53.113833 254.441886
    L 53.113833 245.62211
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 59.191551 254.441886
    L 59.191551 245.62211
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 65.26927 254.441886
    L 65.26927 245.62211
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 71.346988 245.197101
    L 71.346988 229.920802
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 77.424707 245.197101
    L 77.424707 229.920802
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 83.502425 237.558951
    L 83.502425 218.849382
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 89.580144 242.690835
    L 89.580144 226.190545
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 95.657863 237.558951
    L 95.657863 218.849382
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 101.735581 232.309747
    L 101.735581 211.625539
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 107.8133 205.074697
    L 107.8133 176.495357
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 113.891018 224.280325
    L 113.891018 200.945392
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 119.968737 176.964157
    L 119.968737 142.240665
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 126.046455 191.093796
    L 126.046455 159.293642
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 132.124174 165.577249
    L 132.124174 128.68148
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 138.201892 142.63108
    L 138.201892 101.735463
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 144.279611 145.510178
    L 144.279611 105.092888
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 150.357329 119.506021
    L 150.357329 74.968336
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 156.435048 116.605021
    L 156.435048 71.632812
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 162.512767 67.010696
    L 162.512767 15.206243
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 168.590485 110.796843
    L 168.590485 64.967944
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 174.668204 96.242907
    L 174.668204 48.339263
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 180.745922 145.510178
    L 180.745922 105.092888
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 186.823641 87.489517
    L 186.823641 38.383084
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 192.901359 122.404879
    L 192.901359 78.306001
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 198.979078 119.506021
    L 198.979078 74.968336
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 205.056796 131.08795
    L 205.056796 88.3325
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 211.134515 159.860486
    L 211.134515 121.925196
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 217.212233 154.129855
    L 217.212233 115.18278
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 223.289952 188.27842
    L 223.289952 155.872495
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 229.367671 221.571572
    L 229.367671 197.417621
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 235.445389 213.369476
    L 235.445389 186.910148
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 241.523108 218.849382
    L 241.523108 193.903289
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 247.600826 224.280325
    L 247.600826 200.945392
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 253.678545 240.142204
    L 253.678545 222.502652
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 259.756263 245.197101
    L 259.756263 229.920802
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 265.833982 250.031998
    L 265.833982 237.558951
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 271.9117 254.441886
    L 271.9117 245.62211
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 277.989419 250.031998
    L 277.989419 237.558951
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 284.067137 256.268521
    L 284.067137 256.268521
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 290.144856 256.268521
    L 290.144856 256.268521
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 296.222575 256.268521
    L 296.222575 256.268521
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 302.300293 256.268521
    L 302.300293 250.031998
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 308.378012 256.268521
    L 308.378012 256.268521
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 314.45573 256.268521
    L 314.45573 250.031998
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 320.533449 256.268521
    L 320.533449 256.268521
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 326.611167 256.268521
    L 326.611167 256.268521
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 332.688886 256.268521
    L 332.688886 256.268521
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
        <path d="M 338.766604 256.268521
    L 338.766604 250.031998
    " clip-path="url(#pb98193418b)" style="fill: none; stroke: #000000; stroke-width: 1.5"/>
       </g>
       <g id="line2d_18">
        <defs>
         <path id="mfacc111964" d="M 0 3
    C 0.795609 3 1.55874 2.683901 2.12132 2.12132
    C 2.683901 1.55874 3 0.795609 3 0
    C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132
    C 1.55874 -2.683901 0.795609 -3 0 -3
    C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132
    C -2.683901 -1.55874 -3 -0.795609 -3 0
    C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132
    C -1.55874 2.683901 -0.795609 3 0 3
    z
    " style="stroke: #000000"/>
        </defs>
        <g clip-path="url(#pb98193418b)">
         <use xlink:href="#mfacc111964" x="40.958396" y="253.150259" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="47.036114" y="246.913736" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="53.113833" y="250.031998" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="59.191551" y="250.031998" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="65.26927" y="250.031998" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="71.346988" y="237.558951" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="77.424707" y="237.558951" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="83.502425" y="228.204167" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="89.580144" y="234.44069" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="95.657863" y="228.204167" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="101.735581" y="221.967643" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="107.8133" y="190.785027" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="113.891018" y="212.612858" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="119.968737" y="159.602411" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="126.046455" y="175.193719" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="132.124174" y="147.129364" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="138.201892" y="122.183271" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="144.279611" y="125.301533" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="150.357329" y="97.237178" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="156.435048" y="94.118917" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="162.512767" y="41.108469" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="168.590485" y="87.882394" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="174.668204" y="72.291085" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="180.745922" y="125.301533" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="186.823641" y="62.936301" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="192.901359" y="100.35544" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="198.979078" y="97.237178" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="205.056796" y="109.710225" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="211.134515" y="140.892841" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="217.212233" y="134.656318" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="223.289952" y="172.075457" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="229.367671" y="209.494597" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="235.445389" y="200.139812" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="241.523108" y="206.376335" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="247.600826" y="212.612858" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="253.678545" y="231.322428" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="259.756263" y="237.558951" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="265.833982" y="243.795475" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="271.9117" y="250.031998" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="277.989419" y="243.795475" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="284.067137" y="256.268521" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="290.144856" y="256.268521" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="296.222575" y="256.268521" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="302.300293" y="253.150259" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="308.378012" y="256.268521" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="314.45573" y="253.150259" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="320.533449" y="256.268521" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="326.611167" y="256.268521" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="332.688886" y="256.268521" style="stroke: #000000"/>
         <use xlink:href="#mfacc111964" x="338.766604" y="253.150259" style="stroke: #000000"/>
        </g>
       </g>
       <g id="patch_3">
        <path d="M 22.72524 268.321635
    L 22.72524 3.153129
    " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
       </g>
       <g id="patch_4">
        <path d="M 356.99976 268.321635
    L 356.99976 3.153129
    " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
       </g>
       <g id="patch_5">
        <path d="M 22.72524 268.321635
    L 356.99976 268.321635
    " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
       </g>
       <g id="patch_6">
        <path d="M 22.72524 3.153129
    L 356.99976 3.153129
    " style="fill: none; stroke: #000000; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
       </g>
      </g>
     </g>
     <defs>
      <clipPath id="pb98193418b">
       <rect x="22.72524" y="3.153129" width="334.27452" height="265.168506"/>
      </clipPath>
     </defs>
    </svg>

Interactive fitting
-------------------

iminuit optionally supports an interactive fitting mode in Jupyter notebooks.

.. image:: doc/_static/interactive_demo.gif
   :alt: Animated demo of an interactive fit in a Jupyter notebook

Partner projects
----------------

* `boost-histogram`_ from Scikit-HEP provides fast generalized histograms that you can use with the builtin cost functions.
* `numba_stats`_ provides faster implementations of probability density functions than scipy, and a few specific ones used in particle physics that are not in scipy.
* `jacobi`_ provides a robust, fast, and accurate calculation of the Jacobi matrix of any transformation function and building a function for generic error propagation.

Versions
--------

**The current 2.x series has introduced breaking interfaces changes with respect to the 1.x series.**

All interface changes are documented in the `changelog`_ with recommendations how to upgrade. To keep existing scripts running, pin your major iminuit version to <2, i.e. ``pip install 'iminuit<2'`` installs the 1.x series.

.. _changelog: https://iminuit.readthedocs.io/en/stable/changelog.html
.. _tutorials: https://iminuit.readthedocs.io/en/stable/tutorials.html
.. _discussions: https://github.com/scikit-hep/iminuit/discussions
.. _gitter: https://gitter.im/Scikit-HEP/iminuit
.. _jacobi: https://github.com/hdembinski/jacobi
.. _numba_stats: https://github.com/HDembinski/numba-stats
.. _boost-histogram: https://github.com/scikit-hep/boost-histogram
