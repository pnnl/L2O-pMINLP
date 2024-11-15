# MIPcc23: The MIP Workshop 2023 Computational Competition

This directory contains series of 50 related mixed-integer programs built from a common formulation with varying input data.
Depending on the series, some or all of the following input can vary:
1. objective function coefficients,
2. variable bounds,
3. constraint right-hand sides,
4. constraint coefficients.

The official computational evaluation of the competition will be applied to the data sets with constant constraint coefficients, where only types 1, 2, and 3 will occur:
- datasets/testfiles/bnd_series_1.test
- datasets/testfiles/bnd_series_2.test
- datasets/testfiles/obj_series_1.test
- datasets/testfiles/obj_series_2.test
- datasets/testfiles/rhs_obj_series_1.test
- datasets/testfiles/rhs_series_1.test
- datasets/testfiles/rhs_series_2.test

In addition, this directory contains the two series with varying constraint matrix:
- datasets/testfiles/mat_rhs_bnd_obj_series_1.test
- datasets/testfiles/matrix_series_1.test
