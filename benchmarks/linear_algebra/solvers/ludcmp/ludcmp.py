# Copyright 2019 Miguel Angel Abella Gonzalez <miguel.abella@udc.es>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""<replace_with_module_description>"""

from benchmarks.polybench import PolyBench
from benchmarks.polybench_classes import ArrayImplementation
from benchmarks.polybench_classes import PolyBenchOptions, PolyBenchSpec
from numpy.core.multiarray import ndarray


class Ludcmp(PolyBench):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        implementation = options['array_implementation']
        if implementation == ArrayImplementation.LIST:
            return _StrategyList.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.LIST_FLATTENED:
            return _StrategyListFlattened.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.NUMPY:
            return _StrategyNumPy.__new__(cls, options, parameters)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

        # The parameters hold the necessary information obtained from "polybench.spec" file
        params = parameters.DataSets.get(self.DATASET_SIZE)
        if not isinstance(params, dict):
            raise NotImplementedError(f'Dataset size "{self.DATASET_SIZE.name}" not implemented '
                                      f'for {parameters.Category}/{parameters.Name}.')

        # Set up problem size from the given parameters (adapt this part with appropriate parameters)
        self.N = params.get('N')

    def print_array_custom(self, x: list, name: str):
        for i in range(0, self.N):
            if i % 20 == 0:
                self.print_message('\n')
            self.print_value(x[i])

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        A = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        b = self.create_array(1, [self.N], self.DATA_TYPE(0))
        x = self.create_array(1, [self.N], self.DATA_TYPE(0))
        y = self.create_array(1, [self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, b, x, y)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(A, b, x, y)

        # Stop and print instruments
        self.stop_instruments()

        # Return printable data as a list of tuples ('name', value).
        # Each tuple element must have the following format:
        #   (A: str, B: matrix)
        #     - A: a representative name for the data (this string will be printed out)
        #     - B: the actual data structure holding the computed result
        #
        # The syntax for the return statement would then be:
        #   - For single data structure results:
        #     return [('data_name', data)]
        #   - For multiple data structure results:
        #     return [('matrix1', m1), ('matrix2', m2), ... ]
        return [('x', x)]


class _StrategyList(Ludcmp):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, b: list, x: list, y: list):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            x[i] = self.DATA_TYPE(0)
            y[i] = self.DATA_TYPE(0)
            b[i] = (i + 1) / fn / 2.0 + 4

        for i in range(0, self.N):
            for j in range(0, i + 1):
                A[i][j] = -self.DATA_TYPE(j % self.N) / self.N + 1

            for j in range(i + 1, self.N):
                A[i][j] = self.DATA_TYPE(0)
            A[i][i] = self.DATA_TYPE(1)

        # Make the matrix positive semi-definite.
        # not necessary for LU, but using same code as cholesky
        B = self.create_array(2, [self.N], self.DATA_TYPE(0))

        for t in range(0, self.N):
            for r in range(0, self.N):
                for s in range(0, self.N):
                    B[r][s] += A[r][t] * A[s][t]

        for r in range(0, self.N):
            for s in range(0, self.N):
                A[r][s] = B[r][s]

    def kernel(self, A: list, b: list, x: list, y: list):
# scop begin
        for i in range(0, self.N):
            for j in range(0, i):
                w = A[i][j]
                for k in range(0, j):
                    w -= A[i][k] * A[k][j]
                A[i][j] = w / A[j][j]

            for j in range(i, self.N):
                w = A[i][j]
                for k in range(0, i):
                    w -= A[i][k] * A[k][j]
                A[i][j] = w

        for i in range(0, self.N):
            w = b[i]
            for j in range(0, i):
                w -= A[i][j] * y[j]
            y[i] = w

        for i in range(self.N - 1, -1, -1):
            w = y[i]
            for j in range(i + 1, self.N):
                w -= A[i][j] * x[j]
            x[i] = w / A[i][i]
# scop end


class _StrategyListFlattened(Ludcmp):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, b: list, x: list, y: list):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            x[i] = self.DATA_TYPE(0)
            y[i] = self.DATA_TYPE(0)
            b[i] = (i + 1) / fn / 2.0 + 4

        for i in range(0, self.N):
            for j in range(0, i + 1):
                A[self.N * i + j] = -self.DATA_TYPE(j % self.N) / self.N + 1

            for j in range(i + 1, self.N):
                A[self.N * i + j] = self.DATA_TYPE(0)
            A[self.N * i + i] = self.DATA_TYPE(1)

        # Make the matrix positive semi-definite.
        # not necessary for LU, but using same code as cholesky
        B = self.create_array(1, [self.N * self.N], self.DATA_TYPE(0))

        for t in range(0, self.N):
            for r in range(0, self.N):
                for s in range(0, self.N):
                    B[self.N * r + s] += A[self.N * r + t] * A[self.N * s + t]

        for r in range(0, self.N):
            for s in range(0, self.N):
                A[self.N * r + s] = B[self.N * r + s]

    def kernel(self, A: list, b: list, x: list, y: list):
# scop begin
        for i in range(0, self.N):
            for j in range(0, i):
                w = A[self.N * i + j]
                for k in range(0, j):
                    w -= A[self.N * i + k] * A[self.N * k + j]
                A[self.N * i + j] = w / A[self.N * j + j]

            for j in range(i, self.N):
                w = A[self.N * i + j]
                for k in range(0, i):
                    w -= A[self.N * i + k] * A[self.N * k + j]
                A[self.N * i + j] = w

        for i in range(0, self.N):
            w = b[i]
            for j in range(0, i):
                w -= A[self.N * i + j] * y[j]
            y[i] = w

        for i in range(self.N - 1, -1, -1):
            w = y[i]
            for j in range(i + 1, self.N):
                w -= A[self.N * i + j] * x[j]
            x[i] = w / A[self.N * i + i]
# scop end


class _StrategyNumPy(Ludcmp):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, b: ndarray, x: ndarray, y: ndarray):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            x[i] = self.DATA_TYPE(0)
            y[i] = self.DATA_TYPE(0)
            b[i] = (i + 1) / fn / 2.0 + 4

        for i in range(0, self.N):
            for j in range(0, i + 1):
                A[i, j] = -self.DATA_TYPE(j % self.N) / self.N + 1

            for j in range(i + 1, self.N):
                A[i, j] = self.DATA_TYPE(0)
            A[i, i] = self.DATA_TYPE(1)

        # Make the matrix positive semi-definite.
        # not necessary for LU, but using same code as cholesky
        B = self.create_array(2, [self.N], self.DATA_TYPE(0))

        for t in range(0, self.N):
            for r in range(0, self.N):
                for s in range(0, self.N):
                    B[r, s] += A[r, t] * A[s, t]

        for r in range(0, self.N):
            for s in range(0, self.N):
                A[r, s] = B[r, s]

    def kernel(self, A: ndarray, b: ndarray, x: ndarray, y: ndarray):
# scop begin
        for i in range(0, self.N):
            for j in range(0, i):
                w = A[i, j]
                for k in range(0, j):
                    w -= A[i, k] * A[k, j]
                A[i, j] = w / A[j, j]

            for j in range(i, self.N):
                w = A[i, j]
                for k in range(0, i):
                    w -= A[i, k] * A[k, j]
                A[i, j] = w

        for i in range(0, self.N):
            w = b[i]
            for j in range(0, i):
                w -= A[i, j] * y[j]
            y[i] = w

        for i in range(self.N - 1, -1, -1):
            w = y[i]
            for j in range(i + 1, self.N):
                w -= A[i, j] * x[j]
            x[i] = w / A[i, i]
# scop end
