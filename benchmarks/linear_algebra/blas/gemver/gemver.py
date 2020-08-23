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


class Gemver(PolyBench):

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

    def print_array_custom(self, w: list, name: str):
        for i in range(0, self.N):
            if i % 20 == 0:
                self.print_message('\n')
            self.print_value(w[i])

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        alpha = self.DATA_TYPE(1.5)
        beta = self.DATA_TYPE(1.2)

        A = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        u1 = self.create_array(1, [self.N], self.DATA_TYPE(0))
        v1 = self.create_array(1, [self.N], self.DATA_TYPE(0))
        u2 = self.create_array(1, [self.N], self.DATA_TYPE(0))
        v2 = self.create_array(1, [self.N], self.DATA_TYPE(0))
        w = self.create_array(1, [self.N], self.DATA_TYPE(0))
        x = self.create_array(1, [self.N], self.DATA_TYPE(0))
        y = self.create_array(1, [self.N], self.DATA_TYPE(0))
        z = self.create_array(1, [self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, u1, v1, u2, v2, w, x, y, z)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z)

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
        return [('w', w)]


class _StrategyList(Gemver):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, u1: list, v1: list, u2: list, v2: list, w: list, x: list, y: list, z: list):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            u1[i] = i
            u2[i] = ((i + 1) / fn) / 2.0
            v1[i] = ((i + 1) / fn) / 4.0
            v2[i] = ((i + 1) / fn) / 6.0
            y[i] = ((i + 1) / fn) / 8.0
            z[i] = ((i + 1) / fn) / 9.0
            x[i] = 0.0
            w[i] = 0.0
            for j in range(0, self.N):
                A[i][j] = self.DATA_TYPE(i * j % self.N) / self.N

    def kernel(self, alpha, beta, A: list, u1: list, v1: list, u2: list, v2: list, w: list, x: list, y: list, z: list):
# scop begin
        for i in range(0, self.N):
            for j in range(0, self.N):
                A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]

        for i in range(0, self.N):
            for j in range(0, self.N):
                x[i] = x[i] + beta * A[j][i] * y[j]

        for i in range(0, self.N):
            x[i] = x[i] + z[i]

        for i in range(0, self.N):
            for j in range(0, self.N):
                w[i] = w[i] + alpha * A[i][j] * x[j]
# scop end


class _StrategyListFlattened(Gemver):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, u1: list, v1: list, u2: list, v2: list, w: list, x: list, y: list, z: list):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            u1[i] = i
            u2[i] = ((i + 1) / fn) / 2.0
            v1[i] = ((i + 1) / fn) / 4.0
            v2[i] = ((i + 1) / fn) / 6.0
            y[i] = ((i + 1) / fn) / 8.0
            z[i] = ((i + 1) / fn) / 9.0
            x[i] = 0.0
            w[i] = 0.0
            for j in range(0, self.N):
                A[self.N * i + j] = self.DATA_TYPE(i * j % self.N) / self.N

    def kernel(self, alpha, beta, A: list, u1: list, v1: list, u2: list, v2: list, w: list, x: list, y: list, z: list):
# scop begin
        for i in range(0, self.N):
            for j in range(0, self.N):
                A[self.N * i + j] = A[self.N * i + j] + u1[i] * v1[j] + u2[i] * v2[j]

        for i in range(0, self.N):
            for j in range(0, self.N):
                x[i] = x[i] + beta * A[self.N * j + i] * y[j]

        for i in range(0, self.N):
            x[i] = x[i] + z[i]

        for i in range(0, self.N):
            for j in range(0, self.N):
                w[i] = w[i] + alpha * A[self.N * i + j] * x[j]
# scop end


class _StrategyNumPy(Gemver):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, u1: ndarray, v1: ndarray, u2: ndarray, v2: ndarray,
                         w: ndarray, x: ndarray, y: ndarray, z: ndarray):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            u1[i] = i
            u2[i] = ((i + 1) / fn) / 2.0
            v1[i] = ((i + 1) / fn) / 4.0
            v2[i] = ((i + 1) / fn) / 6.0
            y[i] = ((i + 1) / fn) / 8.0
            z[i] = ((i + 1) / fn) / 9.0
            x[i] = 0.0
            w[i] = 0.0
            for j in range(0, self.N):
                A[i, j] = self.DATA_TYPE(i * j % self.N) / self.N

    def kernel(self, alpha, beta, A: ndarray, u1: ndarray, v1: ndarray, u2: ndarray, v2: ndarray,
               w: ndarray, x: ndarray, y: ndarray, z: ndarray):
# scop begin
        for i in range(0, self.N):
            for j in range(0, self.N):
                A[i, j] = A[i, j] + u1[i] * v1[j] + u2[i] * v2[j]

        for i in range(0, self.N):
            for j in range(0, self.N):
                x[i] = x[i] + beta * A[j, i] * y[j]

        for i in range(0, self.N):
            x[i] = x[i] + z[i]

        for i in range(0, self.N):
            for j in range(0, self.N):
                w[i] = w[i] + alpha * A[i, j] * x[j]
# scop end
