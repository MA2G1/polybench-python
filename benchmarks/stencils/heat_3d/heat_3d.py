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


class Heat_3d(PolyBench):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        implementation = options.POLYBENCH_ARRAY_IMPLEMENTATION
        if implementation == ArrayImplementation.LIST:
            return _StrategyList.__new__(_StrategyList, options, parameters)
        elif implementation == ArrayImplementation.LIST_FLATTENED:
            return _StrategyListFlattened.__new__(_StrategyListFlattened, options, parameters)
        elif implementation == ArrayImplementation.NUMPY:
            return _StrategyNumPy.__new__(_StrategyNumPy, options, parameters)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

        # The parameters hold the necessary information obtained from "polybench.spec" file
        params = parameters.DataSets.get(self.DATASET_SIZE)
        if not isinstance(params, dict):
            raise NotImplementedError(f'Dataset size "{self.DATASET_SIZE.name}" not implemented '
                                      f'for {parameters.Category}/{parameters.Name}.')

        # Set up problem size from the given parameters (adapt this part with appropriate parameters)
        self.TSTEPS = params.get('TSTEPS')
        self.N = params.get('N')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        A = self.create_array(3, [self.N, self.N, self.N], self.DATA_TYPE(0))
        B = self.create_array(3, [self.N, self.N, self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, B)

        # Benchmark the kernel
        self.time_kernel(A, B)

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
        return [('A', A)]


class _StrategyList(Heat_3d):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list):
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    A[i][j][k] = B[i][j][k] = self.DATA_TYPE(i + j + (self.N - k)) * 10 / self.N

    def print_array_custom(self, A: list, name: str):
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    if (i * self.N * self.N + j * self.N + k) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(A[i][j][k])

    def kernel(self, A: list, B: list):
# scop begin
        for t in range(1, self.TSTEPS + 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    for k in range(1, self.N - 1):
                        B[i][j][k] = (0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                    + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                    + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                    + A[i][j][k])

            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    for k in range(1, self.N - 1):
                        A[i][j][k] = (0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                    + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                    + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                    + B[i][j][k])
# scop end


class _StrategyListFlattened(Heat_3d):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list):
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    A[(self.N * i + j) * self.N + k] = B[(self.N * i + j) * self.N + k]\
                        = self.DATA_TYPE(i + j + (self.N-k)) * 10 / self.N

    def print_array_custom(self, A: list, name: str):
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    if (i * self.N * self.N + j * self.N + k) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(A[(self.N * i + j) * self.N + k])

    def kernel(self, A: list, B: list):
# scop begin
        for t in range(1, self.TSTEPS + 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    for k in range(1, self.N - 1):
                        B[(self.N * i + j) * self.N + k] = (
                                0.125 * (A[(self.N * (i + 1) + j) * self.N + k]
                                         - 2.0 * A[(self.N * i + j) * self.N + k]
                                         + A[(self.N * (i - 1) + j) * self.N + k])
                                + 0.125 * (A[(self.N * i + (j + 1)) * self.N + k]
                                           - 2.0 * A[(self.N * i + j) * self.N + k]
                                           + A[(self.N * i + (j - 1)) * self.N + k])
                                + 0.125 * (A[(self.N * i + j) * self.N + (k + 1)]
                                           - 2.0 * A[(self.N * i + j) * self.N + k]
                                           + A[(self.N * i + j) * self.N + (k - 1)])
                                + A[(self.N * i + j) * self.N + k]
                        )

            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    for k in range(1, self.N - 1):
                        A[(self.N * i + j) * self.N + k] = (
                                0.125 * (B[(self.N * (i + 1) + j) * self.N + k]
                                         - 2.0 * B[(self.N * i + j) * self.N + k]
                                         + B[(self.N * (i - 1) + j) * self.N + k])
                                + 0.125 * (B[(self.N * i + (j + 1)) * self.N + k]
                                           - 2.0 * B[(self.N * i + j) * self.N + k]
                                           + B[(self.N * i + (j - 1)) * self.N + k])
                                + 0.125 * (B[(self.N * i + j) * self.N + (k + 1)]
                                           - 2.0 * B[(self.N * i + j) * self.N + k]
                                           + B[(self.N * i + j) * self.N + (k - 1)])
                                + B[(self.N * i + j) * self.N + k]
                        )
# scop end


class _StrategyNumPy(Heat_3d):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, B: ndarray):
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    A[i, j, k] = B[i, j, k] = self.DATA_TYPE(i + j + (self.N-k)) * 10 / self.N

    def print_array_custom(self, A: ndarray, name: str):
        for i in range(0, self.N):
            for j in range(0, self.N):
                for k in range(0, self.N):
                    if (i * self.N * self.N + j * self.N + k) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(A[i, j, k])

    def kernel(self, A: ndarray, B: ndarray):
# scop begin
        for t in range(1, self.TSTEPS + 1):
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    for k in range(1, self.N - 1):
                        B[i, j, k] = (0.125 * (A[i+1, j, k] - 2.0 * A[i, j, k] + A[i-1, j, k])
                                    + 0.125 * (A[i, j+1, k] - 2.0 * A[i, j, k] + A[i, j-1, k])
                                    + 0.125 * (A[i, j, k+1] - 2.0 * A[i, j, k] + A[i, j, k-1])
                                    + A[i, j, k])

            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    for k in range(1, self.N - 1):
                        A[i, j, k] = (0.125 * (B[i+1, j, k] - 2.0 * B[i, j, k] + B[i-1, j, k])
                                    + 0.125 * (B[i, j+1, k] - 2.0 * B[i, j, k] + B[i, j-1, k])
                                    + 0.125 * (B[i, j, k+1] - 2.0 * B[i, j, k] + B[i, j, k-1])
                                    + B[i, j, k])
# scop end
