#
# Copyright 2019 Miguel Angel Abella Gonzalez <miguel.abella@udc.es>
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


class Gesummv(PolyBench):

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
        self.N = params.get('N')

    def print_array_custom(self, y: list, name: str):
        for i in range(0, self.N):
            if i % 20 == 0:
                self.print_message('\n')
            self.print_value(y[i])

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        alpha = 1.5
        beta = 1.2

        A = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        B = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        tmp = self.create_array(1, [self.N], self.DATA_TYPE(0))
        x = self.create_array(1, [self.N], self.DATA_TYPE(0))
        y = self.create_array(1, [self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, B, x)

        # Benchmark the kernel
        self.time_kernel(alpha, beta, A, B, tmp, x, y)

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
        return [('y', y)]


class _StrategyList(Gesummv):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list, x: list):
        for i in range(0, self.N):
            x[i] = self.DATA_TYPE(i % self.N) / self.N
            for j in range(0, self.N):
                A[i][j] = self.DATA_TYPE((i * j + 1) % self.N) / self.N
                B[i][j] = self.DATA_TYPE((i * j + 2) % self.N) / self.N

    def kernel(self, alpha, beta, A: list, B: list, tmp: list, x: list, y: list):
# scop begin
        for i in range(0, self.N):
            tmp[i] = 0.0
            y[i] = 0.0
            for j in range(0, self.N):
                tmp[i] = A[i][j] * x[j] + tmp[i]
                y[i] = B[i][j] * x[j] + y[i]
            y[i] = alpha * tmp[i] + beta * y[i]
# scop end


class _StrategyListFlattened(Gesummv):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list, x: list):
        for i in range(0, self.N):
            x[i] = self.DATA_TYPE(i % self.N) / self.N
            for j in range(0, self.N):
                A[self.N * i + j] = self.DATA_TYPE((i * j+1) % self.N) / self.N
                B[self.N * i + j] = self.DATA_TYPE((i * j+2) % self.N) / self.N

    def kernel(self, alpha, beta, A: list, B: list, tmp: list, x: list, y: list):
# scop begin
        for i in range(0, self.N):
            tmp[i] = 0.0
            y[i] = 0.0
            for j in range(0, self.N):
                tmp[i] = A[self.N * i + j] * x[j] + tmp[i]
                y[i] = B[self.N * i + j] * x[j] + y[i]
            y[i] = alpha * tmp[i] + beta * y[i]
# scop end


class _StrategyNumPy(Gesummv):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, B: ndarray, x: ndarray):
        for i in range(0, self.N):
            x[i] = self.DATA_TYPE(i % self.N) / self.N
            for j in range(0, self.N):
                A[i, j] = self.DATA_TYPE((i * j + 1) % self.N) / self.N
                B[i, j] = self.DATA_TYPE((i * j + 2) % self.N) / self.N

    def kernel(self, alpha, beta, A: ndarray, B: ndarray, tmp: ndarray, x: ndarray, y: ndarray):
# scop begin
        for i in range(0, self.N):
            tmp[i] = 0.0
            y[i] = 0.0
            for j in range(0, self.N):
                tmp[i] = A[i, j] * x[j] + tmp[i]
                y[i] = B[i, j] * x[j] + y[i]
            y[i] = alpha * tmp[i] + beta * y[i]
# scop end
