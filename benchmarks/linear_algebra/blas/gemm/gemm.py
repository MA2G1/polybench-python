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


class Gemm(PolyBench):

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
        self.NI = params.get('NI')
        self.NJ = params.get('NJ')
        self.NK = params.get('NK')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        alpha = 1.5
        beta = 1.2

        C = self.create_array(2, [self.NI, self.NJ], self.DATA_TYPE(0))
        A = self.create_array(2, [self.NI, self.NK], self.DATA_TYPE(0))
        B = self.create_array(2, [self.NK, self.NJ], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(C, A, B)

        # Benchmark the kernel
        self.time_kernel(alpha, beta, C, A, B)

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
        return [('C', C)]


class _StrategyList(Gemm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, C: list, A: list, B: list):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[i][j] = self.DATA_TYPE((i * j + 1) % self.NI) / self.NI

        for i in range(0, self.NI):
            for j in range (0, self.NK):
                A[i][j] = self.DATA_TYPE(i * (j + 1) % self.NK) / self.NK

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[i][j] = self.DATA_TYPE(i * (j + 2) % self.NJ) / self.NJ

    def print_array_custom(self, C: list, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(C[i][j])

    def kernel(self, alpha: float, beta: float, C: list, A: list, B: list):
# scop begin
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[i][j] *= beta

            for k in range(0, self.NK):
                for j in range(0, self.NJ):
                    C[i][j] += alpha * A[i][k] * B[k][j]
# scop end


class _StrategyListFlattened(Gemm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, C: list, A: list, B: list):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[self.NJ * i + j] = self.DATA_TYPE((i * j + 1) % self.NI) / self.NI

        for i in range(0, self.NI):
            for j in range(0, self.NK):
                A[self.NK * i + j] = self.DATA_TYPE(i * (j + 1) % self.NK) / self.NK

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[self.NJ * i + j] = self.DATA_TYPE(i * (j + 2) % self.NJ) / self.NJ

    def print_array_custom(self, C: list, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(C[self.NJ * i + j])

    def kernel(self, alpha: float, beta: float, C: list, A: list, B: list):
# scop begin
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[self.NJ * i + j] *= beta

            for k in range(0, self.NK):
                for j in range(0, self.NJ):
                    C[self.NJ * i + j] += alpha * A[self.NK * i + k] * B[self.NJ * k + j]
# scop end


class _StrategyNumPy(Gemm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, C: ndarray, A: ndarray, B: ndarray):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[i, j] = self.DATA_TYPE((i * j + 1) % self.NI) / self.NI

        for i in range(0, self.NI):
            for j in range (0, self.NK):
                A[i, j] = self.DATA_TYPE(i * (j + 1) % self.NK) / self.NK

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[i, j] = self.DATA_TYPE(i * (j + 2) % self.NJ) / self.NJ

    def print_array_custom(self, C: ndarray, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(C[i, j])

    def kernel(self, alpha: float, beta: float, C: ndarray, A: ndarray, B: ndarray):
# scop begin
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[i, j] *= beta

            for k in range(0, self.NK):
                for j in range(0, self.NJ):
                    C[i, j] += alpha * A[i, k] * B[k, j]
# scop end
