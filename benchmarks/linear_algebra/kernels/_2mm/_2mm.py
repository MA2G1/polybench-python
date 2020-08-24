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


class _2mm(PolyBench):

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
        self.NL = params.get('NL')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        alpha = 1.5
        beta = 1.2

        tmp = self.create_array(2, [self.NI, self.NJ], self.DATA_TYPE(0))
        A = self.create_array(2, [self.NI, self.NK], self.DATA_TYPE(0))
        B = self.create_array(2, [self.NK, self.NJ], self.DATA_TYPE(0))
        C = self.create_array(2, [self.NJ, self.NL], self.DATA_TYPE(0))
        D = self.create_array(2, [self.NI, self.NL], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, B, C, D)

        # Benchmark the kernel
        self.time_kernel(alpha, beta, tmp, A, B, C, D)

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
        return [('D', D)]


class _StrategyList(_2mm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list, C: list, D: list):
        for i in range(0, self.NI):
            for j in range(0, self.NK):
                A[i][j] = self.DATA_TYPE((i * j + 1) % self.NI) / self.NI

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[i][j] = self.DATA_TYPE(i * (j + 1) % self.NJ) / self.NJ

        for i in range(0, self.NJ):
            for j in range(0, self.NL):
                C[i][j] = self.DATA_TYPE((i * (j + 3) + 1) % self.NL) / self.NL

        for i in range(0, self.NI):
            for j in range(0, self.NL):
                D[i][j] = self.DATA_TYPE(i * (j + 2) % self.NK) / self.NK

    def print_array_custom(self, D: list, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(D[i][j])

    def kernel(self, alpha, beta, tmp: list, A: list, B: list, C: list, D: list):
# scop begin
        # D := alpha * A * B * C + beta * D
        for i in range(self.NI):
            for j in range(self.NJ):
                tmp[i][j] = 0.0
                for k in range(0, self.NK):
                    tmp[i][j] += alpha * A[i][k] * B[k][j]

        for i in range(0, self.NI):
            for j in range(0, self.NL):
                D[i][j] *= beta
                for k in range(0, self.NJ):
                    D[i][j] += tmp[i][k] * C[k][j]
# scop end


class _StrategyListFlattened(_2mm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list, C: list, D: list):
        for i in range(0, self.NI):
            for j in range(0, self.NK):
                A[self.NK * i + j] = self.DATA_TYPE((i * j + 1) % self.NI) / self.NI

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[self.NJ * i + j] = self.DATA_TYPE(i * (j + 1) % self.NJ) / self.NJ

        for i in range(0, self.NJ):
            for j in range(0, self.NL):
                C[self.NL * i + j] = self.DATA_TYPE((i * (j + 3) + 1) % self.NL) / self.NL

        for i in range(0, self.NI):
            for j in range(0, self.NL):
                D[self.NL * i + j] = self.DATA_TYPE(i * (j + 2) % self.NK) / self.NK

    def print_array_custom(self, D: list, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(D[self.NL * i + j])

    def kernel(self, alpha, beta, tmp: list, A: list, B: list, C: list, D: list):
# scop begin
        # D := alpha * A * B * C + beta * D
        for i in range(self.NI):
            for j in range(self.NJ):
                tmp[self.NJ * i + j] = 0.0
                for k in range(0, self.NK):
                    tmp[self.NJ * i + j] += alpha * A[self.NK * i + k] * B[self.NJ * k + j]

        for i in range(0, self.NI):
            for j in range(0, self.NL):
                D[self.NL * i + j] *= beta
                for k in range(0, self.NJ):
                    D[self.NL * i + j] += tmp[self.NJ * i + k] * C[self.NL * k + j]
# scop end


class _StrategyNumPy(_2mm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, B: ndarray, C: ndarray, D: ndarray):
        for i in range(0, self.NI):
            for j in range(0, self.NK):
                A[i, j] = self.DATA_TYPE((i * j + 1) % self.NI) / self.NI

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[i, j] = self.DATA_TYPE(i * (j + 1) % self.NJ) / self.NJ

        for i in range(0, self.NJ):
            for j in range(0, self.NL):
                C[i, j] = self.DATA_TYPE((i * (j + 3) + 1) % self.NL) / self.NL

        for i in range(0, self.NI):
            for j in range(0, self.NL):
                D[i, j] = self.DATA_TYPE(i * (j + 2) % self.NK) / self.NK

    def print_array_custom(self, D: ndarray, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(D[i, j])

    def kernel(self, alpha, beta, tmp: ndarray, A: ndarray, B: ndarray, C: ndarray, D: ndarray):
# scop begin
        # D := alpha * A * B * C + beta * D
        for i in range(self.NI):
            for j in range(self.NJ):
                tmp[i, j] = 0.0
                for k in range(0, self.NK):
                    tmp[i, j] += alpha * A[i, k] * B[k, j]

        for i in range(0, self.NI):
            for j in range(0, self.NL):
                D[i, j] *= beta
                for k in range(0, self.NJ):
                    D[i, j] += tmp[i, k] * C[k, j]
# scop end
