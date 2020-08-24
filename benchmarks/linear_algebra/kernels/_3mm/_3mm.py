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


class _3mm(PolyBench):

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
        self.NM = params.get('NM')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        E = self.create_array(2, [self.NI, self.NJ], self.DATA_TYPE(0))
        A = self.create_array(2, [self.NI, self.NK], self.DATA_TYPE(0))
        B = self.create_array(2, [self.NK, self.NJ], self.DATA_TYPE(0))
        F = self.create_array(2, [self.NJ, self.NL], self.DATA_TYPE(0))
        C = self.create_array(2, [self.NJ, self.NM], self.DATA_TYPE(0))
        D = self.create_array(2, [self.NM, self.NL], self.DATA_TYPE(0))
        G = self.create_array(2, [self.NI, self.NL], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, B, C, D)

        # Benchmark the kernel
        self.time_kernel(E, A, B, F, C, D, G)

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
        return [('G', G)]


class _StrategyList(_3mm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list, C: list, D: list):
        for i in range(0, self.NI):
            for j in range(0, self.NK):
                A[i][j] = self.DATA_TYPE((i * j + 1) % self.NI) / (5 * self.NI)

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[i][j] = self.DATA_TYPE((i * (j + 1) + 2) % self.NJ) / (5 * self.NJ)

        for i in range(0, self.NJ):
            for j in range(0, self.NM):
                C[i][j] = self.DATA_TYPE(i * (j + 3) % self.NL) / (5 * self.NL)

        for i in range(0, self.NM):
            for j in range(0, self.NL):
                D[i][j] = self.DATA_TYPE((i * (j + 2) + 2) % self.NK) / (5 * self.NK)

    def print_array_custom(self, G: list, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(G[i][j])

    def kernel(self, E: list, A: list, B: list, F: list, C: list, D: list, G: list):
# scop begin
        # E := A * B
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                E[i][j] = 0.0
                for k in range(0, self.NK):
                    E[i][j] += A[i][k] * B[k][j]

        # F := C * D
        for i in range(0, self.NJ):
            for j in range(0, self.NL):
                F[i][j] = 0.0
                for k in range(0, self.NM):
                    F[i][j] += C[i][k] * D[k][j]

        # G := E * F
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                G[i][j] = 0.0
                for k in range(0, self.NJ):
                    G[i][j] += E[i][k] * F[k][j]
# scop end


class _StrategyListFlattened(_3mm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list, C: list, D: list):
        for i in range(0, self.NI):
            for j in range(0, self.NK):
                A[self.NK * i + j] = self.DATA_TYPE((i * j + 1) % self.NI) / (5 * self.NI)

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[self.NJ * i + j] = self.DATA_TYPE((i * (j + 1) + 2) % self.NJ) / (5 * self.NJ)

        for i in range(0, self.NJ):
            for j in range(0, self.NM):
                C[self.NM * i + j] = self.DATA_TYPE(i * (j + 3) % self.NL) / (5 * self.NL)

        for i in range(0, self.NM):
            for j in range(0, self.NL):
                D[self.NL * i + j] = self.DATA_TYPE((i * (j + 2) + 2) % self.NK) / (5 * self.NK)

    def print_array_custom(self, G: list, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(G[self.NL * i + j])

    def kernel(self, E: list, A: list, B: list, F: list, C: list, D: list, G: list):
# scop begin
        # E := A * B
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                E[self.NJ * i + j] = 0.0
                for k in range(0, self.NK):
                    E[self.NJ * i + j] += A[self.NK * i + k] * B[self.NJ * k + j]

        # F := C * D
        for i in range(0, self.NJ):
            for j in range(0, self.NL):
                F[self.NL * i + j] = 0.0
                for k in range(0, self.NM):
                    F[self.NL * i + j] += C[self.NM * i + k] * D[self.NL * k + j]

        # G := E * F
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                G[self.NL * i + j] = 0.0
                for k in range(0, self.NJ):
                    G[self.NL * i + j] += E[self.NJ * i + k] * F[self.NL * k + j]
# scop end


class _StrategyNumPy(_3mm):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, B: ndarray, C: ndarray, D: ndarray):
        for i in range(0, self.NI):
            for j in range(0, self.NK):
                A[i, j] = self.DATA_TYPE((i * j + 1) % self.NI) / (5 * self.NI)

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[i, j] = self.DATA_TYPE((i * (j + 1) + 2) % self.NJ) / (5 * self.NJ)

        for i in range(0, self.NJ):
            for j in range(0, self.NM):
                C[i, j] = self.DATA_TYPE(i * (j + 3) % self.NL) / (5 * self.NL)

        for i in range(0, self.NM):
            for j in range(0, self.NL):
                D[i, j] = self.DATA_TYPE((i * (j + 2) + 2) % self.NK) / (5 * self.NK)

    def print_array_custom(self, G: ndarray, name: str):
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(G[i, j])

    def kernel(self, E: ndarray, A: ndarray, B: ndarray, F: ndarray, C: ndarray, D: ndarray, G: ndarray):
# scop begin
        # E := A * B
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                E[i, j] = 0.0
                for k in range(0, self.NK):
                    E[i, j] += A[i, k] * B[k, j]

        # F := C * D
        for i in range(0, self.NJ):
            for j in range(0, self.NL):
                F[i, j] = 0.0
                for k in range(0, self.NM):
                    F[i, j] += C[i, k] * D[k, j]

        # G := E * F
        for i in range(0, self.NI):
            for j in range(0, self.NL):
                G[i, j] = 0.0
                for k in range(0, self.NJ):
                    G[i, j] += E[i, k] * F[k, j]
# scop end
