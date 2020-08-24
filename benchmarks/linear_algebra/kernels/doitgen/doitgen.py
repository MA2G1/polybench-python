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


class Doitgen(PolyBench):

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
        self.NQ = params.get('NQ')
        self.NR = params.get('NR')
        self.NP = params.get('NP')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        A = self.create_array(3, [self.NR, self.NQ, self.NP], self.DATA_TYPE(0))
        C4 = self.create_array(2, [self.NP, self.NP], self.DATA_TYPE(0))
        sum = self.create_array(1, [self.NP], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, C4)

        # Benchmark the kernel
        self.time_kernel(A, C4, sum)

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


class _StrategyList(Doitgen):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, C4: list):
        for i in range(0, self.NR):
            for j in range(0, self.NQ):
                for k in range(0, self.NP):
                    A[i][j][k] = self.DATA_TYPE((i * j + k) % self.NP) / self.NP

        for i in range(0, self.NP):
            for j in range(0, self.NP):
                C4[i][j] = self.DATA_TYPE(i * j % self.NP) / self.NP

    def print_array_custom(self, A: list, name: str):
        for i in range(0, self.NR):
            for j in range(0, self.NQ):
                for k in range(0, self.NP):
                    if (i * self.NQ * self.NP + j * self.NP + k) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(A[i][j][k])

    def kernel(self, A: list, C4: list, sum: list):
# scop begin
        for r in range(0, self.NR):
            for q in range(self.NQ):
                for p in range(0, self.NP):
                    sum[p] = 0.0
                    for s in range(self.NP):
                        sum[p] += A[r][q][s] * C4[s][p]

                for p in range(0, self.NP):
                    A[r][q][p] = sum[p]
# scop end


class _StrategyListFlattened(Doitgen):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, C4: list):
        for i in range(0, self.NR):
            for j in range(0, self.NQ):
                for k in range(0, self.NP):
                    A[(self.NQ * i + j) * self.NP + k] = self.DATA_TYPE((i * j + k) % self.NP) / self.NP

        for i in range(0, self.NP):
            for j in range(0, self.NP):
                C4[self.NP * i + j] = self.DATA_TYPE(i * j % self.NP) / self.NP

    def print_array_custom(self, A: list, name: str):
        for i in range(0, self.NR):
            for j in range(0, self.NQ):
                for k in range(0, self.NP):
                    if (i * self.NQ * self.NP + j * self.NP + k) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(A[(self.NQ * i + j) * self.NP + k])

    def kernel(self, A: list, C4: list, sum: list):
# scop begin
        for r in range(0, self.NR):
            for q in range(self.NQ):
                for p in range(0, self.NP):
                    sum[p] = 0.0
                    for s in range(self.NP):
                        sum[p] += A[(self.NQ * r + q) * self.NP + s] * C4[self.NP * s + p]

                for p in range(0, self.NP):
                    A[(self.NQ * r + q) * self.NP + p] = sum[p]
# scop end


class _StrategyNumPy(Doitgen):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, C4: ndarray):
        for i in range(0, self.NR):
            for j in range(0, self.NQ):
                for k in range(0, self.NP):
                    A[i, j, k] = self.DATA_TYPE((i * j + k) % self.NP) / self.NP

        for i in range(0, self.NP):
            for j in range(0, self.NP):
                C4[i, j] = self.DATA_TYPE(i * j % self.NP) / self.NP

    def print_array_custom(self, A: ndarray, name: str):
        for i in range(0, self.NR):
            for j in range(0, self.NQ):
                for k in range(0, self.NP):
                    if (i * self.NQ * self.NP + j * self.NP + k) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(A[i, j, k])

    def kernel(self, A: ndarray, C4: ndarray, sum: ndarray):
# scop begin
        for r in range(0, self.NR):
            for q in range(self.NQ):
                for p in range(0, self.NP):
                    sum[p] = 0.0
                    for s in range(self.NP):
                        sum[p] += A[r, q, s] * C4[s, p]

                for p in range(0, self.NP):
                    A[r, q, p] = sum[p]
# scop end
