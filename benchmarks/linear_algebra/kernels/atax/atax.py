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


class Atax(PolyBench):

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
        self.M = params.get('M')
        self.N = params.get('N')

    def print_array_custom(self, y: list, name: str):
        for i in range(0, self.N):
            if i % 20 == 0:
                self.print_message('\n')
            self.print_value(y[i])

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        A = self.create_array(2, [self.M, self.N], self.DATA_TYPE(0))
        x = self.create_array(1, [self.N])
        y = self.create_array(1, [self.N])
        tmp = self.create_array(1, [self.M])

        # Initialize data structures
        self.initialize_array(A, x)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(A, x, y, tmp)

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
        return [('y', y)]


class _StrategyList(Atax):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, x: list):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            x[i] = 1 + (i / fn)

        for i in range(0, self.M):
            for j in range(0, self.N):
                A[i][j] = self.DATA_TYPE((i + j) % self.N) / (5 * self.M)

    def kernel(self, A: list, x: list, y: list, tmp: list):
# scop begin
        for i in range(0, self.N):
            y[i] = 0

        for i in range(0, self.M):
            tmp[i] = 0.0
            for j in range(0, self.N):
                tmp[i] = tmp[i] + A[i][j] * x[j]

            for j in range(0, self.N):
                y[j] = y[j] + A[i][j] * tmp[i]
# scop end


class _StrategyListFlattened(Atax):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, x: list):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            x[i] = 1 + (i / fn)

        for i in range(0, self.M):
            for j in range(0, self.N):
                A[self.N * i + j] = self.DATA_TYPE((i + j) % self.N) / (5 * self.M)

    def kernel(self, A: list, x: list, y: list, tmp: list):
# scop begin
        for i in range(0, self.N):
            y[i] = 0

        for i in range(0, self.M):
            tmp[i] = 0.0
            for j in range(0, self.N):
                tmp[i] = tmp[i] + A[self.N * i + j] * x[j]

            for j in range(0, self.N):
                y[j] = y[j] + A[self.N * i + j] * tmp[i]
# scop end


class _StrategyNumPy(Atax):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, x: ndarray):
        fn = self.DATA_TYPE(self.N)

        for i in range(0, self.N):
            x[i] = 1 + (i / fn)

        for i in range(0, self.M):
            for j in range(0, self.N):
                A[i, j] = self.DATA_TYPE((i + j) % self.N) / (5 * self.M)

    def kernel(self, A: ndarray, x: ndarray, y: ndarray, tmp: ndarray):
# scop begin
        for i in range(0, self.N):
            y[i] = 0

        for i in range(0, self.M):
            tmp[i] = 0.0
            for j in range(0, self.N):
                tmp[i] = tmp[i] + A[i, j] * x[j]

            for j in range(0, self.N):
                y[j] = y[j] + A[i, j] * tmp[i]
# scop end
