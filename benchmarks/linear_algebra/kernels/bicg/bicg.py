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


class Bicg(PolyBench):

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
        self.M = params.get('M')
        self.N = params.get('N')

    def print_array_custom(self, array: list, name: list):
        if name == 's':
            loop_bound = self.M
        else:
            loop_bound = self.N

        for i in range(0, loop_bound):
            if i % 20 == 0:
                self.print_message('\n')
            self.print_value(array[i])

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        A = self.create_array(2, [self.N, self.M], self.DATA_TYPE(0))
        s = self.create_array(1, [self.M], self.DATA_TYPE(0))
        q = self.create_array(1, [self.N], self.DATA_TYPE(0))
        p = self.create_array(1, [self.M], self.DATA_TYPE(0))
        r = self.create_array(1, [self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, r, p)

        # Benchmark the kernel
        self.time_kernel(A, s, q, p, r)

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
        return [('s', s), ('q', q)]


class _StrategyList(Bicg):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, r: list, p: list):
        for i in range(0, self.M):
            p[i] = self.DATA_TYPE(i % self.M) / self.M

        for i in range(0, self.N):
            r[i] = self.DATA_TYPE(i % self.N) / self.N
            for j in range(0, self.M):
                A[i][j] = self.DATA_TYPE(i * (j+1) % self.N) / self.N

    def kernel(self, A: list, s: list, q: list, p: list, r: list):
# scop begin
        for i in range(0, self.M):
            s[i] = 0

        for i in range(0, self.N):
            q[i] = 0.0
            for j in range(0, self.M):
                s[j] = s[j] + r[i] * A[i][j]
                q[i] = q[i] + A[i][j] * p[j]
# scop end


class _StrategyListFlattened(Bicg):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, r: list, p: list):
        for i in range(0, self.M):
            p[i] = self.DATA_TYPE(i % self.M) / self.M

        for i in range(0, self.N):
            r[i] = self.DATA_TYPE(i % self.N) / self.N
            for j in range(0, self.M):
                A[self.M * i + j] = self.DATA_TYPE(i * (j + 1) % self.N) / self.N

    def kernel(self, A: list, s: list, q: list, p: list, r: list):
# scop begin
        for i in range(0, self.M):
            s[i] = 0

        for i in range(0, self.N):
            q[i] = 0.0
            for j in range(0, self.M):
                s[j] = s[j] + r[i] * A[self.M * i + j]
                q[i] = q[i] + A[self.M * i + j] * p[j]
# scop end


class _StrategyNumPy(Bicg):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, r: ndarray, p: ndarray):
        for i in range(0, self.M):
            p[i] = self.DATA_TYPE(i % self.M) / self.M

        for i in range(0, self.N):
            r[i] = self.DATA_TYPE(i % self.N) / self.N
            for j in range(0, self.M):
                A[i, j] = self.DATA_TYPE(i * (j + 1) % self.N) / self.N

    def kernel(self, A: ndarray, s: ndarray, q: ndarray, p: ndarray, r: ndarray):
# scop begin
        for i in range(0, self.M):
            s[i] = 0

        for i in range(0, self.N):
            q[i] = 0.0
            for j in range(0, self.M):
                s[j] = s[j] + r[i] * A[i, j]
                q[i] = q[i] + A[i, j] * p[j]
# scop end
