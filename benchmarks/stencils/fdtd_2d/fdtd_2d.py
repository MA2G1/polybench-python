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


class Fdtd_2d(PolyBench):

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
        self.TMAX = params.get('TMAX')
        self.NX = params.get('NX')
        self.NY = params.get('NY')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        ex = self.create_array(2, [self.NX, self.NY], self.DATA_TYPE(0))
        ey = self.create_array(2, [self.NX, self.NY], self.DATA_TYPE(0))
        hz = self.create_array(2, [self.NX, self.NY], self.DATA_TYPE(0))
        _fict_ = self.create_array(1, [self.TMAX], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(ex, ey, hz, _fict_)

        # Benchmark the kernel
        self.time_kernel(ex, ey, hz, _fict_)

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
        return [('ex', ex), ('ey', ey), ('hz', hz)]


class _StrategyList(Fdtd_2d):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, ex: list, ey: list, hz: list, _fict_: list):
        for i in range(0, self.TMAX):
            _fict_[i] = self.DATA_TYPE(i)

        for i in range(0, self.NX):
            for j in range(0, self.NY):
                ex[i][j] = (self.DATA_TYPE(i) * (j + 1)) / self.NX
                ey[i][j] = (self.DATA_TYPE(i) * (j + 2)) / self.NY
                hz[i][j] = (self.DATA_TYPE(i) * (j + 3)) / self.NX

    def print_array_custom(self, array: list, name: str):
        # Although this function will print three arrays (ex, ey and hz), the code required is the same.
        for i in range(0, self.NX):
            for j in range(0, self.NY):
                if (i * self.NX + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(array[i][j])

    def kernel(self, ex: list, ey: list, hz: list, _fict_: list):
# scop begin
        for t in range(0, self.TMAX):
            for j in range(0, self.NY):
                ey[0][j] = _fict_[t]

            for i in range(1, self.NX):
                for j in range(0, self.NY):
                    ey[i][j] = ey[i][j] - 0.5 * (hz[i][j]-hz[i-1][j])

            for i in range(0, self.NX):
                for j in range(1, self.NY):
                    ex[i][j] = ex[i][j] - 0.5 * (hz[i][j]-hz[i][j-1])

            for i in range(0, self.NX - 1):
                for j in range(0, self.NY - 1):
                    hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
# scop end


class _StrategyListFlattened(Fdtd_2d):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, ex: list, ey: list, hz: list, _fict_: list):
        for i in range(0, self.TMAX):
            _fict_[i] = self.DATA_TYPE(i)

        for i in range(0, self.NX):
            for j in range(0, self.NY):
                ex[self.NY * i + j] = (self.DATA_TYPE(i) * (j+1)) / self.NX
                ey[self.NY * i + j] = (self.DATA_TYPE(i) * (j+2)) / self.NY
                hz[self.NY * i + j] = (self.DATA_TYPE(i) * (j+3)) / self.NX

    def print_array_custom(self, array: list, name: str):
        # Although this function will print three arrays (ex, ey and hz), the code required is the same.
        for i in range(0, self.NX):
            for j in range(0, self.NY):
                if (i * self.NX + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(array[self.NY * i + j])

    def kernel(self, ex: list, ey: list, hz: list, _fict_: list):
# scop begin
        for t in range(0, self.TMAX):
            for j in range(0, self.NY):
                ey[self.NY * 0 + j] = _fict_[t]

            for i in range(1, self.NX):
                for j in range(0, self.NY):
                    ey[self.NY * i + j] = ey[self.NY * i + j] - 0.5 * (hz[self.NY * i + j] - hz[self.NY * (i - 1) + j])

            for i in range(0, self.NX):
                for j in range(1, self.NY):
                    ex[self.NY * i + j] = ex[self.NY * i + j] - 0.5 * (hz[self.NY * i + j] - hz[self.NY * i + j - 1])

            for i in range(0, self.NX - 1):
                for j in range(0, self.NY - 1):
                    hz[self.NY * i + j] = hz[self.NY * i + j] - 0.7 * (ex[self.NY * i + j + 1] - ex[self.NY * i + j] + ey[self.NY * (i + 1) + j] - ey[self.NY * i + j])
# scop end


class _StrategyNumPy(Fdtd_2d):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, ex: ndarray, ey: ndarray, hz: ndarray, _fict_: ndarray):
        for i in range(0, self.TMAX):
            _fict_[i] = self.DATA_TYPE(i)

        for i in range(0, self.NX):
            for j in range(0, self.NY):
                ex[i, j] = (self.DATA_TYPE(i) * (j+1)) / self.NX
                ey[i, j] = (self.DATA_TYPE(i) * (j+2)) / self.NY
                hz[i, j] = (self.DATA_TYPE(i) * (j+3)) / self.NX

    def print_array_custom(self, array: ndarray, name: str):
        # Although this function will print three arrays (ex, ey and hz), the code required is the same.
        for i in range(0, self.NX):
            for j in range(0, self.NY):
                if (i * self.NX + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(array[i, j])

    def kernel(self, ex: ndarray, ey: ndarray, hz: ndarray, _fict_: ndarray):
# scop begin
        for t in range(0, self.TMAX):
            for j in range(0, self.NY):
                ey[0, j] = _fict_[t]

            for i in range(1, self.NX):
                for j in range(0, self.NY):
                    ey[i, j] = ey[i, j] - 0.5 * (hz[i, j]-hz[i-1, j])

            for i in range(0, self.NX):
                for j in range(1, self.NY):
                    ex[i, j] = ex[i, j] - 0.5 * (hz[i, j]-hz[i, j-1])

            for i in range(0, self.NX - 1):
                for j in range(0, self.NY - 1):
                    hz[i, j] = hz[i, j] - 0.7 * (ex[i, j+1] - ex[i, j] + ey[i+1, j] - ey[i, j])
# scop end
