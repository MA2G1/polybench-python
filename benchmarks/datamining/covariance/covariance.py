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


class Covariance(PolyBench):

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

        # Set up problem size
        self.M = params.get('M')
        self.N = params.get('N')

    def run_benchmark(self):
        # Array creation
        float_n = float(self.N)  # we will need a floating point version of N

        data = self.create_array(2, [self.N, self.M], self.DATA_TYPE(0))
        cov = self.create_array(2, [self.M, self.M], self.DATA_TYPE(0))
        mean = self.create_array(1, [self.M], self.DATA_TYPE(0))

        # Initialize array(s)
        self.initialize_array(data)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(float_n, data, cov, mean)

        # Stop and print instruments
        self.stop_instruments()

        # Return printable data as a list of tuples ('name', value)
        return [('cov', cov)]

class _StrategyList(Covariance):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, data: list):
        for i in range(0, self.N):
            for j in range(0, self.M):
                data[i][j] = self.DATA_TYPE(i * j) / self.M

    def print_array_custom(self, cov: list, name: str):
            for i in range(0, self.M):
                for j in range(0, self.M):
                    if (i * self.M + j) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(cov[i][j])

    def kernel(self, float_n: float, data: list, cov: list, mean: list):
# scop begin
        for j in range(0, self.M):
            mean[j] = 0.0
            for i in range(0, self.N):
                mean[j] += data[i][j]
            mean[j] /= float_n

        for i in range(0, self.N):
            for j in range(0, self.M):
                data[i][j] -= mean[j]

        for i in range(0, self.M):
            for j in range(0, self.M):
                cov[i][j] = 0.0
                for k in range(0, self.N):
                    cov[i][j] += data[k][i] * data[k][j]
                cov[i][j] /= float_n - 1.0
                cov[j][i] = cov[i][j]
# scop end


class _StrategyListFlattened(Covariance):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, data: list):
        for i in range(0, self.N):
            for j in range(0, self.M):
                data[self.M * i + j] = self.DATA_TYPE(i * j) / self.M

    def print_array_custom(self, cov: list, name: str):
        for i in range(0, self.M):
            for j in range(0, self.M):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(cov[self.M * i + j])

    def kernel(self, float_n: float, data: list, cov: list, mean: list):
# scop begin
        for j in range(0, self.M):
            mean[j] = 0.0
            for i in range(0, self.N):
                mean[j] += data[self.M * i + j]
            mean[j] /= float_n

        for i in range(0, self.N):
            for j in range(0, self.M):
                data[self.M * i + j] -= mean[j]

        for i in range(0, self.M):
            for j in range(0, self.M):
                cov[self.M * i + j] = 0.0
                for k in range(0, self.N):
                    cov[self.M * i + j] += data[self.M * k + i] * data[self.M * k + j]
                cov[self.M * i + j] /= float_n - 1.0
                cov[self.M * j + i] = cov[self.M * i + j]
# scop end


class _StrategyNumPy(Covariance):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, data: ndarray):
        for i in range(0, self.N):
            for j in range(0, self.M):
                data[i, j] = self.DATA_TYPE(i * j) / self.M

    def print_array_custom(self, cov: ndarray, name: str):
        for i in range(0, self.M):
            for j in range(0, self.M):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(cov[i, j])

    def kernel(self, float_n: float, data: ndarray, cov: ndarray, mean: ndarray):
# scop begin
        for j in range(0, self.M):
            mean[j] = 0.0
            for i in range(0, self.N):
                mean[j] += data[i, j]
            mean[j] /= float_n

        for i in range(0, self.N):
            for j in range(0, self.M):
                data[i, j] -= mean[j]

        for i in range(0, self.M):
            for j in range(0, self.M):
                cov[i, j] = 0.0
                for k in range(0, self.N):
                    cov[i, j] += data[k, i] * data[k, j]
                cov[i, j] /= float_n - 1.0
                cov[j, i] = cov[i, j]
# scop end
