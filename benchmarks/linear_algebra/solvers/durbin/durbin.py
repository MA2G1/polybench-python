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
from benchmarks.polybench_classes import PolyBenchOptions, PolyBenchSpec


class Durbin(PolyBench):

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

        # The parameters hold the necessary information obtained from "polybench.spec" file
        params = parameters.DataSets.get(self.DATASET_SIZE)
        if not isinstance(params, dict):
            raise NotImplementedError(f'Dataset size "{self.DATASET_SIZE.name}" not implemented '
                                      f'for {parameters.Category}/{parameters.Name}.')

        # Set up problem size from the given parameters (adapt this part with appropriate parameters)
        self.N = params.get('N')

    def initialize_array(self, r: list):
        for i in range(0, self.N):
            r[i] = self.N + 1 - i

    def print_array_custom(self, y: list, name: str):
        for i in range(0, self.N):
            if i % 20 == 0:
                self.print_message('\n')
            self.print_value(y[i])

    def kernel(self, r: list, y: list):
        z = self.create_array(1, [self.N], self.DATA_TYPE(0))
# scop begin
        y[0] = -r[0]
        beta = 1.0
        alpha = -r[0]

        for k in range(1, self.N):
            beta = (1-alpha * alpha) * beta
            summ = 0.0

            for i in range(0, k):
                summ += r[k-i-1] * y[i]

            alpha = -(r[k] + summ) / beta

            for i in range(0, k):
                z[i] = y[i] + alpha * y[k-i-1]

            for i in range(0, k):
                y[i] = z[i]

            y[k] = alpha
# scop end

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        r = self.create_array(1, [self.N], self.DATA_TYPE(0))
        y = self.create_array(1, [self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(r)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(r, y)

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
