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

from benchmarks.polybench import PolyBench, PolyBenchParameters


class Mvt(PolyBench):

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options)

        # Validate inputs
        if not isinstance(parameters, PolyBenchParameters):
            raise AssertionError(f'Invalid parameter "parameters": "{parameters}"')

        # The parameters hold the necessary information obtained from "polybench.spec" file
        params = parameters.DataSets.get(self.DATASET_SIZE)
        if not isinstance(params, dict):
            raise NotImplementedError(f'Dataset size "{self.DATASET_SIZE.name}" not implemented '
                                      f'for {parameters.Category}/{parameters.Name}.')

        # Adjust the data type and print modifier according to the data type
        self.DATA_TYPE = parameters.DataType
        self.set_print_modifier(parameters.DataType)

        # Set up problem size from the given parameters (adapt this part with appropriate parameters)
        self.N = params.get('N')

    def initialize_array(self, x1: list, x2: list, y_1: list, y_2: list, A: list):
        for i in range(0, self.N):
            x1[i] = self.DATA_TYPE(i % self.N) / self.N
            x2[i] = self.DATA_TYPE((i + 1) % self.N) / self.N
            y_1[i] = self.DATA_TYPE((i + 3) % self.N) / self.N
            y_2[i] = self.DATA_TYPE((i + 4) % self.N) / self.N
            if self.POLYBENCH_FLATTEN_LISTS:
                for j in range(0, self.N):
                    A[self.N * i + j] = self.DATA_TYPE(i * j % self.N) / self.N
            else:
                for j in range(0, self.N):
                    A[i][j] = self.DATA_TYPE(i * j % self.N) / self.N

    def print_array_custom(self, array: list, name: str):
        # Although two arrays will be printed, they share the same format, so there is no need to check which one comes.
        for i in range(0, self.N):
            if i % 20 == 0:
                self.print_message('\n')
            self.print_value(array[i])

    def kernel(self, x1: list, x2: list, y_1: list, y_2: list, A: list):
# scop begin
        for i in range(0, self.N):
            for j in range(0, self.N):
                x1[i] = x1[i] + A[i][j] * y_1[j]

        for i in range(0, self.N):
            for j in range(0, self.N):
                x2[i] = x2[i] + A[j][i] * y_2[j]
# scop end

    def kernel_flat(self, x1: list, x2: list, y_1: list, y_2: list, A: list):
# scop begin
        for i in range(0, self.N):
            for j in range(0, self.N):
                x1[i] = x1[i] + A[self.N * i + j] * y_1[j]

        for i in range(0, self.N):
            for j in range(0, self.N):
                x2[i] = x2[i] + A[self.N * j + i] * y_2[j]
# scop end

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        if self.POLYBENCH_FLATTEN_LISTS:
            A = self.create_array(1, [self.N * self.N], self.DATA_TYPE(0))
        else:
            A = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        x1 = self.create_array(1, [self.N], self.DATA_TYPE(0))
        x2 = self.create_array(1, [self.N], self.DATA_TYPE(0))
        y_1 = self.create_array(1, [self.N], self.DATA_TYPE(0))
        y_2 = self.create_array(1, [self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(x1, x2, y_1, y_2, A)

        if self.POLYBENCH_FLATTEN_LISTS:
            # Start instruments
            self.start_instruments()
            # Run kernel
            self.kernel_flat(x1, x2, y_1, y_2, A)
            # Stop and print instruments
            self.stop_instruments()
        else:
            # Start instruments
            self.start_instruments()
            # Run kernel
            self.kernel(x1, x2, y_1, y_2, A)
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
        return [('x1', x1), ('x2', x2)]
