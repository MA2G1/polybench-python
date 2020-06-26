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


class Floyd_warshall(PolyBench):

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

    def initialize_array(self, path: list):
        if self.POLYBENCH_FLATTEN_LISTS:
            for i in range(0, self.N):
                for j in range(0, self.N):
                    path[self.N * i + j] = i * j % 7 + 1
                    if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                        path[self.N * i + j] = 999
        else:
            for i in range(0, self.N):
                for j in range(0, self.N):
                    path[i][j] = i * j % 7 + 1
                    if (i + j) % 13 == 0 or (i + j) % 7 == 0 or (i + j) % 11 == 0:
                        path[i][j] = 999

    def print_array_custom(self, path: list, name: str):
        if self.POLYBENCH_FLATTEN_LISTS:
            for i in range(0, self.N):
                for j in range(0, self.N):
                    if (i * self.N + j) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(path[self.N * i + j])
        else:
            for i in range(0, self.N):
                for j in range(0, self.N):
                    if (i * self.N + j) % 20 == 0:
                        self.print_message('\n')
                    self.print_value(path[i][j])

    def kernel(self, path: list):
# scop begin
        for k in range(0, self.N):
            for i in range(0, self.N):
                for j in range(0, self.N):
                    if path[i][j] < path[i][k] + path[k][j]:
                        path[i][j] = path[i][j]
                    else:
                        path[i][j] = path[i][k] + path[k][j]
# scop end

    def kernel_flat(self, path: list):
# scop begin
        for k in range(0, self.N):
            for i in range(0, self.N):
                for j in range(0, self.N):
                    if path[self.N * i + j] < path[self.N * i + k] + path[self.N * k + j]:
                        path[self.N * i + j] = path[self.N * i + j]
                    else:
                        path[self.N * i + j] = path[self.N * i + k] + path[self.N * k + j]
# scop end

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        if self.POLYBENCH_FLATTEN_LISTS:
            path = self.create_array(1, [self.N * self.N], self.DATA_TYPE(0))
        else:
            path = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(path)

        if self.POLYBENCH_FLATTEN_LISTS:
            # Start instruments
            self.start_instruments()
            # Run kernel
            self.kernel_flat(path)
            # Stop and print instruments
            self.stop_instruments()
        else:
            # Start instruments
            self.start_instruments()
            # Run kernel
            self.kernel(path)
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
        return [('path', path)]
