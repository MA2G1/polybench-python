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


class Gemm(PolyBench):

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

        # Adjust the print modifier according to the data type
        self.set_print_modifier(parameters.DataType)

        # Set up problem size from the given parameters (adapt this part with appropriate parameters)
        self.NI = params.get('NI')
        self.NJ = params.get('NJ')
        self.NK = params.get('NK')

    def initialize_array(self, C: list, A: list, B: list):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[i][j] = self.DATA_TYPE((i * j + 1) % self.NI) / self.NI

        for i in range(0, self.NI):
            for j in range (0, self.NK):
                A[i][j] = self.DATA_TYPE(i * (j + 1) % self.NK) / self.NK

        for i in range(0, self.NK):
            for j in range(0, self.NJ):
                B[i][j] = self.DATA_TYPE(i * (j + 2) % self.NJ) / self.NJ

    def print_array_custom(self, C: list):
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                if (i * self.NI + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(C[i][j])

    def kernel(self, alpha: float, beta: float, C: list, A: list, B: list):
# scop begin
        for i in range(0, self.NI):
            for j in range(0, self.NJ):
                C[i][j] *= beta

            for k in range(0, self.NK):
                for j in range(0, self.NJ):
                    C[i][j] += alpha * A[i][k] * B[k][j]
# scop end

    def run_benchmark(self):
        alpha = 1.5
        beta = 1.2

        # Create data structures (arrays, auxiliary variables, etc.)
        C = self.create_array(2, [self.NI, self.NJ])
        A = self.create_array(2, [self.NI, self.NK])
        B = self.create_array(2, [self.NK, self.NJ])

        # Initialize data structures
        self.initialize_array(C, A, B)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(alpha, beta, C, A, B)

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
        return [('C', C)]
