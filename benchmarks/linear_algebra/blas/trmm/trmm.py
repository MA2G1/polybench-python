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
from benchmarks.polybench_classes import PolyBenchParameters
from benchmarks.polybench_options import ArrayImplementation
from numpy.core.multiarray import ndarray


class Trmm(PolyBench):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        implementation = options['array_implementation']
        if implementation == ArrayImplementation.LIST:
            return _TrmmList.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.LIST_FLATTENED:
            return _TrmmListFlattened.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.NUMPY:
            return _TrmmNumPy.__new__(cls, options, parameters)

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
        self.M = params.get('M')
        self.N = params.get('N')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        alpha = 1.5

        A = self.create_array(2, [self.M, self.M], self.DATA_TYPE(0))
        B = self.create_array(2, [self.M, self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(A, B)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(alpha, A, B)

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
        return [('B', B)]


class _TrmmList(Trmm):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_TrmmList)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list):
        for i in range(0, self.M):
            for j in range(0, i):
                A[i][j] = self.DATA_TYPE((i + j) % self.M) / self.M

            A[i][i] = 1.0

            for j in range(0, self.N):
                B[i][j] = self.DATA_TYPE((self.N + (i - j)) % self.N) / self.N

    def print_array_custom(self, B: list, name: str):
        for i in range(0, self.M):
            for j in range(0, self.N):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(B[i][j])

    def kernel(self, alpha, A: list, B: list):
        # BLAS parameters
        # SIDE   = 'L'
        # UPLO   = 'L'
        # TRANSA = 'T'
        # DIAG   = 'U'
        # = > Form  B := alpha * A ** T * B.
        # A is MxM
        # B is MxN
# scrop begin
        for i in range(0, self.M):
            for j in range(0, self.N):
                for k in range(i + 1, self.M):
                    B[i][j] += A[k][i] * B[k][j]
                B[i][j] = alpha * B[i][j]
# scop end


class _TrmmListFlattened(Trmm):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_TrmmListFlattened)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, A: list, B: list):
        for i in range(0, self.M):
            for j in range(0, i):
                A[self.M * i + j] = self.DATA_TYPE((i+j) % self.M) / self.M

            A[self.M * i + i] = 1.0

            for j in range(0, self.N):
                B[self.N * i + j] = self.DATA_TYPE((self.N+(i-j)) % self.N) / self.N

    def print_array_custom(self, B: list, name: str):
        for i in range(0, self.M):
            for j in range(0, self.N):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(B[self.N * i + j])

    def kernel(self, alpha, A: list, B: list):
# scrop begin
        for i in range(0, self.M):
            for j in range(0, self.N):
                for k in range(i + 1, self.M):
                    B[self.N * i + j] += A[self.M * k + i] * B[self.N * k + j]
                B[self.N * i + j] = alpha * B[self.N * i + j]
# scop end


class _TrmmNumPy(Trmm):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_TrmmNumPy)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, A: ndarray, B: ndarray):
        for i in range(0, self.M):
            for j in range(0, i):
                A[i, j] = self.DATA_TYPE((i + j) % self.M) / self.M

            A[i, i] = 1.0

            for j in range(0, self.N):
                B[i, j] = self.DATA_TYPE((self.N + (i - j)) % self.N) / self.N

    def print_array_custom(self, B: ndarray, name: str):
        for i in range(0, self.M):
            for j in range(0, self.N):
                if (i * self.M + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(B[i, j])

    def kernel(self, alpha, A: ndarray, B: ndarray):
        # BLAS parameters
        # SIDE   = 'L'
        # UPLO   = 'L'
        # TRANSA = 'T'
        # DIAG   = 'U'
        # = > Form  B := alpha * A ** T * B.
        # A is MxM
        # B is MxN
# scrop begin
        for i in range(0, self.M):
            for j in range(0, self.N):
                for k in range(i + 1, self.M):
                    B[i, j] += A[k, i] * B[k, j]
                B[i, j] = alpha * B[i, j]
# scop end
