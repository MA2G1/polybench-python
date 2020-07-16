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


class Nussinov(PolyBench):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        implementation = options['array_implementation']
        if implementation == ArrayImplementation.LIST:
            return _NussinovList.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.LIST_FLATTENED:
            return _NussinovListFlattened.__new__(cls, options, parameters)
        elif implementation == ArrayImplementation.NUMPY:
            return _NussinovNumPy.__new__(cls, options, parameters)

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

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        seq = self.create_array(1, [self.N], int(0))  # base type = char; = int in Python
        table = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(seq, table)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(seq, table)

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
        return [('table', table)]


class _NussinovList(Nussinov):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_NussinovList)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, seq: list, table: list):
        for i in range(0, self.N):
            seq[i] = (i + 1) % 4  # right side is AGCT/0..3

        for i in range(0, self.N):
            for j in range(0, self.N):
                table[i][j] = self.DATA_TYPE(0)

    def print_array_custom(self, table: list, name: str):
        t = 0
        for i in range(0, self.N):
            for j in range(i, self.N):
                if t % 20 == 0:
                    self.print_message('\n')
                self.print_value(table[i][j])
                t += 1

    def kernel(self, seq: list, table: list):
        # def max_score(s1, s2):
        #     if s1 >= s2:
        #         return s1
        #     else:
        #         return s2
        #
        # def match(b1, b2):
        #     if b1 + b2 == 3:
        #         return 1
        #     else:
        #         return 0

# scop begin
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1, self.N):
                if j - 1 >= 0:
                    # table[i][j] = max_score(table[i][j], table[i][j - 1])
                    # NOTE: expanded macro max_score
                    if table[i][j] >= table[i][j - 1]:
                       table[i][j] = table[i][j]
                    else:
                       table[i][j] = table[i][j - 1]
                if i+1 < self.N:
                    # table[i][j] = max_score(table[i][j], table[i + 1][j])
                    # NOTE: expanded macro max_score
                    if table[i][j] >= table[i + 1][j]:
                        table[i][j] = table[i][j]
                    else:
                        table[i][j] = table[i + 1][j]

                if j - 1 >= 0 and i + 1 < self.N:
                    # don't allow adjacent elements to bond
                    if i < j - 1:
                        # table[i][j] = max_score(table[i][j], table[i + 1][j - 1] + match(seq[i], seq[j]))
                        # NOTE: expand macro match first
                        if seq[i] + seq[j] == 3:
                            # NOTE: expanded macro max_score; match = +1
                            if table[i][j] >= table[i + 1][j - 1] + 1:
                                table[i][j] = table[i][j]
                            else:
                                table[i][j] = table[i + 1][j - 1] + 1
                        else:
                            # NOTE: expanded macro max_score; match = +0
                            if table[i][j] >= table[i + 1][j - 1] + 0:
                                table[i][j] = table[i][j]
                            else:
                                table[i][j] = table[i + 1][j - 1] + 0
                    else:
                        # table[i][j] = max_score(table[i][j], table[i + 1][j - 1])
                        # NOTE: expanded macro max_score
                        if table[i][j] >= table[i+1][j-1]:
                            table[i][j] = table[i][j]
                        else:
                            table[i][j] = table[i+1][j-1]

                for k in range(i + 1, j):
                    # table[i][j] = max_score(table[i][j], table[i][k] + table[k + 1][j])
                    # NOTE: expanded macro max_score
                    if table[i][j] >= table[i][k] + table[k+1][j]:
                        table[i][j] = table[i][j]
                    else:
                        table[i][j] = table[i][k] + table[k+1][j]
# scop end


class _NussinovListFlattened(Nussinov):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_NussinovListFlattened)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, seq: list, table: list):
        for i in range(0, self.N):
            seq[i] = (i + 1) % 4  # right side is AGCT/0..3

        for i in range(0, self.N):
            for j in range(0, self.N):
                table[self.N * i + j] = self.DATA_TYPE(0)

    def print_array_custom(self, table: list, name: str):
        t = 0
        for i in range(0, self.N):
            for j in range(i, self.N):
                if t % 20 == 0:
                    self.print_message('\n')
                self.print_value(table[self.N * i + j])
                t += 1

    def kernel(self, seq: list, table: list):
# scop begin
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1, self.N):
                if j - 1 >= 0:
                    # table[i][j] = max_score(table[i][j], table[i][j - 1])
                    # NOTE: expanded macro max_score
                    if table[self.N * i + j] >= table[self.N * i + j - 1]:
                        table[self.N * i + j] = table[self.N * i + j]
                    else:
                        table[self.N * i + j] = table[self.N * i + j - 1]
                if i + 1 < self.N:
                    # table[i][j] = max_score(table[i][j], table[i + 1][j])
                    # NOTE: expanded macro max_score
                    if table[self.N * i + j] >= table[self.N * (i + 1) + j]:
                        table[self.N * i + j] = table[self.N * i + j]
                    else:
                        table[self.N * i + j] = table[self.N * (i + 1) + j]

                if j - 1 >= 0 and i + 1 < self.N:
                    # don't allow adjacent elements to bond
                    if i < j - 1:
                        # table[i][j] = max_score(table[i][j], table[i + 1][j - 1] + match(seq[i], seq[j]))
                        # NOTE: expand macro match first
                        if seq[i] + seq[j] == 3:
                            # NOTE: expanded macro max_score; match = +1
                            if table[self.N * i + j] >= table[self.N * (i + 1) + j - 1] + 1:
                                table[self.N * i + j] = table[self.N * i + j]
                            else:
                                table[self.N * i + j] = table[self.N * (i + 1) + j - 1] + 1
                        else:
                            # NOTE: expanded macro max_score; match = +0
                            if table[self.N * i + j] >= table[self.N * (i + 1) + j - 1] + 0:
                                table[self.N * i + j] = table[self.N * i + j]
                            else:
                                table[self.N * i + j] = table[self.N * (i + 1) + j - 1] + 0
                    else:
                        # table[i][j] = max_score(table[i][j], table[i + 1][j - 1])
                        # NOTE: expanded macro max_score
                        if table[self.N * i + j] >= table[self.N * (i + 1) + j - 1]:
                            table[self.N * i + j] = table[self.N * i + j]
                        else:
                            table[self.N * i + j] = table[self.N * (i + 1) + j - 1]

                for k in range(i + 1, j):
                    # table[i][j] = max_score(table[i][j], table[i][k] + table[k + 1][j])
                    # NOTE: expanded macro max_score
                    if table[self.N * i + j] >= table[self.N * i + k] + table[self.N * (k + 1) + j]:
                        table[self.N * i + j] = table[self.N * i + j]
                    else:
                        table[self.N * i + j] = table[self.N * i + k] + table[self.N * (k + 1) + j]
# scop end


class _NussinovNumPy(Nussinov):

    def __new__(cls, options: dict, parameters: PolyBenchParameters):
        return object.__new__(_NussinovNumPy)

    def __init__(self, options: dict, parameters: PolyBenchParameters):
        super().__init__(options, parameters)

    def initialize_array(self, seq: ndarray, table: ndarray):
        for i in range(0, self.N):
            seq[i] = (i + 1) % 4  # right side is AGCT/0..3

        for i in range(0, self.N):
            for j in range(0, self.N):
                table[i, j] = self.DATA_TYPE(0)

    def print_array_custom(self, table: ndarray, name: str):
        t = 0
        for i in range(0, self.N):
            for j in range(i, self.N):
                if t % 20 == 0:
                    self.print_message('\n')
                self.print_value(table[i, j])
                t += 1

    def kernel(self, seq: ndarray, table: ndarray):
        # def max_score(s1, s2):
        #     if s1 >= s2:
        #         return s1
        #     else:
        #         return s2
        #
        # def match(b1, b2):
        #     if b1 + b2 == 3:
        #         return 1
        #     else:
        #         return 0

# scop begin
        for i in range(self.N - 1, -1, -1):
            for j in range(i + 1, self.N):
                if j - 1 >= 0:
                    # table[i][j] = max_score(table[i][j], table[i][j - 1])
                    # NOTE: expanded macro max_score
                    if table[i, j] >= table[i, j - 1]:
                        table[i, j] = table[i, j]
                    else:
                        table[i, j] = table[i, j - 1]
                if i + 1 < self.N:
                    # table[i][j] = max_score(table[i][j], table[i + 1][j])
                    # NOTE: expanded macro max_score
                    if table[i, j] >= table[i + 1, j]:
                        table[i, j] = table[i, j]
                    else:
                        table[i, j] = table[i + 1, j]

                if j - 1 >= 0 and i + 1 < self.N:
                    # don't allow adjacent elements to bond
                    if i < j - 1:
                        # table[i][j] = max_score(table[i][j], table[i + 1][j - 1] + match(seq[i], seq[j]))
                        # NOTE: expand macro match first
                        if seq[i] + seq[j] == 3:
                            # NOTE: expanded macro max_score; match = +1
                            if table[i, j] >= table[i + 1, j - 1] + 1:
                                table[i, j] = table[i, j]
                            else:
                                table[i, j] = table[i + 1, j - 1] + 1
                        else:
                            # NOTE: expanded macro max_score; match = +0
                            if table[i, j] >= table[i + 1, j - 1] + 0:
                                table[i, j] = table[i, j]
                            else:
                                table[i, j] = table[i + 1, j - 1] + 0
                    else:
                        # table[i][j] = max_score(table[i][j], table[i + 1][j - 1])
                        # NOTE: expanded macro max_score
                        if table[i, j] >= table[i + 1, j - 1]:
                            table[i, j] = table[i, j]
                        else:
                            table[i, j] = table[i + 1, j - 1]

                for k in range(i + 1, j):
                    # table[i][j] = max_score(table[i][j], table[i][k] + table[k + 1][j])
                    # NOTE: expanded macro max_score
                    if table[i, j] >= table[i, k] + table[k + 1, j]:
                        table[i, j] = table[i, j]
                    else:
                        table[i, j] = table[i, k] + table[k + 1, j]
# scop end
