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


class Adi(PolyBench):

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
        self.TSTEPS = params.get('TSTEPS')
        self.N = params.get('N')

    def initialize_array(self, u: list):
        for i in range(0, self.N):
            for j in range(0, self.N):
                u[i][j] = self.DATA_TYPE(i + self.N - j) / self.N

    def print_array_custom(self, u: list, name: str):
        for i in range(0, self.N):
            for j in range(0, self.N):
                if (i * self.N + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(u[i][j])

    def kernel(self, u: list, v: list, p: list, q: list):
#scop begin
        DX = 1.0 / self.DATA_TYPE(self.N)
        DY = 1.0 / self.DATA_TYPE(self.N)
        DT = 1.0 / self.DATA_TYPE(self.TSTEPS)
        B1 = 2.0
        B2 = 1.0
        mul1 = B1 * DT / (DX * DX)
        mul2 = B2 * DT / (DY * DY)

        a = -mul1 / 2.0
        b = 1.0 + mul1
        c = a
        d = -mul2 / 2.0
        e = 1.0 + mul2
        f = d

        for t in range(1, self.TSTEPS + 1):
            # Column Sweep
            for i in range(1, self.N - 1):
                v[0][i] = 1.0
                p[i][0] = 0.0
                q[i][0] = v[0][i]
                for j in range(1, self.N - 1):
                    p[i][j] = -c / (a * p[i][j-1]+b)
                    q[i][j] = (-d * u[j][i-1]+(1.0+2.0 * d) * u[j][i] - f * u[j][i+1]-a * q[i][j-1]) / (a * p[i][j-1]+b)

                v[self.N-1][i] = 1.0
                for j in range(self.N - 2, 0, -1):
                    v[j][i] = p[i][j] * v[j+1][i] + q[i][j]

            # Row Sweep
            for i in range(1, self.N - 1):
                u[i][0] = 1.0
                p[i][0] = 0.0
                q[i][0] = u[i][0]
                for j in range(1, self.N - 1):
                    p[i][j] = -f / (d * p[i][j-1]+e)
                    q[i][j] = (-a * v[i-1][j]+(1.0+2.0 * a) * v[i][j] - c * v[i+1][j]-d * q[i][j-1]) / (d * p[i][j-1]+e)

                u[i][self.N-1] = 1.0
                for j in range(self.N - 2, 0, -1):
                    u[i][j] = p[i][j] * u[i][j+1] + q[i][j]
#scop end

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        u = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        v = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        p = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))
        q = self.create_array(2, [self.N, self.N], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(u)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(u, v, p, q)

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
        return [('u', u)]
