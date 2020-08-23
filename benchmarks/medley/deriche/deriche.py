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
import math


class Deriche(PolyBench):

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

        # Set up problem size from the given parameters (adapt this part with appropriate parameters)
        self.W = params.get('W')
        self.H = params.get('H')

    def run_benchmark(self):
        # Create data structures (arrays, auxiliary variables, etc.)
        alpha = 0.25

        imgIn = self.create_array(2, [self.W, self.H], self.DATA_TYPE(0))
        imgOut = self.create_array(2, [self.W, self.H], self.DATA_TYPE(0))
        y1 = self.create_array(2, [self.W, self.H], self.DATA_TYPE(0))
        y2 = self.create_array(2, [self.W, self.H], self.DATA_TYPE(0))

        # Initialize data structures
        self.initialize_array(imgIn, imgOut)

        # Start instruments
        self.start_instruments()

        # Run kernel
        self.kernel(alpha, imgIn, imgOut, y1, y2)

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
        return [('imgOut', imgOut)]


class _StrategyList(Deriche):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyList)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, imgIn: list, imgOut: list):
        # input should be between 0 and 1 (grayscale image pixel)
        for i in range(0, self.W):
            for j in range(0, self.H):
                imgIn[i][j] = self.DATA_TYPE((313 * i + 991 * j) % 65536) / 65535.0

    def print_array_custom(self, imgOut: list, name: str):
        for i in range(0, self.W):
            for j in range(0, self.H):
                if (i * self.H + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(imgOut[i][j])

    def kernel(self, alpha, imgIn: list, imgOut: list, y1: list, y2: list):
# scop begin
        k = (1.0 - math.exp(-alpha)) * (1.0 - math.exp(-alpha)) / (
            1.0 + 2.0 * alpha * math.exp(-alpha) - math.exp(2.0 * alpha))
        a1 = a5 = k
        a2 = a6 = k * math.exp(-alpha) * (alpha - 1.0)
        a3 = a7 = k * math.exp(-alpha) * (alpha + 1.0)
        a4 = a8 = -k * math.exp(-2.0 * alpha)
        b1 = math.pow(2.0, -alpha)
        b2 = -math.exp(-2.0 * alpha)
        c1 = c2 = 1

        for i in range(0, self.W):
            ym1 = 0.0
            ym2 = 0.0
            xm1 = 0.0
            for j in range(0, self.H):
                y1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2
                xm1 = imgIn[i][j]
                ym2 = ym1
                ym1 = y1[i][j]

        for i in range(0, self.W):
            yp1 = 0.0
            yp2 = 0.0
            xp1 = 0.0
            xp2 = 0.0
            for j in range(self.H - 1, -1, -1):
                y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
                xp2 = xp1
                xp1 = imgIn[i][j]
                yp2 = yp1
                yp1 = y2[i][j]

        for i in range(0, self.W):
            for j in range(0, self.H):
                imgOut[i][j] = c1 * (y1[i][j] + y2[i][j])

        for j in range(0, self.H):
            tm1 = 0.0
            ym1 = 0.0
            ym2 = 0.0
            for i in range(0, self.W):
                y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2
                tm1 = imgOut[i][j]
                ym2 = ym1
                ym1 = y1[i][j]

        for j in range(0, self.H):
            tp1 = 0.0
            tp2 = 0.0
            yp1 = 0.0
            yp2 = 0.0
            for i in range(self.W - 1, -1, -1):
                y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
                tp2 = tp1
                tp1 = imgOut[i][j]
                yp2 = yp1
                yp1 = y2[i][j]

        for i in range(0, self.W):
            for j in range(0, self.H):
                imgOut[i][j] = c2 * (y1[i][j] + y2[i][j])
# scop end


class _StrategyListFlattened(Deriche):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyListFlattened)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, imgIn: list, imgOut: list):
        # input should be between 0 and 1 (grayscale image pixel)
        for i in range(0, self.W):
            for j in range(0, self.H):
                imgIn[self.H * i + j] = self.DATA_TYPE((313 * i + 991 * j) % 65536) / 65535.0

    def print_array_custom(self, imgOut: list, name: str):
        for i in range(0, self.W):
            for j in range(0, self.H):
                if (i * self.H + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(imgOut[self.H * i + j])

    def kernel(self, alpha, imgIn: list, imgOut: list, y1: list, y2: list):
# scop begin
        k = (1.0 - math.exp(-alpha)) * (1.0 - math.exp(-alpha)) / (
                1.0 + 2.0 * alpha * math.exp(-alpha) - math.exp(2.0 * alpha))
        a1 = a5 = k
        a2 = a6 = k * math.exp(-alpha) * (alpha - 1.0)
        a3 = a7 = k * math.exp(-alpha) * (alpha + 1.0)
        a4 = a8 = -k * math.exp(-2.0 * alpha)
        b1 = math.pow(2.0, -alpha)
        b2 = -math.exp(-2.0 * alpha)
        c1 = c2 = 1

        for i in range(0, self.W):
            ym1 = 0.0
            ym2 = 0.0
            xm1 = 0.0
            for j in range(0, self.H):
                y1[self.H * i + j] = a1 * imgIn[self.H * i + j] + a2 * xm1 + b1 * ym1 + b2 * ym2
                xm1 = imgIn[self.H * i + j]
                ym2 = ym1
                ym1 = y1[self.H * i + j]

        for i in range(0, self.W):
            yp1 = 0.0
            yp2 = 0.0
            xp1 = 0.0
            xp2 = 0.0
            for j in range(self.H - 1, -1, -1):
                y2[self.H * i + j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
                xp2 = xp1
                xp1 = imgIn[self.H * i + j]
                yp2 = yp1
                yp1 = y2[self.H * i + j]

        for i in range(0, self.W):
            for j in range(0, self.H):
                imgOut[self.H * i + j] = c1 * (y1[self.H * i + j] + y2[self.H * i + j])

        for j in range(0, self.H):
            tm1 = 0.0
            ym1 = 0.0
            ym2 = 0.0
            for i in range(0, self.W):
                y1[self.H * i + j] = a5 * imgOut[self.H * i + j] + a6 * tm1 + b1 * ym1 + b2 * ym2
                tm1 = imgOut[self.H * i + j]
                ym2 = ym1
                ym1 = y1[self.H * i + j]

        for j in range(0, self.H):
            tp1 = 0.0
            tp2 = 0.0
            yp1 = 0.0
            yp2 = 0.0
            for i in range(self.W - 1, -1, -1):
                y2[self.H * i + j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
                tp2 = tp1
                tp1 = imgOut[self.H * i + j]
                yp2 = yp1
                yp1 = y2[self.H * i + j]

        for i in range(0, self.W):
            for j in range(0, self.H):
                imgOut[self.H * i + j] = c2 * (y1[self.H * i + j] + y2[self.H * i + j])
# scop end


class _StrategyNumPy(Deriche):

    def __new__(cls, options: PolyBenchOptions, parameters: PolyBenchSpec):
        return object.__new__(_StrategyNumPy)

    def __init__(self, options: PolyBenchOptions, parameters: PolyBenchSpec):
        super().__init__(options, parameters)

    def initialize_array(self, imgIn: ndarray, imgOut: ndarray):
        # input should be between 0 and 1 (grayscale image pixel)
        for i in range(0, self.W):
            for j in range(0, self.H):
                imgIn[i, j] = self.DATA_TYPE((313 * i + 991 * j) % 65536) / 65535.0

    def print_array_custom(self, imgOut: ndarray, name: str):
        for i in range(0, self.W):
            for j in range(0, self.H):
                if (i * self.H + j) % 20 == 0:
                    self.print_message('\n')
                self.print_value(imgOut[i, j])

    def kernel(self, alpha, imgIn: ndarray, imgOut: ndarray, y1: ndarray, y2: ndarray):
# scop begin
        k = (1.0 - math.exp(-alpha)) * (1.0 - math.exp(-alpha)) / (
            1.0 + 2.0 * alpha * math.exp(-alpha) - math.exp(2.0 * alpha))
        a1 = a5 = k
        a2 = a6 = k * math.exp(-alpha) * (alpha - 1.0)
        a3 = a7 = k * math.exp(-alpha) * (alpha + 1.0)
        a4 = a8 = -k * math.exp(-2.0 * alpha)
        b1 = math.pow(2.0, -alpha)
        b2 = -math.exp(-2.0 * alpha)
        c1 = c2 = 1

        for i in range(0, self.W):
            ym1 = 0.0
            ym2 = 0.0
            xm1 = 0.0
            for j in range(0, self.H):
                y1[i, j] = a1*imgIn[i, j] + a2*xm1 + b1*ym1 + b2*ym2
                xm1 = imgIn[i, j]
                ym2 = ym1
                ym1 = y1[i, j]

        for i in range(0, self.W):
            yp1 = 0.0
            yp2 = 0.0
            xp1 = 0.0
            xp2 = 0.0
            for j in range(self.H - 1, -1, -1):
                y2[i, j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
                xp2 = xp1
                xp1 = imgIn[i, j]
                yp2 = yp1
                yp1 = y2[i, j]

        for i in range(0, self.W):
            for j in range(0, self.H):
                imgOut[i, j] = c1 * (y1[i, j] + y2[i, j])

        for j in range(0, self.H):
            tm1 = 0.0
            ym1 = 0.0
            ym2 = 0.0
            for i in range(0, self.W):
                y1[i, j] = a5 * imgOut[i, j] + a6 * tm1 + b1 * ym1 + b2 * ym2
                tm1 = imgOut[i, j]
                ym2 = ym1
                ym1 = y1[i, j]

        for j in range(0, self.H):
            tp1 = 0.0
            tp2 = 0.0
            yp1 = 0.0
            yp2 = 0.0
            for i in range(self.W - 1, -1, -1):
                y2[i, j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
                tp2 = tp1
                tp1 = imgOut[i, j]
                yp2 = yp1
                yp1 = y2[i, j]

        for i in range(0, self.W):
            for j in range(0, self.H):
                imgOut[i, j] = c2 * (y1[i, j] + y2[i, j])
# scop end
