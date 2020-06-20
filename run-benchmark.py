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

"""Polybench/Python is the reimplementation of the polyhedral benchmark Polybench/C in the Python programming language.

This module implements a main program which allows the user to run benchmarks easily without the burden of creating
makefiles or using configuration scripts for generating and using those.
This program allows to use only the Python runtime for everything the user should need when evaluating or implementing
the different kernels."""


# Import the basic elements for searching benchmark implementations
from benchmarks import benchmark_classes
from benchmarks.polybench import PolyBench, PolyBenchParameters
from benchmarks.polybench import DatasetSize
import benchmarks.polybench_options as polybench_options

# Using argparse for parsing commandline options. See: https://docs.python.org/3.7/library/argparse.html
import argparse
import os
from pathlib import Path

from filecmp import cmp  # used when verifying if two files have the same content
from sys import stderr, stdout


if __name__ == '__main__':

    def check_benchmark_availability() -> None:
        """
        Checks whether there are any benchmarks available or not.
        :return: None.
        :raise: NotImplementedError when there are no benchmarks available.
        """
        if len(benchmark_classes) < 1:
            raise NotImplementedError("There are no available benchmarks to run.")


    def parse_command_line() -> {
        'benchmark': str,
        'polybench_options': dict,
        'verify': bool,
    }:
        """
        Parse command line arguments and generate normalized results.

        :return: A dictionary with the decoded parameters.
        :rtype: dict[str, Any]
        """
        parser = argparse.ArgumentParser(description='Runs a given benchmark without setting up a shell environment at '
                                                     'all.')
        # Default parameter, mandatory
        parser.add_argument('benchmark', metavar='benchmark.py', nargs='?', default=None,
                            help='The path, relative to this script, to any file having a class implementing Polybench.'
                                 ' All implementations must reside somewhere inside the "benchmarks" folder.')
        # Optional parameters
        parser.add_argument('--polybench-options', dest='options', default=None,
                            help='A comma separated list of options passed to PolyBench. Available options can be found'
                                 ' in the README file. Usage: run.py --options '
                                 'POLYBENCH_PADDING_FACTOR=3,POLYBENCH_LINUX_FIFO_SCHEDULER')
        parser.add_argument('--dataset-size', dest='dataset_size', default=None,
                            help='Specify a working dataset size to use from "polybench.spec" file. Valid values are:'
                                 '"MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE".')
        parser.add_argument('--output', dest='output', default=None,
                            help='Alias for POLYBENCH_DUMP_TARGET. Also enables POLYBENCH_DUMP_ARRAYS. Prints the '
                                 'benchmark''s result into a file. In order to print into the console use either '
                                 '"stdout" or "stderr".')
        parser.add_argument('--verify-file', dest='verify_file_name', default=None,
                            help='Verify the results of the benchmark against the results stored in a file. This '
                                 'option enables --output and makes the output file to target a file name whose name '
                                 'matches the one passed by this argument, appending the .verify suffix. When the '
                                 'benchmark terminates, its result is compared and a message indicating the comparison '
                                 'result will be printed on "stdout".')
        parser.add_argument('--verify-polybench-path', dest='verify_polybench_path', default=None,
                            help='Combined with --verify-file, this parameter allows to specify the path where '
                                 'PolyBench/C is present to allow for automatic discovery of the appropriate output '
                                 'file. Please note that the files containing the results of PolyBench/C must already '
                                 'exist and must be next to where the actual implementation resides. This is the '
                                 'default behavior when PolyBench/C is run from Perl scripts.')
        # Parse the commandline arguments. This process will fail on error
        args = parser.parse_args()

        # Process the "benchmark" argument
        if args.benchmark is None:
            print_available_benchmarks()
            exit(-1)

        # Initialize the result dictionary
        result = {
            'benchmark': None,  # should hold a string
            'polybench_options': None,  # should hold a dict with options
            'verify': {
                'enabled': False,  # Controls whether to verify results or not
                'file': None,      # The file name to verify against
                'path': None,      # The path to PolyBench/C
                'full_path': None
            },
        }

        # Blindly replace the directory separator character with a commonly supported forward slash.
        # Reason: user may input the benchmark file by using tab-completion from a command line. On Windows,
        # tab-completion only works with backslashes while on Linux, MacOS and BSDs it works with forward slashes.
        # We are not taking into consideration other systems where the forward slash separator may not work.
        benchmark_py = args.benchmark.replace(os.sep, '/')
        # Remove the "py" extension and change slashes into dots.
        # What we are left with should be compatible with Class.__module__
        result['benchmark'] = benchmark_py.split('.')[0].replace('/', '.')

        print_result = False  # By default, do not print anything

        # Define some auxiliary functions.
        # Some parameters assign values with the same meanings. This prevents typing errors from happening.
        def set_output(file_name: str):
            if file_name == 'stderr':
                handle = stderr
            elif file_name == 'stdout':
                handle = stdout
            else:
                handle = open(file_name, 'w', newline='\n')
            result['output'] = handle

        # Process the "output" argument
        if not (args.output is None):
            set_output(args.output)
            print_result = True
        else:
            set_output('stderr')  # Just setting a default value

        # Process the "verify" arguments. May alter 'output'
        result['verify'] = {
            'enabled': False
        }
        if not (args.verify_file_name is None):
            result['verify']['enabled'] = True
            result['verify']['file'] = args.verify_file_name
            set_output(args.verify_file_name + '.verify')
            print_result = True

        if not (args.verify_polybench_path is None):
            result['verify']['path'] = args.verify_polybench_path.rstrip('/')

        # Check if the arguments passed to "verify*" are valid
        if result['verify']['enabled']:
            # Calculate the full file name (path + file)
            if result['verify']['path'] is None:
                result['verify']['full_path'] = result['verify']['file']
            else:
                # PolyBench/C path given. Search the appropriate category/benchmark
                # splitted_cat will contain, as a list, the category without "benchmarks/" nor "benchmark.py"
                split_cat = result['benchmark'].replace('benchmarks.', '').split('.')[:-1]
                category_name = ''
                for token in split_cat:
                    category_name += token + '/'
                result['verify']['full_path'] = result['verify']['path'] + '/' +\
                    category_name + result['verify']['file']

            file = Path(result['verify']['full_path'])
            if not file.is_file():
                # May need to switch underscores into hyphens!
                # The following replacement wont work with mixed underscores and hyphens on the same path
                file = Path(result['verify']['full_path'].replace('_', '-'))
                if not file.is_file():
                    raise RuntimeError(f'Validation file does not exist: "{result["verify"]["full_path"]}"')
                # Update the validated full_path
                result['verify']['full_path'] = result['verify']['full_path'].replace('_', '-')

        # Process PolyBench options
        # First import default options into polybench_options
        result['polybench_options'] = polybench_options.polybench_default_options.copy()
        if not (args.options is None):
            # Comma separated text -> split
            options = args.options.split(',')
            for option in options:
                if option in result['polybench_options']:  # simple "exists" validation
                    # Only boolean options can pass this validation
                    result['polybench_options'][option] = True
                else:  # may not exist if the text does not match
                    # Check if it is of the form OPT=val
                    opval = list(option.split('='))
                    if len(opval) == 2:
                        # Ok. Key-value found. Only a few of these exist... check manually
                        # ... for numerical conversions (currently all integers)
                        if opval[1].isnumeric():
                            result['polybench_options'][opval[0]] = int(opval[1])

        # Custom command line options can override output printing (the verify option). Update polybench_options
        if print_result:
            result['polybench_options'][polybench_options.POLYBENCH_DUMP_ARRAYS] = True
            result['polybench_options'][polybench_options.POLYBENCH_DUMP_TARGET] = result['output']

        # Append the dataset size if required
        if not (args.dataset_size is None):
            # Try to set the enumeration value from user input. On error, an exception is raised.
            if args.dataset_size not in [DatasetSize.MINI.name, DatasetSize.SMALL.name, DatasetSize.MEDIUM.name,
                                         DatasetSize.LARGE.name, DatasetSize.EXTRA_LARGE.name]:
                raise RuntimeError(f'Invalid value for parameter --dataset-size: "{args.dataset_size}"')
            result['polybench_options'][polybench_options.POLYBENCH_DATASET_SIZE] = DatasetSize[args.dataset_size]

        return result


    def print_available_benchmarks() -> None:
        """
        Prints on screen the available benchmarks (if any)
        :return: information on screen (commandline)
        """
        check_benchmark_availability()
        print('List of available benchmarks:')
        for impl in benchmark_classes:
            print(f'  {impl.__module__.replace(".", "/")}.py')


    def parse_spec_file() -> list:
        """Parses the polybench.spec file for obtaining all data for customizing a benchmark.

        :return: a list of dictionaries, each one holding all the information for a benchmark.
        :rtype: list[dict]
        """
        result = []
        with open('polybench.spec') as spec_file:
            # Since the file is small enough, processing line per line should not have any noticeable performance hit.
            spec_file.readline()  # skip header line
            for line in spec_file:
                dictionary = {}
                elements = line.split('\t')
                dictionary['kernel'] = elements[0]
                dictionary['category'] = elements[1]
                dictionary['datatype'] = elements[2]
                dictionary['params'] = elements[3].split(' ')
                not_numbers = elements[4].split(' ')
                numbers = []
                for nn in not_numbers:
                    numbers.append(int(nn))
                dictionary['MINI'] = numbers

                not_numbers = elements[5].split(' ')
                numbers = []
                for nn in not_numbers:
                    numbers.append(int(nn))
                dictionary['SMALL'] = numbers

                not_numbers = elements[6].split(' ')
                numbers = []
                for nn in not_numbers:
                    numbers.append(int(nn))
                dictionary['MEDIUM'] = numbers

                not_numbers = elements[7].split(' ')
                numbers = []
                for nn in not_numbers:
                    numbers.append(int(nn))
                dictionary['LARGE'] = numbers

                not_numbers = elements[8].split(' ')
                numbers = []
                for nn in not_numbers:
                    numbers.append(int(nn))
                dictionary['EXTRALARGE'] = numbers

                result.append(dictionary)

        return result


    def validate_benchmark_results(options: dict):
        """Compare two files and report if they match or not.

        This validation is intentionally very simple. In case of error, the user must manually compare the results"""
        output_file_name = options['output'].name
        verify_file_name = options['verify']['full_path']

        print(f'Verifying if files "{output_file_name}" and "{verify_file_name}" match... ', end='')
        if cmp(output_file_name, verify_file_name):
            print('OK')
        else:
            print('FAIL')
            print('Please, check contents manually.')


    def run(options: dict, parameters: PolyBenchParameters) -> None:
        # Set up parameters which may modify execution behavior
        module_name = options['benchmark']
        # Parameters used in case of verification
        verify_result = options['verify']

        # Search the module within available implementations
        instance = None
        for implementation in benchmark_classes:
            if implementation.__module__ == module_name:
                # Module found! Instantiate a new class with it
                instance = implementation(options['polybench_options'], parameters)  # creates a new instance

        # Check if the module was found
        if instance is None:
            module = module_name.replace(".", "/") + '.py'
            raise NotImplementedError(f'Module {module} not implemented.')

        # When the module was found, run it.
        if isinstance(instance, PolyBench):
            instance.run()

            # Verify benchmark's results against other implementation's results
            if verify_result:
                validate_benchmark_results(options)


    # Parse the command line arguments first. We need at least one mandatory parameter.
    opts = parse_command_line()
    # Parse the spec file for obtaining all of the benchmark's parameters
    spec_params = parse_spec_file()
    # Filter out the parameters and pick only the one for the current benchmark
    bench_params = {}
    for spec in spec_params:
        if spec['kernel'] in opts['benchmark']:
            bench_params = spec
            break
    # Create a parameters object for passing to the benchmark
    params = PolyBenchParameters(bench_params)
    # Run the benchmark (and other user options)
    run(opts, params)
