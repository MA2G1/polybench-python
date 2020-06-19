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
from benchmarks.polybench import PolyBench
import benchmarks.polybench_options as polybench_options

# Using argparse for parsing commandline options. See: https://docs.python.org/3.7/library/argparse.html
import argparse
import logging
import os

from filecmp import cmp  # used when verifying if two files have the same content
from sys import stderr, stdout


if __name__ == '__main__':

    main_logger = logging.getLogger(__name__)
    if not (main_logger is None):  # I just want to initialize the logger in a different scope
        log_handler_console = logging.StreamHandler()
        log_formatter_console = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

        log_handler_console.setLevel(logging.DEBUG)
        log_handler_console.setFormatter(log_formatter_console)

        main_logger.addHandler(log_handler_console)

    def check_benchmark_availability() -> None:
        """
        Checks whether there are any benchmarks available or not.
        :return: None.
        :raise: NotImplementedError when there are no benchmarks available.
        """
        if len(benchmark_classes) < 1:
            raise NotImplementedError("There are no available benchmarks to run.")


    def print_available_benchmarks() -> None:
        """
        Prints on screen the available benchmarks (if any)
        :return: information on screen (commandline)
        """
        check_benchmark_availability()
        print('List of available benchmarks:')
        for impl in benchmark_classes:
            print(f'  {impl.__module__.replace(".", "/")}.py')


    def parse_command_line() -> {
        'benchmark': str,
        'output': stderr,
        'verify': bool,
        'verify_file': str,
        'polybench_options': dict
    }:
        """
        Parse command line arguments and generate normalized results.

        :return: A dictionary with the decoded parameters.
        :rtype: dict[str, Any]

        The possible return values for the dictionary are listed below:
            - **benchmark**: the module name implementing the user selected benchmark.
            - **print**: tells whether the user requested printing the output or not.
            - **output**: when the print value is set, this value specifies where the output must be written.
            - **verify**: tells whether the user requested verifying this run's output against a pre-recorded one.
            - **verify_file**: when verify is set, this value specifies the name of the original file
        """
        parser = argparse.ArgumentParser(description='Runs a given benchmark without setting up a shell environment at '
                                                     'all.')
        parser.add_argument('benchmark', metavar='benchmark.py', nargs='?', default=None,
                            help='The path, relative to this script, to any file having a class implementing Polybench.'
                                 ' All implementations must reside somewhere inside the "benchmarks" folder.')
        parser.add_argument('--output', dest='output', default=None,
                            help='Prints the benchmark''s result into a file. There are two special file names in '
                                 'order to print the output to console: "stdout" and "stderr".')
        parser.add_argument('-v', dest='verbose', action='count', default=0,
                            help='Prints more information on screen. The amount of information is determined by the '
                                 'number of "v" character appearance and is directly related to logging levels. The '
                                 'default behavior (without this flag set) is to just log error messages. Setting this '
                                 'flag once rises the logging to WARNING level, "-vv" sets INFO level and "-vvv" sets '
                                 'DEBUG level. Please note that this option does not affect the behavior of other '
                                 'messages which may be printed when setting other options (for instance, printing the '
                                 'algorithm results).')
        parser.add_argument('--verify', dest='verify', metavar='file', default=None,
                            help='Verify the output generated by the benchmark. A file name is expected. This file '
                                 'name corresponds to the output generated by the same benchmark in Polybench/C. This '
                                 'file name will also be used for generating the output of Polybench/Python, appending '
                                 'the ".verify" extension. When the new file is generated it is compared against the '
                                 'existing one. The comparison result is either "OK" or "FAIL". Both files are '
                                 'expected to have Unix line endings. Since this option generates an output file, it '
                                 'also overrides whatever you may specify in --output')
        parser.add_argument('--options', dest='options', default=None,
                            help='A comma separated list of options passed to PolyBench. Available options can be found'
                                 ' in the README file. Usage: run.py --options '
                                 'POLYBENCH_PADDING_FACTOR=3,POLYBENCH_LINUX_FIFO_SCHEDULER')
        # Parse the commandline arguments. This process will fail on error
        args = parser.parse_args()

        # Process verbosity first, as this affects logging
        if args.verbose == 0:    # show errors (or worse) only
            main_logger.setLevel(logging.ERROR)
        elif args.verbose == 1:    # show warnings
            main_logger.setLevel(logging.WARNING)
        elif args.verbose == 2:  # show info
            main_logger.setLevel(logging.INFO)
        else:  # show all logs
            main_logger.setLevel(logging.DEBUG)

        # Process the "benchmark" argument
        if args.benchmark is None:
            print_available_benchmarks()
            exit(-1)

        result = {}

        # Blindly replace the directory separator character with a commonly supported forward slash.
        # Reason: user may input the benchmark file by using tab-completion from a command line. On Windows,
        # tab-completion only works with backslashes while on Linux, MacOS and BSDs it works with forward slashes.
        # We are not taking into consideration other systems where the forward slash separator may not work.
        benchmark_py = args.benchmark.replace(os.sep, '/')
        # Remove the "py" extension and change slashes into dots.
        # What we are left with should be compatible with Class.__module__
        result['benchmark'] = benchmark_py.split('.')[0].replace('/', '.')

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

        print_result = False  # By default, do not print anything

        # Process the "output" argument
        if not (args.output is None):
            print_result = True
            set_output(args.output)
        else:
            set_output('stderr')

        # Process the "verify" argument. May alter 'output'
        result['verify'] = False
        if not (args.verify is None):
            print_result = True
            set_output(args.verify + '.verify')  # file_name + '.verify' -> file.ext.verify
            result['verify_file'] = args.verify
            result['verify'] = True
        else:
            result['verify_file'] = None
            result['verify'] = False

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
                    # Check if it is of type OPT=val
                    opval = option.split('=')
                    if len(opval) == 2:
                        # Ok. Key-value found. Only a few of these exist... check manually
                        # ... for numerical conversions (currently all integers)
                        if opval[1].isnumeric():
                            result['polybench_options'][opval[0]] = int(opval[1])

        # Custom command line options can override output printing (the verify option). Update polybench_options
        if print_result:
            result['polybench_options'][polybench_options.POLYBENCH_DUMP_ARRAYS] = True
            result['polybench_options'][polybench_options.POLYBENCH_DUMP_TARGET] = result['output']

        return result


    def run(params: dict) -> None:
        # Set up parameters which may modify execution behavior
        module_name = params['benchmark']
        # Parameters used in case of verification
        verify_result = params['verify']

        # Search the module within available implementations
        instance = None
        for implementation in benchmark_classes:
            if implementation.__module__ == module_name:
                # Module found! Instantiate a new class with it
                if main_logger.isEnabledFor(logging.INFO):
                    main_logger.info('Module "%s" found. Implementor class is "%s"',
                                     implementation.__module__, implementation.__name__)
                instance = implementation(params['polybench_options'])  # creates a new instance

        # Check if the module was found
        if instance is None:
            module = module_name.replace(".", "/") + '.py'
            raise NotImplementedError(f'Module {module} not implemented.')

        # When the module was found, run it.
        if isinstance(instance, PolyBench):
            if main_logger.isEnabledFor(logging.INFO):
                main_logger.info(f'Running "{instance.__class__.__name__}"')
            instance.run()

            # Verify benchmark's results against other implementation's results
            if verify_result:
                output_file = params['output']
                verify_file = params['verify_file']
                output_file.flush()  # Need to manually flush the file or the compare may fail
                print(f'Verifying if files "{verify_file}" and "{output_file.name}" match... ', end='')
                if cmp(verify_file, output_file.name):
                    print('OK')
                else:
                    print('FAIL')


    # Parse the command line arguments first. We need at least one mandatory parameter.
    parameters = parse_command_line()
    run(parameters)
