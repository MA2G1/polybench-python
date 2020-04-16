# Copyright 2020 Miguel Angel Abella Gonzalez <miguel.abella@udc.es>
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

"""This utility simplifies the task of creating a new benchmark.

Manually generating a benchmark implies creating a directory structure and
the required initialization files for the Python's package system to work.

This tool does all required steps for creating a package and a template file
for the new benchmark. Once created, this benchmark will be available when
running the launcher script (run-benchmark.py)"""

# Using argparse for parsing commandline options. See: https://docs.python.org/3.7/library/argparse.html
import argparse
# Using Path for creating folders representing categories
from pathlib import Path
import os.path
import shutil
from string import capwords

if __name__ == '__main__':

    # Define some "constant" references
    kernel_path = 'kernels'
    template_file = 'util/template-benchmark.py'
    template_init_file = 'util/template-package-init.py'

    # Define some "variable" references
    benchmark_name = None
    category_name = None

    # This function parses the command line arguments and returns the decoded parameters.
    # The returning order is defined as follows: [benchmark, category]
    #
    # The parameter "category" will be normalized to use slashes.
    def parse_command_line() -> [str, str]:
        parser = argparse.ArgumentParser(description='Generate a template for implementing a new benchmark.',
                                         epilog='All benchmarks are created under the "kernels" folder.')
        parser.add_argument('--name', '-N', nargs=1, type=str, required=True,
                            metavar='<benchmark_name>',
                            help='The name of the benchmark')
        parser.add_argument('--category', '-C', nargs=1, type=str,
                            metavar='<category[.subcategory] | category[/subcategory]>',
                            help='The category name. This will place the benchmark template into a package, generating '
                                 'all required intermediate files. Subcategories can be separated either with a dot or '
                                 'a slash. Example: datamining/correlation and datamining.correlation are valid.')

        # The parsing will fail on error.
        args = parser.parse_args()

        # Process the benchmark name
        benchmark = args.name[0]

        # Process the category name. Since the category is optional, we need to check if it exists first
        category = None
        if not (args.category is None):
            category = args.category[0].replace('.', '/')

        return [benchmark, category]

    # Create the full package structure given the category name. This operation creates all of the required files and
    # directories for building the package starting from the "kernel_path".
    # This function returns the full path for the package.
    def create_package_structure(category: str) -> str:
        target_path = None
        if not (category is None):
            target_path = kernel_path + '/' + category
            # Create, if necessary, all of the directories in the package path
            Path(target_path).mkdir(parents=True, exist_ok=True)

            # Check the existence of "__init__.py" files and, if necessary, create them from the template
            init_file_locations = target_path.split('/')[1:-1]  # exclude "kernels" and the final dir from this list
            init_file_path = kernel_path + '/'
            for check in init_file_locations:
                init_file_path += check + '/'
                init_file = init_file_path + '__init__.py'
                if not os.path.isfile(init_file):
                    shutil.copy2(template_init_file, init_file)

        return target_path

    # Create the required files for properly importing the benchmark from the package.
    # The files include a custom "__init__.py" with the imported benchmark and the "benchmark.py" file implementing, as
    # a template, the "Benchmark" class.
    def create_benchmark(directory: str, benchmark: str):
        # Check benchmark location for the existence of an "__init__.py" file
        init_file = directory + '/__init__.py'
        if not os.path.isfile(init_file):
            # The init file does not exist. Read the template's contents into memory, update them for this benchmark and
            # save the results into the new init file.
            with open(template_init_file) as f:
                contents = f.read()
                # Append: from path.to.package.benchmark import Benchmark
                contents += f'\nfrom {directory.replace("/", ".")}.{benchmark} import {capwords(benchmark)}\n'
                # Save the new init file
                with open(init_file, 'x') as g:
                    g.write(contents)

        target_file = directory + '/' + benchmark + '.py'
        # Check benchmark location for the existence of the benchmark. If a previous benchmark exists it will be kept,
        # otherwise a new benchmark template will be created with the correct benchmark's name.
        if not os.path.isfile(target_file):
            with open(template_file) as f:
                contents = f.read()
                with open(target_file, 'x') as g:
                    contents = contents.replace('TemplateClass', capwords(benchmark))
                    g.write(contents)


    # Parse the commandline and obtain the required parameters
    benchmark_name, category_name = parse_command_line()

    # Build the directory structure represented by the category
    package_path = create_package_structure(category_name)
    create_benchmark(package_path, benchmark_name)
