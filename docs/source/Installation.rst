Installation Guide
==================

Prerequisites
-------------

Before proceeding with the installation of AcousticZ, ensure that your system 
meets the following prerequisites:

- Python 3.11 or higher installed on your machine.
- Pip package manager (usually bundled with Python installations).
- A compatible operating system (Windows, macOS, or Linux).

Further Disclaimers
-------------------

For the Ray calculations are mesh based file formats beneficial and it needs 
normal vectors and material information. Therefore the code only supports one 
file format which is .obj (OBJ). Please assure that your custom Room geometry 
is provided in that format.

Installing AcousticZ
--------------------

Option 1: Installing from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install AcousticZ from the Python Package Index (PyPI), follow these steps:

Step 1: Open a command-line interface (CLI) or terminal.

Step 2: Execute the following command to install AcousticZ and its dependencies::

    pip install acousticz

Option 2: Installing from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can install AcousticZ directly from the source code. Here's 
how:

Step 1: Download the source code package from the official AcousticZ repository on GitHub.

Step 2: Extract the downloaded package to a directory of your choice.

Step 3: In a CLI or terminal, navigate to the extracted directory containing the source code.

Step 4: Run the following command to install AcousticZ and its dependencies::

    pip install .

Verifying the Installation
--------------------------

To ensure a successful installation, you can perform a quick verification. Open a Python interactive shell or a Python script and import the AcousticZ module::

    import acousticz

If the import statement executes without errors, congratulations! AcousticZ is successfully installed on your system.

Getting Started
---------------

To get started with AcousticZ, refer to the documentation and examples provided. Familiarize yourself with the available functions, classes, and methods to make the most of the package's capabilities.

Updating AcousticZ
------------------

To update AcousticZ to the latest version, use the following command::

    pip install --upgrade acousticz

By executing this command, you will fetch and install the latest release from PyPI.

Note: It is recommended to periodically check for updates and keep AcousticZ up to date with the latest enhancements, bug fixes, and new features.

Congratulations! You have successfully installed AcousticZ on your system. Now you're ready to embark on an acoustic journey, leveraging the power of ray tracing and simulation to unlock the secrets of sound within virtual environments.
