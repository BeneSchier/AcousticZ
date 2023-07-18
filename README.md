# AcousticZ

AcousticZ is a powerful Python package that allows you to simulate and analyze the real-world acoustic characteristics of rooms. By utilizing stochastic ray tracing, it generates a comprehensive Histogram that captures energy values correlated with frequency and time. This Histogram serves as the foundation for generating Room Impulse Responses (RIRs) for any arbitrary 3D room geometry.

## Key Features

- Simulate room impulse responses with ray tracing: AcousticZ employs advanced ray tracing techniques to generate realistic RIRs for virtual rooms.
- Comprehensive Histogram generation: The package produces a Histogram that stores energy values based on frequency and time, allowing for detailed acoustic analysis.
- Auralization and reverbant effects: Use the RIRs to apply realistic room effects, such as auralization and reverb, to audio files.

## Installation

To install AcousticZ, ensure that you have Python 3.11 or higher installed. Use pip, the Python package manager, to install AcousticZ from the Python Package Index (PyPI):

```console
pip install git+https://github.com/BeneSchier/AcousticZ
```


For detailed installation instructions and alternative installation methods, refer to the [Installation Guide](./docs/build/html/Installation.html) in the documentation.

## Usage
```python
from AcousticZ.Room import Room
```
Now, you sucessfully imported the package and you can start working.
### Here is a simple example

If you want to simulate the RIR of a simple shoebox you can do this by first reading mesh file and construct a room geometry:

```python
shoebox_file = '<your_meshfile.obj>'
shoebox = Room(shoebox_file)
```
Alternatively you can use the provided example mesh for a shoebox:
```python
import pkg_resources
shoebox_file =  pkg_resources.resource_filename('AcousticZ', '../data/example_meshes/shoebox.obj')
shoebox = Room(shoebox_file)
```

Then add a source and a receiver by:
```python
import numpy as np
# Just example values, ensure that the coordinates are located inside the room
source = np.array([2.0, 2.0, 2.0]) 
receiver = np.array([5.0, 5.0, 1.8])

# Adding source 
shoebox.createSource(source)

# Addding receiver
shoebox.createReceiver(receiver, radiusOfReceiverSphere=0.0875)
```

Perform the Ray Tracing algorithm with 100 rays:
```python
# Perform Ray Tracing with 100 rays
shoebox.performRayTracing(100)
```

And genrate and display the RIR by:
```python
# Generate the Room Impulse Response
shoebox.generateRIR()
shoebox.plotWaveform()
```

The shoebox mesh is included in the package and therefore can easily be read by 


## Getting Started

Once you have installed AcousticZ, check out the [Getting Started](./docs/build/html/GettingStarted.html) guide to begin using the package. It provides an overview of the basic usage, explains the required prerequisites, and offers example code to simulate room impulse responses and perform auralization.

## Documentation

For comprehensive information on the AcousticZ package, including the API reference, advanced features, and tutorials, refer to the full [Documentation](./docs/build/html/index.html). It covers various topics such as room geometry, sound source positioning, and more.
