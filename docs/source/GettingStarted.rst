Getting Started
===============

Welcome to AcousticZ! This section will guide you through the initial steps to 
start using the package effectively.

Prerequisites
-------------

Before getting started with AcousticZ, ensure that you have completed the 
installation process as outlined in the 
:doc:`Installation Guide <Installation>`. Make sure you have the required 
dependencies installed and your environment is set up correctly.

Basic Usage
-----------

AcousticZ provides a range of functions, classes, and methods to simulate room 
impulse responses and analyze acoustic characteristics. Here's a basic example 
to get you started:

1. Import the AcousticZ module in your Python script or interactive shell:

   .. code-block:: python
      
      from AcousticZ.Room import Room 

2. Read the .obj geometry file and create an instance of Room:

    .. code-block:: python
        
        
        room_file = '<your_meshfile.obj>'
        custom_room = Room(room_file)

Alternatively you can use the provided example mesh for a shoebox:
    .. code-block:: python
        
        
        import pkg_resources
        room_file =  pkg_resources.resource_filename('AcousticZ', '../data/example_meshes/shoebox.obj')
        custom_room = Room(room_file)

3. Define the parameters for your acoustic simulation, such as room geometry, 
sound source position, and receiver positions.

    .. code-block:: python
        
        import numpy as np
        source = np.array([2.0, 2.0, 2.0]) 
        receiver = np.array([5.0, 5.0, 1.8])

        custom_room.createSource(source)
        custom_room.createReceiver(receiver, radiusOfReceiverSphere=0.0875)
        

4. Use the provided functions and methods to simulate the room impulse response 
and obtain the desired acoustic properties. For example, you can generate a 
histogram of energy values over frequency and time:

    .. code-block:: python

        # Perform Ray Tracing with 1000 rays
       custom_room.performRayTracing(1000)

       custom_room.plotEnergyHistogram()




This Histogram can then be used to generate a room impulse response (RIR) for 
further processing and analysis

4. Explore the various features and functionalities of AcousticZ to enhance your 
acoustic simulations. Refer to the :doc:`API documentation <APIReference>` for 
detailed information on available functions, classes, and methods.

Example Code
------------

To further illustrate the capabilities of AcousticZ, here's an example that 
demonstrates how to simulate a room impulse response and perform auralization:

Hier Code 


Next steps
----------
Now that you have a basic understanding of AcousticZ and its usage, you can dive 
deeper into the advanced features and functionalities. Refer to the 
:doc:`API documentation <APIReference>` for comprehensive information on 
available methods and their usage. Additionally, explore the provided examples 
and tutorials to gain further insights into using AcousticZ effectively.

Congratulations! You are now ready to embark on an acoustic journey with 
AcousticZ, leveraging its powerful capabilities to simulate and analyze 
real-world acoustic environments.

