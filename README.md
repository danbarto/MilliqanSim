# MilliqanSim

## What is it?

A general-purpose particle trajectory simulator that can propagate a particle through a magnetic field, implement
basic multiple scattering and Bethe-Bloch energy loss, and generate statistics on the intersections of these trajectories
with an exernal plane (e.g., the Milliqan detector).

<p align="center">
<img src="./bfield/bfield_with_trajs.png" alt="cross section of CMS b field" width="600"/>
</p>

*Note on units and coordinates: throughout the software, units used are meters for distance, MeV for mass/momentum/energy, nanoseconds
for time, and electron-charge units for charge. Coordinates used are CMS coordinates, with the beamline being the z direction.*

## Dependencies and setup

The package is entirely written in Python (v 2.7). The only absolute requirement is the [numpy](http://www.numpy.org) 
library. Optional but useful are the [matplotlib](http://matplotlib.org/) library to make plots, and pyROOT to get input 
from or output to a ROOT file.

To setup, run `. setup.sh` from the main directory. This will properly set your PYTHONPATH environment variable
and check for the required packages.

To run a simple check that everything is working, you can go to the `test` directory and run `python validate_helix.py`.
This will propagate a particle through a uniform magnetic field and compare the trajectory with the expected helix.
You should see a plot of the x and y coorinates as a function of time.

## Repository contents

The `src` directory contains 5 main modules, which are imported into a user-written main script.

1. **Params.py** - contains all of the configurable parameters, material definitions, etc.
2. **Integrator.py** -  routines to propagate the particle through a magnetic field
3. **Detector.py** - Functions related to detector and environment setup - magnetic field, material, etc.
4. **MultipleScatter.py** - methods implementing multiple scattering and Bethe-Bloch energy loss
5. **Drawing.py** - various methods to make plots of trajectories

The `test` directory contains a few well-commented example scripts that should give a good idea of how to use the program. In particular, 
`milliqan_test.py` is a program to simulate particle hits on a pseudo Milliqan detector.

The `bfield` directory contains the pickled-version of the CMS magnetic field map. This is loaded into memory at the start
of the simulation. There is also a script to make a 2D plot of the field.

The `p_eta_dist` directory contains a sample root file with pT and eta distributions of hypothetical milli-charged particles.
This is used by `test/milliqan_test.py` to simulate the trajectories of these particles.

## General instructions

For out-of-the-box use, you shouldn't have to edit any of the files in the `src` directory. Everything can be controlled from
the user-written main script. A few examples of these are in the `test` directory. You should start each script by loading
the CMS magnetic field (if using), and then setting the desired parameters from `Params.py`. The configurable parameters include

1. Params.Q - charge of the particle, in electron charge units
2. Params.m - mass of the particle (*Note: Q and m can be changed at any time from within the main script. Whatever their values are at the time of the
Integrator.rk4 call will be used*)
3. Params.MSCtype - type of multiple scattering algorithm. Can be "PDG" or "KUHN". Probably should just use "PDG". (use "none" for no multiple scattering).
4. Params.EnergyLossOn - boolean for whether or not to use Bethe-Bloch energy loss
5. Params.BFieldType - use "CMS" for the CMS B-field map. "Uniform" gives a uniform 1 T field, "none" turns magnetic field off.
6. Params.MatSetup - material setup to use. "CMS" uses a very rough cylindrical model of CMS (air until r=1.3, PbWO4 until r=1.8, Iron until r=7.0, and air outside.
Solid concrete for x>16.5). "iron" is solid iron for testing purposes. 'air' is just air at normal pressure. To simulate a vacuum, set Params.EnergyLossOn=False and
Params.MSCtype='none'. For custom materials or material setups, you will have to edit the appropriate sections in the Params and Detector modules.

Once the parameters are set, you can call the `Integrator.rk4` method to compute a particle trajectory. This is a general-purpose Runge-Kutta integrator
defined as follows:

`Integrator.rk4(x0, dt, nsteps, cutoff=None, cutoffaxis=None, update_func=traverseBField)`

- `x0` is a 6-element vector of initial conditions. The first 3 entries are position, and the next 3 are momentum
- `dt` - the timestep, in ns. 0.2 is generally sufficient.
- `nsteps` - the maximum number of steps to simulate
- 'cutoff' and 'cutoffaxis' control an optional distance cutoff. If they are not specified, then the full `nsteps` are simulated.
Otherwise, the trajectory stops after the particle reaches a distance `cutoff` along `cutoffaxis` (0=x axis, 1=y, 2=z, 3=radial)
- 'update_func` specifies the function to compute dx/dt and dp/dt. Integrator.traverseBField is the default, probably don't need to touch.

The return value of `Integrator.rk4` is a 6-by-(nsteps+1) array. Each column contains (x,y,z,px,py,pz) at a specific timestep.

The `Detector.FindIntersection` method takes the trajectory array defined above and computes statistics on the intersection with
and external plane. See `src/Detector.py` and `test/milliqan_test.py` for instructions on how to use and descriptions of the return variables.
