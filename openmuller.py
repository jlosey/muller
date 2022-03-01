"""Propagating 2D dynamics on the muller potential using OpenMM.

Currently, we just put a harmonic restraint on the z coordinate,
since OpenMM needs to work in 3D. This isn't really a big deal, except
that it affects the meaning of the temperature and kinetic energy. So
take the meaning of those numbers with a grain of salt.
"""
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, dalton
import openmm as mm

import matplotlib.pyplot as pp
from matplotlib.animation import FuncAnimation
#from matplotlib import cmap
import numpy as np

class MullerForce(mm.CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j], AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            expression += '''+ {AA}*exp({aa} *(x - {XX})^2 + {bb} * (x - {XX}) 
                               * (y - {YY}) + {cc} * (y - {YY})^2)'''.format(**fmt)
        super(MullerForce, self).__init__(expression)
    
    @classmethod
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for j in range(4):
            value += cls.AA[j] * np.exp(cls.aa[j] * (x - cls.XX[j])**2 + \
                cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) + cls.cc[j] * (y - cls.YY[j])**2)
        return value

    @classmethod
    def plot(cls, ax=None, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = cls.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = pp
        ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)


##############################################################################
# Global parameters
##############################################################################
# random seed
seed = 123456
_r = np.random.seed(seed)
# each particle is totally independent, propagating under the same potential
nParticles = 100  
mass = 1.0 * dalton
temperature = 300 * kelvin
friction = 100 / picosecond
timestep = 10.0 * femtosecond

# Choose starting conformations uniform on the grid between (-1.5, -0.2) and (1.2, 2)
posi = (np.random.rand(nParticles, 3) * np.array([2.7, 1.8, 1])) + np.array([-1.5, -0.2, 0])
###############################################################################


system = mm.System()
mullerforce = MullerForce()
for i in range(nParticles):
    system.addParticle(mass)
    mullerforce.addParticle(i, [])
system.addForce(mullerforce)

integrator = mm.LangevinIntegrator(temperature, friction, timestep)
context = mm.Context(system, integrator)

context.setPositions(posi)
context.setVelocitiesToTemperature(temperature)
#xi = np.linspace(-1.5,1.2,300)
#y = np.linspace(-0.2,2.,300)
#xx,yy = np.meshgrid(xi,y)
minx=-1.5; maxx=1.2; miny=-0.2; maxy=2.0
grid_width = max(maxx-minx, maxy-miny) / 200.0
xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
v = MullerForce.potential(xx,yy)
#v =np.ma.masked_array(v,v>400)

### Binning ###
nbins = 4 
x_l = maxx-minx
y_l = maxy-miny
x_width = x_l / nbins
y_width = y_l / nbins
x_bins = np.arange(minx+x_width,maxx,x_width)
y_bins = np.arange(miny+y_width,maxy,y_width)

x_dig = np.digitize(posi[:,0],x_bins)
y_dig = np.digitize(posi[:,1],y_bins)
x_dig_i = x_dig
y_dig_i = y_dig
### transition matrix ###
T = np.zeros((nbins,nbins,nbins,nbins),dtype=int)

### Loop over number of steps
steps = 1e4
for i in range(int(steps)):
    integrator.step(1)
    posi = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
    x_dig = np.digitize(posi[:,0],x_bins)
    y_dig = np.digitize(posi[:,1],y_bins)
    np.add.at(T,(x_dig_i,x_dig,y_dig_i,y_dig),1)
    x_dig_i = x_dig
    y_dig_i = y_dig

np.savetxt(f"data/transitions/transition_{seed}.txt"
        ,T.reshape(nbins*nbins,nbins*nbins)
        ,fmt="%d"
        ,header=f"Transitions for 2D Muller potential\n" +
            f"{nbins} bins\n" + 
            f"{steps} steps\n" + 
            f"{nParticles} Particles\n" +
            f"T = {temperature}\n" +
            f"friction = {friction}"
        )
np.save(f"data/transitions/transitions_{seed}",T)
