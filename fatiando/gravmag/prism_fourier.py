r"""
Calculate the F.D. expression of the potentials of a 3D right-rectangular prism

.. note:: All input units are SI.  Output is in conventional units: SI for the
    gravitational potential, mGal for gravity, Eotvos for gravity gradients, nT
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East, and z -> Down.

**Gravity**

The gravitational fields are calculated based on the transform of the integral
equation solution.  This is similar, but not identical, to the derivation by
Bhattacharyya and Navolio (1976).

Let :math:\tilde{\phi}(x,y,z) be the generic Fourier domain description of 
potential of a point source (a function that decays linearly with distance).
That is, 
:math:\phi(\mathbf{r})=\frac{1}{r}=\frac{1}{(x^2+y^2+(z-h)^2)^{1/2}}
with 
:math:\tilde{\phi}(\omega_x,\omega_y,z)=F[\phi(r)]
with :math:F[\cdot] denoting the Fourier Transform.
:math:\tilde{\phi}=\frac{\exp{-(h-z)\omega_r}{\omega_r}
where :math:\omega_r is the radial wavenumber.

The expressions for the gravity and magnetic fields, as well as their
gradients, can be derived by multiplication by the appropriate Fourier-Domain
operator.  Thus
:math:\tilde{g_z}=\omega_r\tilde{\phi}
:math:\tilde{g_x}=i\omega_x\tilde{\phi}
and so forth.

Generalization to a prism involves solving the volume integral over the prism.


Available functions are:

* :func:`~fatiando.gravmag.prism_fourier.potential`
* :func:`~fatiando.gravmag.prism_fourier.gx`
* :func:`~fatiando.gravmag.prism_fourier.gy`
* :func:`~fatiando.gravmag.prism_fourier.gz`
* :func:`~fatiando.gravmag.prism_fourier.gxx`
* :func:`~fatiando.gravmag.prism_fourier.gxy`
* :func:`~fatiando.gravmag.prism_fourier.gxz`
* :func:`~fatiando.gravmag.prism_fourier.gyy`
* :func:`~fatiando.gravmag.prism_fourier.gyz`
* :func:`~fatiando.gravmag.prism_fourier.gzz`

**Magnetic**
Available fields are the total-field anomaly and x,y,z, components of the 
magnetic induction:

* :func:`~fatiando.gravmag.prism_fourier.bx`
* :func:`~fatiando.gravmag.prism_fourier.by`
* :func:`~fatiando.gravmag.prism_fourier.bz`
* :func:`~fatiando.gravmag.prism_fourier.tf`

**References**
Bhattacharyya, B.K., and M.E. Navolio (1976), A Fast Fourier Transform Method
for Rapid Computation of Gravity and Magnetic Anomalies Due to Arbitrary
Bodies: Geophysical Prospecting, 24, 633-649.

----
"""

import numpy as np
import transform
from ..constants import G, SI2EOTVOS, CM T2NT SI2MGAL

def potential(xp, yp, zp, prisms, dens=None):
    """
    Calculates the gravitational potential in the Fourier Domain.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input and output values in **SI** units(!)!

    Fourier domain expression of potential for a right-rectangular prism
    with center at the xy origin, x extent of [-a,a], y [-b,b], and z [h1,h2],
    is given by
    4*G*rho*a*b*sinc(a*Wx/pi)*sinc(b*Wy/pi)*(exp(-(h1-z)Wr)-exp(-(h2-z)Wr))/
        (Wr^2)
    where Wr is the radial wavenumber.


    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~fatiando.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~fatiando.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp in the space domain
    * wx : 2D array of wavenumbers in the x direction
    * wy : 2D array of wavenumbers in the y direction
    * resf : 2D comple array of Fourier Domain expression of potential

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length.")
    size = len(xp)
    res = np.zeros(size, dtype=np.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density=dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2




    return 42

