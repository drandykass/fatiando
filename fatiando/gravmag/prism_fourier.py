r"""
Calculate the Fourier Domain expression of the potentials of a 3D right-rectangular prism.

.. note:: All input units are SI.  Output is in conventional units: SI for the 
    gravitational potential, mGal for gravity, Eotvos for gravity gradients, nT 
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East, and z -> Down.

**Background**

The potential fields are calculated based on the transform of the integral
equation solution.  This is similar, but not identical, to the derivation by
Bhattacharyya and Navolio (1976).

Let :math:`\tilde{\phi}(x,y,z)` be the generic Fourier domain description of 
potential of a point source (a function that decays linearly with distance).
That is, 

.. math:: \phi(\mathbf{r})=\frac{1}{r}=\frac{1}{(x^2+y^2+(z-h)^2)^{1/2}}

with 

.. math:: \tilde{\phi}(\omega_x,\omega_y,z)=F[\phi(r)]

and :math:`F[\cdot]` denoting the Fourier Transform.

.. math:: \tilde{\phi}=\frac{e^{-(h-z)\omega_r}}{\omega_r}

where :math:`\omega_r` is the radial wavenumber.


Generalisation to a prism involves solving the volume integral over the prism.  For a prism centred horizontally at the origin with :math:`\nu\in[-a,a]`, :math:`\eta\in[-b,b]`, and :math:`\zeta\in[h_1,h_2]`, where :math:`[\nu,\eta,\zeta]` are the Cartesian coordinates of an element inside the volume of a 3D prism, the potential is given by

.. math::

    \tilde{\phi}(\omega_x,\omega_y,z)&=\displaystyle\int_{h_1}^{h_2}{\int_{-b}^{b}{\int_{-a}^{a}{\frac{e^{-(z-\zeta)\omega_r}}{\omega_r}e^{-i\omega_x\nu}e^{-i\omega_y\eta}~d\nu~d\eta~d\zeta}}}\\
   &=4ab\text{sinc}\left(\frac{a\omega_x}{\pi}\right)\text{sinc}\left(\frac{b\omega_y}{\pi}\right)\frac{e^{-(h_1-z)\omega_r}-e^{-(h_2-z)\omega_r}}{\omega_r^2}

To generalise for a prism in an arbitrary location, let :math:`2a` be the width of the prism in the :math:`x` direction and :math:`2b` be the width in the :math:`y` direction.  Let the horizontal coordinate of the centre of the prism be given by :math:`[\nu_0,\eta_0]`.  The potential then becomes

.. math::
   =4ab\text{sinc}\left(\frac{a\omega_x}{\pi}\right)\text{sinc}\left(\frac{b\omega_y}{\pi}\right)\frac{e^{-(h_1-z)\omega_r}-e^{-(h_2-z)\omega_r}}{\omega_r^2}e^{-i\omega_x\nu_0}e^{-i\omega_y\eta_0}.

Available functions are:

* :func:`~fatiando.gravmag.prism_fourier.general_potential`

**Gravity**

The expressions for the gravity and magnetic fields, as well as their
gradients, can be derived by multiplication by the appropriate Fourier-Domain
operator.  Thus

.. math::
    
    \tilde{g_z}&=\gamma\rho\omega_r\tilde{\phi}\\
    \tilde{g_x}&=\gamma\rho i\omega_x\tilde{\phi}

and so forth.

Available functions are:

* :func:`~fatiando.gravmag.prism_fourier.gravity_potential`
* :func:`~fatiando.gravmag.prism_fourier.gx`
* :func:`~fatiando.gravmag.prism_fourier.gy`
* :func:`~fatiando.gravmag.prism_fourier.gz`
* :func:`~fatiando.gravmag.prism_fourier.gxx`
* :func:`~fatiando.gravmag.prism_fourier.gxy`
* :func:`~fatiando.gravmag.prism_fourier.gxz`
* :func:`~fatiando.gravmag.prism_fourier.gyy`
* :func:`~fatiando.gravmag.prism_fourier.gyz`
* :func:`~fatiando.gravmag.prism_fourier.gzz`

**Magnetics**

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
from ..constants import G, SI2EOTVOS, CM, T2NT, SI2MGAL

def gravity_potential(xp, yp, zp, prisms, dens=None):
    """
    Calculates the gravitational potential in the Fourier Domain.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input and output values in **SI** units(!)!

    Computes the gravitational potential of a right rectangular 3D prism
    in the Fourier Domain.

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

def general_potential(xp,yp,zp,x1,x2,y1,y2,z1,z2):
    """
    Calculates the generalised potential in the fourier domain

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y-> East, and z -> Down

    This function calculates the potential response of a right rectangular
    prism in the Fourier domain.  This is the generalised potential; to 
    convert to gravitational potential, for example, multiply by the 
    density and the gravitational constant.  This function is the base function
    for computing all the other gravitational and magnetic responses.

    Parameters:

    * x1,x2,y1,y2,z1,z2 : floats
        Floats with the edge coordinates of the prism to compute
    * xp,yp,zp : arrays
        Arrays with the x, y, and z coordinates of the computation points

    Returns:

    * res_space : array
        Resulting array of potentials in the space domain
    * res_fourier : complex array
        Resulting array of potentials in the Fourier domain

    """


    return 42

