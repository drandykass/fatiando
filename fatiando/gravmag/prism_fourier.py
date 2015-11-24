r"""
Calculate the Fourier Domain expression of the potentials of a 3D right-rectangular prism.

.. note:: All input units are SI.  Output is in conventional units: SI for the 
    gravitational potential, mGal for gravity, Eotvos for gravity gradients, nT 
    for magnetic total field anomalies.

.. note:: The coordinate system of the input parameters is x -> North,
    y -> East, and z -> Down.

.. note:: Because the modelling is done is the Fourier domain, it is 
    assumed that the data observation locations are regularly gridded
    and the number of observations in each location is a power of 2.

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
# For testing only:
import matplotlib.pyplot as plt

def itxfm(dx,dy,a,padx=0,pady=0):
    """
    Calculates the inverse transform of the modelled data

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y-> East, and z -> Down.

    Computes the ifft of the modelled data.  Note that because this is a one-
    way transform, the data need to be normalised by (1/dx * 1/dy).  The
    result needs to be multiplied by 2pi.

    Parameters:

    * dx, dy : float
        Scalars describing the discretization in the spatial domain
    * a : Complex 2D array
        The complex array holding the modelled data in the F.D.

    Returns:

    * b : 2D array
        Real-valued spatial domain data

    """
    #b = (2*np.pi)*_unfold(np.real(np.fft.ifft2(_fold(a)*(1/dx)*(1/dy))))
    #nx = len(b[:,0])-padx*2
    #ny = len(b[0,:])-pady*2
    #c = b[padx:padx+nx,pady:pady+ny]
    #c = (2*np.pi)*np.fft.fftshift(np.real(np.fft.ifft2(
    #    np.fft.fftshift(a)*(1/(dx*dy)))))
    c = (2.*np.pi)*np.fft.fftshift(np.real(np.fft.ifft2(
        np.fft.ifftshift((a)*(1./(dx*dy))))))

    return np.ravel(c)


def gz(xp, yp, zp, sh, prisms, dens=None):
    """
    Calculates the vertical gravity field in the Fourier Domain.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units. Outputs are SI and mGal.

    Computes the vertical component of the gravity field due to a right
    rectangular 3D prism in the Fourier Domain.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * sh : tuple = (nx,ny)
        The shape of the grid
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

    * res_fourier : array
        The field calculated on xp, yp, zp in the Fourier domain
    * kx,ky : 2D array
        2D array of wavenumbers
    * X,Y : 2D array
        2D array of spatial coordinates
    * padx, pady : Scalar
        Scalars containing any padding cells added in the x and y direction

    """

    res_fourier,kx,ky,X,Y,padx,pady = gravity_potential(xp,yp,zp,sh,
        prisms, dens=None)

    kr = np.sqrt(np.square(kx)+np.square(ky)) * SI2MGAL
    #for ii in range(0,len(kr[:,0])):
    #    for jj in range(0,len(kr[0,:])):
    #        if kr[ii,jj] == 0:
    #            kr[ii,jj] = SI2MGAL
    kr[(len(res_fourier[:,0])/2),(len(res_fourier[0,:])/2)] = SI2MGAL              
    res_fourier *= kr

    # For testing only
    #plt.figure()
    #plt.pcolormesh(ky,kx,np.log10(np.abs(res_fourier)))
    #plt.colorbar()
    #plt.show()

    return res_fourier,kx,ky,X,Y,padx,pady


def gravity_potential(xp, yp, zp, sh, prisms, dens=None):
    """
    Calculates the gravitational potential in the Fourier Domain.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input and output values in **SI** units(!)!

    Computes the gravitational potential of a right rectangular 3D prism
    in the Fourier Domain.

    Gravity potential splits a mesh into multiple prisms if necessary and
    sums the result.

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * sh : tuple = (nx,ny)
        The shape of the grid
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

    * res_space : array
        The field calculated on xp, yp, zp in the space domain
    * wx : 2D array of wavenumbers in the x direction
    * wy : 2D array of wavenumbers in the y direction
    * resf : 2D comple array of Fourier Domain expression of potential

    """

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
        t2,kx,ky,X,Y,padx,pady = general_potential(xp,yp,zp,sh,x1,x2,
            y1,y2,z1,z2)
        try:
            res_fourier += (t2 * density * G)
        except UnboundLocalError:
            res_fourier = np.zeros((sh[0]+padx,sh[1]+pady), dtype=np.complex)
            res_fourier += (t2 * density * G)
        
        print prism

    # For testing only
    #R = np.log10(np.abs(res_fourier))
    #theta = np.angle(res_fourier)
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.pcolormesh(ky,kx,R)
    #plt.colorbar()
    #plt.subplot(1,2,2)
    #plt.pcolormesh(ky,kx,theta)
    #plt.colorbar()
    #plt.show()

    return res_fourier,kx,ky,X,Y,padx,pady



def general_potential(xp,yp,zp,sh,x1,x2,y1,y2,z1,z2):
    r"""
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
    * xp,yp,zp : 1D arrays
        Arrays with the x, y, and z coordinates of the computation points
    * sh : tuple = (nx,ny)
        Shape of the grid

    Returns:

    * pot : 2D complex array
        Resulting array of potentials in the Fourier domain
    * kx,ky : 2D array
        Wavenumber vectors
    * X, Y : 2D array
        Meshgrid of spatial domain coordinates

    """

    # Compute dx,dy
    # compute kx, ky, dxo, dyo
    # compute omegax and omegay
    # compute omegax,omegay meshgrids
    kx, ky, padz, padx, pady, X, Y = _prep_transform(xp,yp,zp,sh)

    # compute radial wavenumber
    kr = np.sqrt(np.square(kx)+np.square(ky))

    ## compute forward model in FD
    a = .5 * (x2 - x1)      # 1/2 prism width
    b = .5 * (y2 - y1)
    x0 = x1 + a         # Center of prism horizontally
    y0 = y1 + b
    h = zp[0]
    pv = (2.*a) * (2.*b) * (z2 - z1)
    # The division gives a RuntimeWarning because of the zero wavenumber.
    # Suppress the warning.
    with np.errstate(divide='ignore', invalid='ignore'):
        pot = 4 * a * b * np.sinc(a * kx / np.pi) * np.sinc(b * ky / np.pi) * (
            (np.exp(-(z1 - h) * kr) - np.exp(-(z2 - h) * kr)) / 
            np.square(kr)) * (np.exp(-1 * 1j * kx * x0) * 
            np.exp(-1 * 1j * ky * y0))

    # correct for zero wavenumber. scale by volume
    for ii in range(0,len(kr[:,0])):
        for jj in range(0,len(kr[0,:])):
            if kr[ii,jj] == 0:
                pot[ii,jj] = pv
    #pot[(len(pot[:,0])/2),(len(pot[0,:])/2)] = pv

    return pot,kx,ky,X,Y,padx,pady

def _prep_transform(xp,yp,zp,s):
    """
    Prepare arrays for computing values in the Fourier Domain.

    Parameters:

    * xp, yp, zp : 1D arrays
        Coordinates of data observation points
    * s : tuple = (nx,ny)
        Shape of the grid

    Returns:

    * kx, ky : 2D array
        Meshgrid of wavenumbers
    * padz : 2D array
        Padded elevation grid (which should be planar!)
    * padx,pady : scalars
        Number of padding cells on either side
    * X,Y : 2D arrays
        Meshgrid of spatial coordinates

    """

    _validate_arrays(xp,yp,zp,s)
    # The next four lines are how it should be done, but I'm having trouble
    # with the wavenumbers returned by _fftfreqs
    #padz, padx, pady = transform._pad_data(zp,s)
    padz = zp.reshape(s)
    #ky, kx = transform._fftfreqs(xp,yp,s,padz.shape)


    # Note: fftfreqs returns a negative nyquist for some reason

    x = xp.reshape(s)[:,0]
    y = yp.reshape(s)[0,:]
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    [Y,X] = np.meshgrid(y,x)
    # fftshift puts the 0 wavenumber in the middle
    fx = np.fft.fftshift(2*np.pi*np.fft.fftfreq(s[0],dx))
    fy = np.fft.fftshift(2*np.pi*np.fft.fftfreq(s[1],dy))
    fx *= -1
    fy *= -1
    [ky,kx] = np.meshgrid(fy,fx)


    """
    kxo = s[0]/2
    kyo = s[1]/2
    dxo = (2*np.pi)/(dx*s[0])
    dyo = (2*np.pi)/(dy*s[1])
    kxs = np.zeros(s[0])
    kys = np.zeros(s[1])
    for ii in range(-kxo+1,kxo+1):
        kxs[(ii-1)+kxo] = ii * dxo
    for ii in range(-kyo+1,kyo+1):
        kys[(ii-1)+kyo] = ii * dyo
    [ky,kx] = np.meshgrid(kys,kxs)
    padz = zp.reshape(s)
    """
    padx = 0
    pady = 0

    #for testing only
    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.pcolormesh(ky,(kx),(kx))
    #plt.colorbar()
    #plt.subplot(1,2,2)
    #plt.pcolormesh(ky,kx,ky)
    #plt.colorbar()
    #plt.show()

    return kx,ky,padz,padx,pady,X,Y

def _validate_arrays(xp,yp,zp,s):
    """
    Verifies that the arrays and shape all match. Throws ValueError if not.

    Parameters:

    *xp,yp,zp : 1D arrays
        Arrays of coordinates
    *s : tuple = (nx,ny)
        Shape of the grid

    Returns:

    * none

    """

    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length.")
    sz = len(xp)

    # Check to make sure dimensions are correct
    if sz != s[0]*s[1]:
        raise ValueError("Input arrays and shape do not match.")

    # Check to make sure zp is planar
    if (np.sum(zp-zp[0])) != 0.:
        raise ValueError("Observation surface must be planar for Fourier ops.")

    return 0

def _fold(ain):

    nd = ain.ndim
    idx = []
    for ii in range(0,nd):
        nx = ain.shape[ii]
        kx = int(np.floor(nx/2))
        if kx > 1:
            idx.append(np.concatenate((np.arange(kx-1,nx,step=1),
                np.arange(0,kx-1,step=1)),axis=1))
        else:
            idx.append(1)
    if nd == 1:
        b = ain[idx[0]]
    elif nd == 2:
        c=ain[:,idx[1]]
        b = c[idx[0],:]
    return b

def _unfold(ain):
    
    nd = ain.ndim
    idx = []
    for ii in range(0,nd):
        nx = ain.shape[ii]
        kx = int(np.floor(nx/2))
        if kx > 1:
            idx.append(np.concatenate((np.arange(kx+1,nx,step=1),
                np.arange(0,kx+1,step=1)),axis=1))
        else:
            idx.append(1)
    if nd == 1:
        b = ain[idx[0]]
    elif nd == 2:
        c = ain[:,idx[1]]
        b = c[idx[0],:]
    return b
