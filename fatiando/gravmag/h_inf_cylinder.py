r"""
The potential fields of a homogeneous, horizontal, infinitely extending
cylinder.
"""
from __future__ import division, absolute_import

import numpy as np

from .. import utils
from .._our_duecredit import due, Doi
from ..constants import CM, T2NT

due.cite(Doi("10.1017/CBO9780511549816"),
         description='Forward modeling formula for horizontal cylinder.',
         path='fatiando.gravmag.HorizontalInfCylinder.py')

def tf(xp, yp, zp, cyls, inc, dec, pmag=None):
    r"""
    The total-field magnetic anomaly.

    The anomaly is defined as (Blakely, 1995):

    .. math::

        \Delta T = |\mathbf{T}| - |\mathbf{F}|,

    where :math:`\mathbf{T}` is the measured field and :math:`\mathbf{F}` is a
    reference (regional) field.

    The anomaly of a homogeneous cylinder can be calculated as:

    .. math::

        \Delta T \approx \hat{\mathbf{F}}\cdot\mathbf{B}.

    where :math:`\mathbf{B}` is the magnetic induction produced by the cylinder.

    The coordinate system of the input parameters is x -> North, y -> East and
    z -> Down.

    Input units should be SI. Output is in nT.

    Parameters:

    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * cyls : list of :class:`fatiando.mesher.HorizontalInfCylinder`
        The Cylinders. Cylinders must have the physical property
        ``'magnetization'``. Cylinders that are ``None`` or without
        ``'magnetization'`` will be ignored. The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization given
        as a 3-component vector.
    * inc, dec : floats
        The inclination and declination of the regional field (in degrees)
    * pmag : [mx, my, mz] or None
        A magnetization vector. If not None, will use this value instead of the
        ``'magnetization'`` property of the spheres. Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * tf : array
        The total-field anomaly

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.

    """

    fx, fy, fz = utils.dircos(inc, dec)
    if pmag is not None:
        pmx, pmy, pmz = pmag
    res = 0
    b1 = 0
    b2 = 0
    b3 = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        Bx = bx(xp, yp, zp, [cyl])
        By = by(xp, yp, zp, [cyl])
        Bz = bz(xp, yp, zp, [cyl])
        #Compute total field anomaly
        b1 += Bx
        b2 += By
        b3 += Bz
    res += (fx*b1 + fy*b2 + fz*b3)
    return res

def bx(xp, yp, zp, cyls, pmag=None):
    """
    The x component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y-> East and
    z -> Down.
    
    Input units should be SI.  Output is in nT.

    Parameters:
    
    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * cyls : list of :class:`fatiando.mesher.HorizontalInfCylinder`
        The cylinders.  Cylinders must have the physical property 
        ``'magnetization'``.  Cylinders that are ``None`` or without
        ``'magnetization'`` will be ignored.  The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization
        given as a 3-component vector, volume-averaged.
    * pmag : [mx, my, mz] or None
        A magnetization vector.  If not None, will use this value instead of
        the ``'magnetization'`` property of the spheres.  Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bx: array
        The x component of the magnetic induction.

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.
    
    """

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        #Calculate the perpendicular distance vectors
        # if r is the distance vector, a is any point along the line, p is 
        # the observation vector, and n is the unit vector along the line
        # r = - (a - p) + ((a-p)\cdot\hat{n})\hat{n}
        nx, ny, nz = utils.dircos(0., cyl.declination)
        tx = cyl.x - xp
        ty = cyl.y - yp
        tz = cyl.z - zp
        d1 = tx*nx + ty*ny + tz*nz
        rx = d1*nx - tx
        ry = d1*ny - ty
        rz = d1*nz - tz
        #Convert magnetization to dipole moment per unit length
        mx = mx * np.pi * cyl.radius**2
        my = my * np.pi * cyl.radius**2
        mz = mz * np.pi * cyl.radius**2
        #Compute B
        m = np.sqrt(mx**2 + my**2 + mz**2)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        dr = (rx**2. + ry**2. + rz**2.)**(-3./2.) * rx * (nx**2. - 1.)
        dr2 = (2. / r) * dr
        drx = -nx**2 + 1
        dry = -1. * nx * ny
        drz = -1. * nx * nz
        Bx = 2. * CM * ((mx * rx * dr2) + ((1./r**2) * mx * drx) + 
             (my * ry * dr2) + ((1./r**2) * my * dry) + (mz * rz * dr2) +
             ((1./r**2) * mz * drz))
        #Bx = ((2. * (CM) * mx ) / r**2. ) * (
        #     (2. * dprod * (rx / r)) - (mx / m))
        B += Bx
    B*=T2NT
    return B
        
def by(xp, yp, zp, cyls, pmag=None):
    """
    The y component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y-> East and
    z -> Down.
    
    Input units should be SI.  Output is in nT.

    Parameters:
    
    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * cyls : list of :class:`fatiando.mesher.HorizontalInfCylinder`
        The cylinders.  Cylinders must have the physical property 
        ``'magnetization'``.  Cylinders that are ``None`` or without
        ``'magnetization'`` will be ignored.  The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization
        given as a 3-component vector, volume-averaged.
    * pmag : [mx, my, mz] or None
        A magnetization vector.  If not None, will use this value instead of
        the ``'magnetization'`` property of the spheres.  Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * by: array
        The y component of the magnetic induction.

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.
    
    """

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        #Calculate the perpendicular distance vectors
        # if r is the distance vector, a is any point along the line, p is 
        # the observation vector, and n is the unit vector along the line
        # r = - (a - p) + ((a-p)\cdot\hat{n})\hat{n}
        nx, ny, nz = utils.dircos(0., cyl.declination)
        tx = cyl.x - xp
        ty = cyl.y - yp
        tz = cyl.z - zp
        d1 = tx*nx + ty*ny + tz*nz
        rx = d1*nx - tx
        ry = d1*ny - ty
        rz = d1*nz - tz
        #Convert magnetization to dipole moment per unit length
        mx = mx * np.pi * cyl.radius**2
        my = my * np.pi * cyl.radius**2
        mz = mz * np.pi * cyl.radius**2
        #Compute B
        m = np.sqrt(mx**2 + my**2 + mz**2)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        dr = (rx**2. + ry**2. + rz**2.)**(-3./2.) * ry * (ny**2. - 1.)
        dr2 = (2. / r) * dr
        drx = -1. * nx * ny
        dry = (-1. * ny**2) + 1
        drz = -1. * ny * nz
        By = 2. * CM * ((mx * rx * dr2) + ((1./r**2) * mx * drx) + 
             (my * ry * dr2) + ((1./r**2) * my * dry) + (mz * rz * dr2) +
             ((1./r**2) * mz * drz))

        B += By
    B*=T2NT
    return B

def bz(xp, yp, zp, cyls, pmag=None):
    """
    The z component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y-> East and
    z -> Down.
    
    Input units should be SI.  Output is in nT.

    Parameters:
    
    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * cyls : list of :class:`fatiando.mesher.HorizontalInfCylinder`
        The cylinders.  Cylinders must have the physical property 
        ``'magnetization'``.  Cylinders that are ``None`` or without
        ``'magnetization'`` will be ignored.  The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization
        given as a 3-component vector, volume-averaged.
    * pmag : [mx, my, mz] or None
        A magnetization vector.  If not None, will use this value instead of
        the ``'magnetization'`` property of the spheres.  Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bz: array
        The z component of the magnetic induction.

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.
    
    """

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        #Calculate the perpendicular distance vectors
        # if r is the distance vector, a is any point along the line, p is 
        # the observation vector, and n is the unit vector along the line
        # r = - (a - p) + ((a-p)\cdot\hat{n})\hat{n}
        nx, ny, nz = utils.dircos(0., cyl.declination)
        tx = cyl.x - xp
        ty = cyl.y - yp
        tz = cyl.z - zp
        d1 = tx*nx + ty*ny + tz*nz
        rx = d1*nx - tx
        ry = d1*ny - ty
        rz = d1*nz - tz
        #Convert magnetization to dipole moment per unit length
        mx = mx * np.pi * cyl.radius**2
        my = my * np.pi * cyl.radius**2
        mz = mz * np.pi * cyl.radius**2
        #Compute B
        m = np.sqrt(mx**2 + my**2 + mz**2)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        dr = (rx**2. + ry**2. + rz**2.)**(-3./2.) * rz * (nz**2. - 1.)
        dr2 = (2. / r) * dr
        drx = -1. * nx * nz
        dry = -1. * ny * nz
        drz = (-1. * nz**2) + 1.
        Bz = 2. * CM * ((mx * rx * dr2) + ((1./r**2) * mx * drx) + 
             (my * ry * dr2) + ((1./r**2) * my * dry) + (mz * rz * dr2) +
             ((1./r**2) * mz * drz))

        B += Bz
    B*=T2NT
    return B

def bxx(xp, yp, zp, cyls, pmag=None, noise=None):
    """
    The x derivative of the x component of the magnetic induction.

    The coordinate system of the input parameters is x -> North, y-> East and
    z -> Down.
    
    Input units should be SI.  Output is in nT.

    Parameters:
    
    * xp, yp, zp : arrays
        The x, y, and z coordinates where the anomaly will be calculated
    * cyls : list of :class:`fatiando.mesher.HorizontalInfCylinder`
        The cylinders.  Cylinders must have the physical property 
        ``'magnetization'``.  Cylinders that are ``None`` or without
        ``'magnetization'`` will be ignored.  The magnetization is the total
        (remanent + induced + any demagnetization effects) magnetization
        given as a 3-component vector, volume-averaged.
    * pmag : [mx, my, mz] or None
        A magnetization vector.  If not None, will use this value instead of
        the ``'magnetization'`` property of the spheres.  Use this, e.g., for
        sensitivity matrix building.

    Returns:

    * bxx: array
        The x component of the x derivative of magnetic induction.

    References:

    Blakely, R. J. (1995), Potential Theory in Gravity and Magnetic
    Applications, Cambridge University Press.
    
    """

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        #Calculate the perpendicular distance vectors
        # if r is the distance vector, a is any point along the line, p is 
        # the observation vector, and n is the unit vector along the line
        # r = - (a - p) + ((a-p)\cdot\hat{n})\hat{n}
        nx, ny, nz = utils.dircos(0., cyl.declination)
        tx = cyl.x - xp
        ty = cyl.y - yp
        tz = cyl.z - zp
        dotprod = tx*nx + ty*ny + tz*nz
        rx = dotprod*nx - tx
        ry = dotprod*ny - ty
        rz = dotprod*nz - tz
        #Convert magnetization to dipole moment per unit length
        mx = mx * np.pi * cyl.radius**2
        my = my * np.pi * cyl.radius**2
        mz = mz * np.pi * cyl.radius**2
        #Compute B
        m = np.sqrt(mx**2 + my**2 + mz**2)
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        p = (((mx / m) * (rx / r)) + ((my / m) * (ry / r)) + 
                ((mz / m) * (rz / r)))
        drx = -1. * nx**2. + 1              #d/dx0 (rx)
        dr = (rx / r) * drx                 #d/dx0 (r)
        d1or = (-1. * rx) * (rx**2 + ry**2 + rz**2)**(-3. / 2.) * rx * drx
        d1or2 = (2. / r) * d1or
        dp = ((mx * drx) / (m * r)) + ((d1or / m) * ((mx * rx) + (my * ry) +
             (mz * rz)))
        drxor = (rx * d1or) + (drx / r)
        Bxx = (((2 * CM * mx) / r**2) * ((2 * p * drxor) + (2 * (rx / r) *
              dp))) + ((2 * CM * mx * d1or2) * ((2. * p * (rx / r)) - 
              (mx / m)))
        
        B += Bxx
    B*=T2NT
    return B

def bxx_numerical(xp, yp, zp, cyls, pmag=None, noise=None):

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B1 = 0
    B2 = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        Bx1 = bx(xp,yp,zp,[cyl])
        Bx2 = bx(xp-0.5,yp,zp,[cyl])
        B1 += Bx1
        B2 += Bx2
    if noise is not None:
        B1 = utils.contaminate(B1,noise)
        B2 = utils.contaminate(B2,noise)
    B = (B1 - B2) / 0.5
    return B
      
def bxy_numerical(xp, yp, zp, cyls, pmag=None, noise=None):
    if pmag is not None:
        pmx, pmy, pmz = pmag
    B1 = 0
    B2 = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        B1 += bx(xp,yp,zp,[cyl])
        B2 += bx(xp,yp-0.5,zp,[cyl])
    if noise is not None:
        B1 = utils.contaminate(B1,noise)
        B2 = utils.contaminate(B2,noise)
    B = (B1 - B2) / 0.5
    return B

def bxz_numerical(xp, yp, zp, cyls, pmag=None, noise=None):
    if pmag is not None:
        pmx, pmy, pmz = pmag
    B1 = 0
    B2 = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        B1 += bx(xp,yp,zp,[cyl])
        B2 += bx(xp,yp,zp-1.,[cyl])
    if noise is not None:
        B1 = utils.contaminate(B1,noise)
        B2 = utils.contaminate(B2,noise)
    B = (B1 - B2) / 1.
    return B

def byy_numerical(xp, yp, zp, cyls, pmag=None, noise=None):

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B1 = 0
    B2 = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        B1 += by(xp,yp,zp,[cyl])
        B2 += by(xp,yp-0.5,zp,[cyl])
    if noise is not None:
        B1 = utils.contaminate(B1,noise)
        B2 = utils.contaminate(B2,noise)
    B = (B1 - B2) / 0.5
    return B

def byz_numerical(xp, yp, zp, cyls, pmag=None, noise=None):

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B1 = 0
    B2 = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        B1 += by(xp,yp,zp,[cyl])
        B2 += by(xp,yp,zp-1.0,[cyl])
    if noise is not None:
        B1 = utils.contaminate(B1,noise)
        B2 = utils.contaminate(B2,noise)
    B = (B1 - B2) / 1.
    return B

def bzz_numerical(xp, yp, zp, cyls, pmag=None, noise=None):

    if pmag is not None:
        pmx, pmy, pmz = pmag
    B1 = 0
    B2 = 0
    for cyl in cyls:
        if cyl is None:
            continue
        if 'magnetization' not in cyl.props and pmag is None:
            continue
        if pmag is None:
            mx, my, mz = cyl.props['magnetization']
        else:
            mx, my, mz = pmx, pmy, pmz
        B1 += bz(xp,yp,zp,[cyl])
        B2 += bz(xp,yp,zp-1.0,[cyl])
    if noise is not None:
        B1 = utils.contaminate(B1,noise)
        B2 = utils.contaminate(B2,noise)
    B = (B1 - B2) / 1.
    return B

