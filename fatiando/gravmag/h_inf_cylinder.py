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
        dprod = (((mx / m) * (rx / r)) + ((my / m) * (ry / r)) + 
                ((mz / m) * (rz / r)))
        #Bx = (( 2 * (CM) * mx ) / cyl.radius**2 ) * ( 2 *
        #     (dprod * (rx / r)) - (mx / m))
        #By = (( 2 * (CM) * my ) / cyl.radius**2 ) * ( 2 *
        #     (dprod * (ry / r)) - (my / m))
        #Bz = (( 2 * (CM) * mz ) / cyl.radius**2 ) * ( 2 *
        #     (dprod * (rz / r)) - (mz / m))
        Bx = (( 2 * (CM) * mx ) / r**2 ) * ( 2 *
             (dprod * (rx / r)) - (mx / m))
        By = (( 2 * (CM) * my ) / r**2 ) * ( 2 *
             (dprod * (ry / r)) - (my / m))
        Bz = (( 2 * (CM) * mz ) / r**2 ) * ( 2 *
             (dprod * (rz / r)) - (mz / m))
        #Compute total field anomaly
        res += (fx*Bx + fy*By + fz*Bz)
    res*=T2NT
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
        dprod = (((mx / m) * (rx / r)) + ((my / m) * (ry / r)) + 
                ((mz / m) * (rz / r)))
        #Bx = (( 2 * (CM) * mx ) / cyl.radius**2 ) * ( 2 *
        #     (dprod * (rx / r)) - (mx / m))
        Bx = (( 2 * (CM) * mx ) / r**2 ) * ( 2 *
             (dprod * (rx / r)) - (mx / m))
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
        dprod = (((mx / m) * (rx / r)) + ((my / m) * (ry / r)) + 
                ((mz / m) * (rz / r)))
        By = (( 2 * (CM) * my ) / r**2 ) * ( 2 *
             (dprod * (ry / r)) - (my / m))
        #By = (( 2 * (CM) * my ) / cyl.radius**2 ) * ( 2 *
        #     (dprod * (ry / r)) - (my / m))
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
        dprod = (((mx / m) * (rx / r)) + ((my / m) * (ry / r)) + 
                ((mz / m) * (rz / r)))
        Bz = (( 2 * (CM) * mz ) / r**2 ) * ( 2 *
             (dprod * (rz / r)) - (mz / m))
        #Bz = (( 2 * (CM) * mz ) / cyl.radius**2 ) * ( 2 *
        #     (dprod * (rz / r)) - (mz / m))
        B += Bz
    B*=T2NT
    return B

def bxx(xp, yp, zp, cyls, pmag=None):
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

def bxx_numerical(xp, yp, zp, cyls, pmag=None):

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
        dprod = (((mx / m) * (rx / r)) + ((my / m) * (ry / r)) + 
                ((mz / m) * (rz / r)))
        #Bx = (( 2 * (CM) * mx ) / cyl.radius**2 ) * ( 2 *
        #     (dprod * (rx / r)) - (mx / m))
        Bx1 = (( 2 * (CM) * mx ) / r**2 ) * ( 2 *
             (dprod * (rx / r)) - (mx / m))
        xp = xp - 0.1
        tx = cyl.x - xp
        dotprod = tx*nx + ty*ny + tz*nz
        rx = dotprod*nx - tx
        ry = dotprod*ny - ty
        rz = dotprod*nz - tz
        r = np.sqrt(rx**2 + ry**2 + rz**2)
        dprod = (((mx / m) * (rx / r)) + ((my / m) * (ry / r)) + 
                ((mz / m) * (rz / r)))
        Bx2 = (( 2 * (CM) * mx ) / r**2 ) * ( 2 *
             (dprod * (rx / r)) - (mx / m))
        
        B += (Bx1-Bx2) / (0.1)
    B*=T2NT
    return B
      

