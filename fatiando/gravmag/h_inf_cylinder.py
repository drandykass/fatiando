r"""
The potential fields of a homogeneous, horizontal, infinitely extending
cylinder.
"""
from __future__ import division, absolute_import

import numpy as np

from .. import utils
from .._our_duecredit import due, Doi
from ..constants import CM

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
        Bx = (( 2 * (CM) * mx ) / cyl.radius**2 ) * ( 2 *
             (dprod * (rx / r)) - (mx / m))
        By = (( 2 * (CM) * my ) / cyl.radius**2 ) * ( 2 *
             (dprod * (ry / r)) - (my / m))
        Bz = (( 2 * (CM) * mz ) / cyl.radius**2 ) * ( 2 *
             (dprod * (rz / r)) - (mz / m))
        #Compute total field anomaly
        res += (fx*Bx + fy*By + fz*Bz)
    return res
        
        

        

