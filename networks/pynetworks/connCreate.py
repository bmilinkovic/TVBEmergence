from copy import copy

import numpy
import scipy.stats
from tvb.basic.neotraits.api import Attr, NArray, List, HasTraits, Int, narray_summary_info
from tvb.basic.readers import ZipReader, H5Reader, try_get_absolute_path


# Creating a class that can create a connectivity object which can alter the network to create a variety of subnetworks
# that is desired


class connCreate(Connectivity):
