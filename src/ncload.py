"""
Script to load NetCDF files and read variables.
Written by:
    yilong.wang@lsce.ipsl.fr
"""

from netCDF4 import Dataset, Variable


class NCLoad:
    """
    A class for loading NetCDF files and accessing their variables.

    This class encapsulates the functionality to open a NetCDF file (with or without a '.nc'
    extension) and provides methods to retrieve one or more variables from the file.
    """

    def __init__(self, fname: str) -> None:
        """
        Initialize the NetCDF loader and open the specified file.

        Parameters:
            fname (str): Path to the NetCDF file. The file may be specified with or without the '.nc' extension.
        """
        ncfile = fname if fname.endswith('.nc') else fname + '.nc'
        self.nc: Dataset = Dataset(ncfile, 'r')
        self.name: str = ncfile

    def _getvar(self, vname: str) -> Variable:
        """
        Retrieve a single variable from the NetCDF file.

        Parameters:
            vname (str): The name of the variable to retrieve.

        Returns:
            Variable: The NetCDF variable object corresponding to vname.
        """
        return self.nc.variables[vname]

    def get(self, *vnames: str):
        """
        Retrieve one or more variables from the NetCDF file.

        Parameters:
            *vnames (str): One or more variable names to retrieve.

        Returns:
            Variable: If a single variable name is provided.
            list[Variable]: If multiple variable names are provided.
        """
        varlist = [self._getvar(vname) for vname in vnames]
        self._refvar = varlist[0]
        return varlist[0] if len(vnames) == 1 else varlist

    def close(self) -> None:
        """
        Close the NetCDF file.
        """
        self.nc.close()
