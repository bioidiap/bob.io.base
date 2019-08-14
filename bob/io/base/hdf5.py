#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>


from ._library import File as _File_C, HDF5File as _HDF5File_C


class File(_File_C):
    __doc__ = _File_C.__doc__

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class HDF5File(_HDF5File_C):
    __doc__ = _HDF5File_C.__doc__

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self.close()

    def __contains__(self, x):
        __doc__ = self.has_key.__doc__
        return self.has_key(x)

    def __iter__(self):
        __doc__ = self.keys.__doc__
        return iter(self.keys())

    def __getitem__(self, name):
        __doc__ = self.get.__doc__
        return self.get(name)

    def __setitem__(self, name, value):
        __doc__ = self.set.__doc__
        return self.set(name, value)

    def values(self):
        """Yields the datasets contained in the current directory.

    Yields
    -------
    object
        The datasets that are being read.
    """
        return (self[key] for key in self)

    def items(self):
        """Yields the keys and the datasets contained in the current directory.

    Yields
    -------
    tuple
        The key and the datasets that are being read in a tuple.
    """
        return ((key, self[key]) for key in self)
        
        
    def to_dict(self, tree_level=0):
        """
        Read the structure of a :py:class:`bob.io.base.HDF5File` and returns it as a :py:class:`dict`
        
        **Parameters**
        
        tree_level:
          Controls which level of the hdf5 tree the algorithm is
          
        **Returns**
          A :py:class:`dict` with the exact same structure of the :py:class:`bob.io.base.HDF5File`
          
        """

        output_dict = dict()
        visited_groups = [] # Keeping track of visited groups. We don't have the function the returns only leaves implemented
        for key, value in self.items():
            key = key[1:].split("/")[tree_level]
            if self.has_group(key) and key not in visited_groups:
                visited_groups.append(key)
                self.cd(key)
                output_dict[key] = self.to_dict(tree_level=tree_level+1)
                self.cd("..")
            else:
                if key not in visited_groups:
                    output_dict[key] = value
            
        return output_dict
        
        
    def import_from_dict(self, input_dict):
        """
        Save the content of a :py:class:`dict` into a :py:class:`bob.io.base.HDF5File`.
        This function replicates the exact same structure from one format to the other one.
        
        **Parameters**
              
        input_dict: :py:class:`dict`
          Input dictionary to be persisted
          
        """

        for k in input_dict:
            if isinstance(input_dict[k], dict):
                self.create_group(k)
                self.cd(k)
                self.import_from_dict(input_dict[k])
                self.cd("..")
            else:
                self.set(k, input_dict[k])

