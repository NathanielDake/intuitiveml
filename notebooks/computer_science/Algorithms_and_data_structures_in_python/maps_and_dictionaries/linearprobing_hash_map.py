import random
from abc import abstractmethod
from collections import MutableMapping


class MapBase(MutableMapping):
    """Abstract base class that includes a nonpublic _Item class
       Notes:
       - We see that by inheriting from MutableMapping we must implement the Abstract Methods:
         __getitem__, __setitem__, __delitem__, __iter__, and __len__ (see docs)
         
    """
    class _Item:
        """Lightweight composite to store key-value pairs as map items."""

        __slots__ = '_key', '_value'

        def __init__(self, k, v):
            self._key = k
            self._value = v
        
        def __eq__(self, other):
            return self._key == other._key # Compare items based on their keys

        def __ne__(self, other):
            return not(self == other) # opposite of __eq__


class HashMapBase(MapBase):
    """Abstract base class for map using hash-table with MAD compression.
        - Note: MAD stands for multiply-add-and-divide compression
        - Note: We inherit from MapBase, which inherits from MutableMapping, so we must implement certain 
                abstract methods
       High Level Overview:
        - First a _table is instantiated. This is simply a python list. Under the hood this how our hash map will
          store data.
        - We then wish to add to it via a call to __setitem__. Note that this will call a function _bucket_setitem
          that must be implemented by our concrete subclass

    """

    def __init__(self, cap=11, p=109345121):
        self._table = cap * [None]
        self._n = 0
        self._prime = p
        self._scale = 1 + random.randrange(p - 1)
        self._shift = random.randrange(p)

    def _hash_function(self, k):
        """Takes in some key, k, hashes it (maps it to an integer), and then compresses it (maps the hash 
        code to an integer within the range of indices of _table)"""
        hash_code = hash(k)
        compressed_code = (hash_code * self._scale + self._shift) % self._prime % len(self._table)
        return compressed_code

    def __len__(self):
        return self._n

    def __getitem__(self, k):
        j = self._hash_function(k) # Get index in list via the hash function
        return self._bucket_getitem(j, k) # Gets item from the bucket via this index

    def __setitem__(self, k, v):
        j = self._hash_function(k)
        self._bucket_setitem(j, k, v) # Calls _bucket_setitem which is implemented by concrete subclass
        if self._n > len(self._table) // 2: # Note that this resizing logic is common to all hash maps so we put in abstract class
            self._resize(2 * len(self._table) - 1)

    def __delitem__(self, k):
        j = self._hash_function(k)
        self._bucket_delitem(j, k)
        self._n -= 1

    def _resize(self, c):
        """Resize bucket array, called when __setitem__ causes us to occupy more than 1/2 of array slots"""
        old = list(self.items())
        self._table = c * [None]
        self._n = 0
        for (k, v) in old:
            self[k] = v

    @abstractmethod
    def _bucket_getitem(self, j, k):
        pass

    @abstractmethod
    def _bucket_setitem(self, j, k, v):
        pass

    @abstractmethod
    def _bucket_delitem(self, j, k):
        pass


class ProbeHashMap(HashMapBase):
    _AVAIL = object()

    def _is_available(self, j):
        return self._table[j] is None or self._table[j] is ProbeHashMap._AVAIL

    def _find_slot(self, j, k):
        """Search for key k in bucket at index j.

        Return (success, index) tuple. 
        If match was found, success is True and index denotes location.
        If no match found, success is False and index denotes first available spot
        """
        first_avail = None
        while True:
            if self._is_available(j):
                if first_avail is None:
                    first_avail = j
                if self._table[j] is None:
                    return (False, first_avail)
            elif k == self._table[j]._key:
                return (True, j)
            j = (j + 1) % len(self._table)

    def _bucket_getitem(self, j, k):
        found, s = self._find_slot(j, k)
        if not found:
            raise KeyError('Key Error: ' + repr(k))
        return self._table[s]._value
    
    def _bucket_setitem(self, j, k, v):
        found, s = self._find_slot(j, k)
        if not found:
            self._table[s] = self._Item(k, v) # Insert new item
            self._n += 1                      # size has increased
        else:
            self._table[s]._value = v         # overwrite existing
            
    def _bucket_delitem(self, j, k):
        found, s = self._find_slot(j, k)
        if not found:
            raise KeyError('Key Error: ' + repr(k))
        self._table[s] = ProbeHashMap._AVAIL

    def __iter__(self):
        for j in range(len(self._table)):
            if not self._is_available(j):
                yield self._table[j]._key


if __name__ == '__main__':
    pass
