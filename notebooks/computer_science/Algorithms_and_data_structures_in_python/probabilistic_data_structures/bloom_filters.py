import bitarray
import math
import mmh3


class BloomFilter(object):
    
    def __init__(self, capacity, error=0.005):
        self.capacity = capacity
        self.error = error
        self.num_bits = int(-capacity  * math.log(error) / math.log(2)**2) + 1
        self.num_hashes = int(self.num_bits * math.log(2) / float(capacity)) + 1
        self.data = bitarray.bitarray(self.num_bits)

    def _indexes(self, key):
        h1, h2 = mmh3.hash64(key)
        for i in range(self.num_hashes):
            yield(h1 + i * h2) % self.num_bits 

