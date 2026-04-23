from abc import ABC

class Hierarchy(ABC):
    def get(self, key):
        return self.mapping[key]
    
class Blocks(Hierarchy):
    def __init__(self, size: int):
        base_items = list(range(size))
        block_sizes = [s+1 for s in base_items if size%(s+1) == 0]
        self.mapping = {}
        key = 0

        for size in block_sizes:
            chunks = [base_items[i:i+size] for i in range(0, len(base_items), size)]
            for chunk in chunks:
                self.mapping[key] = chunk
                key += 1

class OrderedPairs(Hierarchy):
    def __init__(self, size: int):
        base_items = list(range(size))
        self.mapping = {}
        key = 0

        for element1 in base_items:
            for element2 in range(element1+1, base_items[-1]+1):
                self.mapping[key] = [element1, element2]
                key += 1
