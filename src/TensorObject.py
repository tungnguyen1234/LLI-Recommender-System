class TensorObject():
    def __init__(self, device, dim, dataname, percent, limit = None):
        self.device = device
        self.dim = dim
        self.percent = percent
        self.limit = limit 
        self.dataname = dataname