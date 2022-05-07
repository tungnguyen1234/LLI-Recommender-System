class TensorObject():
    def __init__(self, device, dataname, percent, limit = None):
        self.device = device 
        self.percent = percent
        self.limit = limit 
        self.dataname = dataname