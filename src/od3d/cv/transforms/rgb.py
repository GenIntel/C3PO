from torchvision.transforms.transforms import Normalize
class RGB_UInt8ToFloat:
    def __init__(self):
        pass
    def __call__(self, frame):
        frame._rgb = frame.rgb / 255.
        return frame
class RGB_Normalize:
    def __init__(self, mean=None, std=None):
        self.normalize = Normalize(mean=mean, std=std)

    def __call__(self, frame):
        frame._rgb = self.normalize(frame.rgb)
        return frame