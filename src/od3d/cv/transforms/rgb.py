from torchvision.transforms.transforms import Normalize
import torchvision
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

class RGB_Random:

    def __init__(self):
        self.transform = torchvision.transforms.Compose([
            #torchvision.transforms.RandomApply(
            #    torchvision.transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)), p=0.1),
            torchvision.transforms.RandomSolarize(threshold=128, p=0.2),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
        ])

    def __call__(self, frame):
        frame._rgb = self.transform(frame.rgb)
        return frame