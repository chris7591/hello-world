import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if not flip_img:
            return img
        
        return img[:, ::-1, :]
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        if len(img.shape) != 3:
            print('shape is not correct !!')
            img = img.reshape(28, 28, 1)
        H, W, C = img.shape
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        padded_image = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0,0)))

        start_shift_x = shift_x + self.padding
        start_shift_y = shift_y + self.padding

        return padded_image[start_shift_x:start_shift_x + H, start_shift_y:start_shift_y+W,:]
        ### END YOUR SOLUTION
