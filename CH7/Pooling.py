import numpy as np
from Img2Col import im2col
from Col2Img import col2im

class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, padding = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

        self.x = None
        self.arg_max = None

    def forward(self, x):
        #N은 이미지 개수, C는 channel수, H는 높이, W는 너비
        N, C, H, W = x.shape
        #출력값의 높이와 너비의 크기를 계산
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        #image data를 column data인 2차원으로 펼쳐주어서 4중 for문을 사용할 필요가 없게끔 한다.
        col = im2col(x, out_h, out_w, self.stride, self.padding)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        out = np.max(col, axis = 1)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)

        return dx