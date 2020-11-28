import numpy as np
import sys, os
sys.path.append(os.pardir)
from Img2Col import im2col
from Col2Img import col2im

x1 = np.random.randn(4,2,28,28)

#im2col함수는 이미지를 행렬 데이터로 바꾸어 준다.
#im2col(input_data, filter_height, filter_width, stride = 1, padding = 0)이 기본 설정값이다.

col1 = im2col(x1, 4,4,stride = 1, pad = 0)
print(col1.shape)
#(2500, 32)
#만약에 batch size를 4가 아니라 1로 했다면 data의 개수는 2500개가 아닌 625개였을 것이다.

class Convolution:
    def __init__(self, W, b, stride = 1, padding = 0):
        self.W = W   #가중치
        self.b = b   #편향
        self.stride = stride
        self.padding = padding

        #역전파 작용시에 사용
        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None
    
    def forward(self, x):
        #Filter개수, Channel, Height, Width
        FN, C, FH, FW = self.W.shape
        N,C,H,W = x.shape

        out_h = int(1 + (H+2*self.padding - FH) / self.stride)
        out_w = int(1 + (W + 2*self.padding - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W)
        #numpy의 transpose 함수를 이용해서 축의 순서를 변경한다.
        #data_format = channels first로 저장할 떄 와 같은 순서로 데이터의 형상을 바꾼 것이다.
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    #데이터를 im2col을 이용해서 2차원으로 펼쳐 놓았기 떄문에 역전파를 수행해 주기가 쉽다.
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        #전체 데이터의 출력값의 합을 구해줌
        self.db = np.sum(dout, axis = 0)
        self.dW = np.dot(self.col.T, dout)
        #다시 원래의 이미지의 shape대로 4차원으로 바꾸어 준다.
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        #위에서 구한 가중치의 traverse와의 합성곱을 구해준다.
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


