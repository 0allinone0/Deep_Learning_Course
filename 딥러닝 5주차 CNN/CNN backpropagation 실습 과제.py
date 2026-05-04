import numpy as np

class perceptron():  # one layer perceptron
    def __init__(self, c_in, c_out, patch_sz, is_final=False):
        # c_in is the number of input neuron
        # c_out is the number of output neuron
        self.patch_sz = patch_sz
        self.c_out = c_out
        self.w = np.random.rand(c_in * patch_sz * patch_sz, c_out) * 1e-3
        self.b = np.zeros((1, c_out))
        self.is_final = is_final

    def forward(self, x):
        self.x = x.copy()
        batch_sz, h, w, c = x.shape

        h_out = h - (self.patch_sz - 1)
        w_out = w - (self.patch_sz - 1)

        self.h = np.zeros((batch_sz, h_out, w_out, self.c_out))

        for i in range(h_out):
            for j in range(w_out):
                h_patch = x[:, i:i+self.patch_sz, j:j+self.patch_sz, :]
                h_patch = h_patch.reshape(batch_sz, -1)
                h_val = h_patch @ self.w + self.b
                self.h[:, i, j, :] = h_val

        if self.is_final:
            # softmax
            exp_h = np.exp(self.h)
            y_pred = exp_h / np.sum(exp_h, axis=-1, keepdims=True)
        else:
            # ReLU
            y_pred = np.maximum(self.h, 0)

        return y_pred
    
def backward(self, grad, learning_rate):
    # Compute gradient
    grad_h = grad.copy()    #이전 레이어에서 넘어온 gradient

    batch_sz, h, w, c = grad_h.shape #batch_sz는 reshape에, h,w는 루프 범위를 위해
    if not self.is_final:    
        grad_h[self.h < 0] = 0  # ReLU backward

    #gradient 저장 공간 초기화
    grad_next = np.zeros_like(self.x)  #이전 레이어로 전달할 gradient값
    overlap   = np.zeros_like(self.x)  #각 픽셀이 몇 번 패치에 참여했는지 카운트
    grad_w    = np.zeros_like(self.w)  #weight의 gradient
    grad_b    = np.zeros((1, self.c_out)) #bias의 gradient

    for i in range(h):
        for j in range(w):
            # 현재 위치의 입력 패치 flatten
            x = self.x[:, i:i+self.patch_sz, j:j+self.patch_sz, :].reshape(batch_sz, -1)

            # upstream gradient: (batch_sz, c_out)
            g = grad_h[:, i, j, :]

            # ∂L/∂W 누적
            grad_w += x.T @ g

            # ∂L/∂b 누적
            grad_b += g.sum(axis=0, keepdims=True)

            # ∂L/∂X 패치에 누적
            grad_x_patch = (g @ self.w.T).reshape(batch_sz, self.patch_sz, self.patch_sz, -1)
            grad_next[:, i:i+self.patch_sz, j:j+self.patch_sz, :] += grad_x_patch

            # 각 픽셀이 패치에 참여한 횟수 누적
            overlap[:, i:i+self.patch_sz, j:j+self.patch_sz, :] += 1

    # 중복 참여 횟수로 나눠서 평균 gradient
    grad_next /= np.maximum(overlap, 1)

    # 파라미터 업데이트
    self.w -= learning_rate * grad_w
    self.b -= learning_rate * grad_b

    return grad_next
    