import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),  # (0~255) → (0~1)
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    # TODO 1: Input Normalization
    # tensor → numpy
    X_train = train_dataset.data.numpy().astype(np.float32)
    Y_train = train_dataset.targets.numpy()

    X_test = test_dataset.data.numpy().astype(np.float32)
    Y_test = test_dataset.targets.numpy()

    # flatten (28x28 → 784)
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)

    #X_train_mean = X_train.mean()
    #X_train_std = X_train.std()
    #X_train = (X_train - X_train_mean)/X_train_std
    #X_test = (X_test - X_train_mean) / X_train_std
    return X_train, Y_train, X_test, Y_test

DROPOUT_RATE = 0.5
L2_LAMBDA = 0.001

class perceptron(): # one layer perceptron
    def __init__(self, c_in, c_out, is_final=False):
        # c_in is the number of input neuron
        # c_out is the number of output neuron
        self.w = np.random.randn(c_in, c_out) * np.sqrt(2 / c_in)
        self.b = np.zeros([1, c_out])
        self.is_final = is_final
        self.mask = None  #

    def forward(self, x, training=True):
        self.x = x.copy()

        # TODO 2: Parameter noise  training 일때만 실행
        #if training:
        #    noise = np.random.normal(0, 0.01, size=self.w.shape)
        #    w_noise = self.w + noise
        #else:
        #    w_noise = self.w

        # x is input of size Batch_size X c_in
        self.h = x @ self.w + self.b
        
        if self.is_final:
            # Softmax function
            h_shift = self.h - np.max(self.h, axis=1, keepdims=True)
            exp_h = np.exp(h_shift)
            y_pred = exp_h / np.sum(exp_h, axis=1, keepdims=True)
        else:
            # ReLU
            y_pred = np.maximum(self.h, 0)

            # TODO 3: Dropout
            #확률에 따라 0으로 두기
            #if training:
            #    self.mask = (np.random.rand(*y_pred.shape)> DROPOUT_RATE).astype(np.float32)
            #    y_pred = y_pred * self.mask / (1.0 - DROPOUT_RATE)
            #else:
            #    self.mask = None
        return y_pred
    

    def backward(self, grad, learning_rate):
        # Compute gradient
        grad_h = grad.copy()
        # TODO 3: Dropout
        #dropout 된 노드에 역전파 안되게 하기
        #if self.mask is not None:
        #    grad_h = grad_h * self.mask / (1.0 - DROPOUT_RATE)

        if not self.is_final:
            grad_h[self.h < 0] = 0              
        grad_w = self.x.T @ grad_h
        grad_next = grad_h @ self.w.T 
        # Update parameters
        # TODO 4: Regularization L2 사용
        self.w = (1-L2_LAMBDA)*self.w - learning_rate * grad_w
        self.b = self.b - learning_rate * np.sum(grad_h, 0, keepdims=True)
        return grad_next
    
def evaluate(F, X, Y):
    Y_pred = X.copy()
    for p in F:
        Y_pred = p.forward(Y_pred, training=False)

    Label = [(Y==i)*1 for i in range(10)]
    Label = np.stack(Label, -1)

    loss = -np.mean(np.sum(Label * np.log(Y_pred + 1e-12), axis=1))
    acc = np.mean(np.argmax(Y_pred, -1) == Y)

    return loss, acc


# F = [perceptron(28*28, 1024), perceptron(1024, 1024), perceptron(1024, 1024), perceptron(1024, 10, True)]
F = [perceptron(28*28, 256), perceptron(256, 256), perceptron(256, 10, True)]


X_train, Y_train, X_test, Y_test = load_data()

# Input data Visualization
plt.imshow(X_train[0].reshape(28,28), cmap='gray')
plt.title(Y_train[0])
plt.show()

lr = 1e-3
batch_size = 128
N = len(X_train)
TrainLoss = []
TestLoss = []
for epoch in range(100):
    idx = np.arange(N) # get all samples's indexes
    np.random.shuffle(idx) # shuffle the indexes
    
    for start in range(0, N, batch_size):
        batch_idx = idx[start:start+batch_size]
        X = X_train[batch_idx]
        Y = Y_train[batch_idx]

        Label = [(Y==i)*1 for i in range(10)]
        Label = np.stack(Label, -1)

        Y_pred = X.copy()
        for p in F:
            Y_pred = p.forward(Y_pred)

        loss = -np.mean(np.sum(Label * np.log(Y_pred + 1e-12), axis=1))

        grad = (Y_pred - Label) / X.shape[0]
        for p in F[::-1]:
            grad = p.backward(grad, lr)

    train_loss, train_acc = evaluate(F, X_train, Y_train)
    test_loss, test_acc = evaluate(F, X_test, Y_test)

    TrainLoss.append(train_loss)
    TestLoss.append(test_loss)

    print(f"[Epoch {epoch}] "
          f"Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, "
          f"Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")

    plt.clf()
    plt.plot(TrainLoss, label='Train Loss')
    plt.plot(TestLoss, label='Test Loss')
    plt.legend()
    plt.pause(0.01)

plt.plot(TrainLoss, label='Train Loss')
plt.plot(TestLoss, label='Test Loss')
plt.legend()
plt.show()