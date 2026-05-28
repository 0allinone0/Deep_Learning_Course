import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WAY = 3       # for each episode, # class
K_SHOT = 3      # for each class, # support sample
Q_QUERY = 8     # for each class, # query sample
NUM_META_STEPS = 500
META_BATCH_SIZE = 4
LR = 1e-3
LAMBDA_CON = 0.1

TRAIN_CLASSES = [0, 1, 2, 3, 4, 5, 6]
TEST_CLASSES = [7, 8, 9]

# CLASS_NAMES = [
#     "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
#     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
# ]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_data = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)


def make_class_index(dataset, classes):
    targets = np.array(dataset.targets)
    return {c: np.where(targets == c)[0] for c in classes}

train_index = make_class_index(train_data, TRAIN_CLASSES)
test_index = make_class_index(test_data, TEST_CLASSES)

def sample_episode(dataset, class_index, n_way, k_shot, q_query):
    selected_classes = random.sample(list(class_index.keys()), n_way)

    sx, sy, qx, qy = [], [], [], []

    for new_label, cls in enumerate(selected_classes):
        selected_idx = np.random.choice(
            class_index[cls],
            size=k_shot + q_query,
            replace=False
        )

        support_idx = selected_idx[:k_shot]
        query_idx = selected_idx[k_shot:]

        for idx in support_idx:
            img, _ = dataset[int(idx)]
            sx.append(img)
            sy.append(new_label)

        for idx in query_idx:
            img, _ = dataset[int(idx)]
            qx.append(img)
            qy.append(new_label)

    sx = torch.stack(sx).to(DEVICE)
    sy = torch.tensor(sy).long().to(DEVICE)
    qx = torch.stack(qx).to(DEVICE)
    qy = torch.tensor(qy).long().to(DEVICE)

    return sx, sy, qx, qy, selected_classes


class Encoder(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 14 -> 7
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(7), # 7 -> 1
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        z = self.fc(h)
        z = F.normalize(z, dim=1)
        return z

def make_prototypes(z_support, y_support, n_way):
    prototypes = []
    for c in range(n_way):
        prototypes.append(z_support[y_support == c].mean(dim=0))
    prototypes = torch.stack(prototypes)
    prototypes = F.normalize(prototypes, dim=1)
    return prototypes


def proto_logits(z_query, prototypes, temperature=0.2):
    return z_query @ prototypes.T / temperature

def supcon_loss(z, y, temperature=0.1):
    """
    z: embedding, y: label
    """
    sim = z @ z.T / temperature

    n = z.size(0)
    self_mask = torch.eye(n, device=DEVICE).bool()

    pos_mask = (y[:, None] == y[None, :])
    pos_mask[self_mask] = False

    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = F.log_softmax(sim, dim=1)

    loss = -(log_prob * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp_min(1)
    return loss.mean()

@torch.no_grad()
def evaluate(model, dataset, class_index, episodes=100):
    model.eval()
    acc_list = []

    for _ in range(episodes):
        sx, sy, qx, qy, _ = sample_episode(dataset, class_index, N_WAY, K_SHOT, Q_QUERY)

        z_support = model(sx)
        z_query = model(qx)

        prototypes = make_prototypes(z_support, sy, N_WAY)
        logits = proto_logits(z_query, prototypes)

        pred = logits.argmax(dim=1)
        acc = (pred == qy).float().mean().item()
        acc_list.append(acc)

    model.train()
    return np.mean(acc_list)

model = Encoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for meta_step in range(1, NUM_META_STEPS + 1):

    meta_loss = 0

    for task_id in range(META_BATCH_SIZE):
        sx, sy, qx, qy, _ = sample_episode(
            train_data, train_index, N_WAY, K_SHOT, Q_QUERY
        )

        z_support = model(sx)
        z_query = model(qx)

        prototypes = make_prototypes(z_support, sy, N_WAY)

        logits = proto_logits(z_query, prototypes)
        loss_proto = F.cross_entropy(logits, qy)

        z_all = torch.cat([z_support, z_query], dim=0)
        y_all = torch.cat([sy, qy], dim=0)
        loss_con = supcon_loss(z_all, y_all)

        task_loss = loss_proto + LAMBDA_CON * loss_con

        meta_loss += task_loss

    meta_loss = meta_loss / META_BATCH_SIZE

    optimizer.zero_grad()
    meta_loss.backward()
    optimizer.step()

    if meta_step % 5 == 0:
        test_acc = evaluate(model, test_data, test_index, episodes=50)
        print(
            f"Step {meta_step:04d} | "
            f"Loss {meta_loss.item():.3f} | "
            f"Test Acc {test_acc:.3f}"
        )