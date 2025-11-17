import torch, torchvision, torch.nn as nn, torch.optim as optim
from torchvision import transforms

# 数据加载
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # ★ 随机裁剪（狗耳朵可能被放大）
    transforms.RandomHorizontalFlip(p=0.5),    # ★ 50 % 左右翻转（狗左右都学）
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # ★ 亮度/对比度扰动
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

# 网络定义
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(),
    nn.MaxPool2d(2),                      # 16×16
    nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
    nn.MaxPool2d(2),                      # 8×8
    nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),  # ★ 新增第三层
    nn.MaxPool2d(2),                      # 4×4
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10))             # ★ 输入改成 64*4*4
    def forward(self, x): return self.cnn(x)

# ========== 训练入口 ==========
if __name__ == "__main__":
    device = torch.device("cpu")
    net = Net().to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(10):          # 训练 10 个 epoch
        running_loss = 0.0
        for i, (x, y) in enumerate(trainloader, 0):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            outputs = net(x)
            loss = crit(outputs, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/100:.3f}")
                running_loss = 0.0
    print("Finished 10 epoch on CPU!")
        # ===== 验证集准确率 =====
    net.eval()                 # 切换到评估模式
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            pred = net(x)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"验证准确率: {acc:.3f}  ({correct}/{total})")
    net.train()                # 切回训练模式（如果后面还要继续训练）
    torch.save(net.state_dict(), 'demo.pth')
    print('demo.pth 已保存')