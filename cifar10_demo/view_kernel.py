import torch, demo, matplotlib.pyplot as plt
net = demo.Net()
net.load_state_dict(torch.load("demo.pth", map_location="cpu"))
kernels = net.cnn[3].weight.detach()   # ★ 第二层 Conv 权重
fig, axes = plt.subplots(4,4, figsize=(4,4))
for i in range(16):
    axes[i//4, i%4].imshow(kernels[i].mean(0), cmap='gray'); axes[i//4, i%4].axis('off')
plt.savefig("kernels.png", dpi=150); print("kernels.png 已生成")