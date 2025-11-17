import torch, matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import demo
import numpy as np

net = demo.Net()
net.load_state_dict(torch.load('demo.pth', map_location='cpu'))
net.eval()

# 加载狗的照片
img = Image.open('mydog.jpg').convert('RGB')   # ★ 这里改成 mydog.jpg
img_for_network = img.resize((32, 32))
img_for_display = img.resize((128, 128))

x = transforms.ToTensor()(img_for_network).unsqueeze(0)

activations = []
gradients = []

target_layer = net.cnn[8]   # 最后一层卷积

def forward_hook(module, input, output):
    activations.append(output.detach().clone())

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach().clone())

handle_forward = target_layer.register_forward_hook(forward_hook)
handle_backward = target_layer.register_full_backward_hook(backward_hook)

# 前向 + 反向
logits = net(x)
prob, pred = logits.softmax(1).max(1)
target_class = pred.item()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
predicted_class_name = classes[target_class]
actual_class_name = 'dog'

print(f"模型预测: {predicted_class_name}, 概率: {prob.item():.3f}")
print(f"实际类别: {actual_class_name}")

# 反向传播
net.zero_grad()
one_hot = torch.zeros_like(logits)
one_hot[0, target_class] = 1
logits.backward(gradient=one_hot)

handle_forward.remove()
handle_backward.remove()

# 生成CAM
feat = activations[0][0]
grad = gradients[0][0]
weights = grad.mean(dim=(1, 2), keepdim=True)
cam = (weights * feat).sum(0).relu()

# 调整大小用于显示
cam_resized = transforms.Resize((128, 128))(cam.unsqueeze(0)).squeeze(0)
cam_np = cam_resized.detach().numpy()

if cam_np.max() > cam_np.min():
    cam_normalized = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
else:
    cam_normalized = cam_np

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_for_display)
axes[0].set_title('原图 (128×128)')
axes[0].axis('off')

axes[1].imshow(img_for_display)
axes[1].imshow(cam_normalized, cmap='jet', alpha=0.6, interpolation='nearest')
status = "✓ 正确" if predicted_class_name == actual_class_name else "✗ 错误"
axes[1].set_title(f'{status}  预测: {predicted_class_name} (p={prob.item():.3f})\n实际: {actual_class_name}', 
                  color='green' if predicted_class_name == actual_class_name else 'red')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('gradcam_dog_analysis.png', dpi=200, bbox_inches='tight')
print('分析结果已保存为 gradcam_dog_analysis.png')