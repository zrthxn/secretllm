import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np



# Define a simple CNN model for MNIST
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Quantization function
def quantize(x, bits=8):
    qmin = 0
    qmax = 2**bits - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    zero_point = qmin - x.min() / scale
    q_x = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return q_x.to(torch.uint8), scale, zero_point  # Explicitly convert to uint8

def quantize_and_dequantize(x, scale, zero_point):
    return ((x.float() - zero_point) * scale).to(torch.float32)  # Explicit float conversion

def hessian_weighted_quantize(x, inv_hessian, bits=8):
    """Quantize weights using Hessian information to guide quantization"""
    qmin = 0
    qmax = 2**bits - 1
    
    # Reshape inverse Hessian to match weight dimensions
    inv_hessian_reshaped = inv_hessian.view(x.shape)
    
    # Compute sensitivity weights from inverse Hessian
    # Smaller values in inv_hessian mean parameters are more sensitive
    sensitivity = 1.0 / (torch.abs(inv_hessian_reshaped) + 1e-8)
    sensitivity = sensitivity / sensitivity.max()  # Normalize
    
    # Compute initial scale and zero point
    scale = (x.max() - x.min()) / (qmax - qmin)
    zero_point = qmin - x.min() / scale
    
    # Apply quantization with Hessian-weighted error correction
    q_x = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    
    # Apply Hessian-weighted correction
    quant_error = x - (q_x - zero_point) * scale
    correction = sensitivity * quant_error / scale
    q_x = torch.clamp(torch.round(q_x + correction), qmin, qmax)
    
    return q_x.to(torch.uint8), scale, zero_point

class QuantizedLayer:
    def __init__(self, weight, scale, zero_point):
        self.weight = weight.to(torch.uint8)  # Store as uint8
        self.scale = scale
        self.zero_point = zero_point
        
    def dequantize(self):
        return quantize_and_dequantize(self.weight, self.scale, self.zero_point)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
size = 0

# Print original model size
print("Original model size:")
original_size = 0
for name, param in model.named_parameters():
    param_size = param.nelement() * param.element_size()
    original_size += param_size
    print(f'Parameter {name}: dtype={param.dtype}, shape={param.shape}, size={param_size} bytes')
print(f'Total size: {original_size} bytes')

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (for demonstration, training for a few epochs)
for epoch in range(2):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Quantization process
quantized_layers = {}
total_quantized_size = 0

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        weights = module.weight.data
        print(f'Quantizing layer {name} with shape {weights.shape}')
        
        # Calculate Hessian for the specific layer
        data_batch, target_batch = next(iter(train_loader))
        data_batch, target_batch = data_batch.to(device), target_batch.to(device)
        
        loss = criterion(model(data_batch), target_batch)
        grads = torch.autograd.grad(loss, module.weight, create_graph=True)[0]
        
        # Compute per-parameter sensitivity correctly
        hessian = torch.pow(grads, 2).detach()  # Squared gradients as Hessian approximation
        hessian_flat = hessian.view(-1)
        
        # Compute inverse with damping
        damping_factor = 1e-3
        inv_hessian_flat = 1.0 / (hessian_flat + damping_factor)
        
        # Reshape back to original weight shape
        inv_hessian = inv_hessian_flat.view(weights.shape)
        
        # Quantize weights
        q_weights, scale, zero_point = hessian_weighted_quantize(
            weights.cpu(), 
            inv_hessian.cpu(),
            bits=8
        )
        
        # Store quantized weights
        quantized_layers[name] = QuantizedLayer(q_weights, scale, zero_point)
        
        # Update module weights with dequantized values for inference
        module.weight.data = quantized_layers[name].dequantize().to(device)
        
        # Calculate storage size
        weight_size = q_weights.nelement() * 1  # uint8 = 1 byte
        bias_size = module.bias.nelement() * module.bias.element_size() if module.bias is not None else 0
        total_quantized_size += weight_size + bias_size
        print(f'Layer {name} quantized size: {weight_size + bias_size} bytes')

# Print quantized model size
print("\nQuantized model size:")
print(f'Total size: {total_quantized_size} bytes')
print(f'Size reduction: {(1 - total_quantized_size/original_size)*100:.2f}%')

# Evaluation after quantization
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        # print progress
        print(f'Running test batch with shape {data.shape}')
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Accuracy of the model on the test images after quantization: %d %%' % (100 * correct / total))

print('Finished quantization')
# print data types and size of the model
print("\nFinal model parameters:")
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if name in quantized_layers:
            weight_size = module.weight.nelement()
            print(f'Parameter {name}.weight: dtype=uint8, shape={module.weight.shape}, '
                  f'size={weight_size} bytes')
        if module.bias is not None:
            bias_size = module.bias.nelement() * module.bias.element_size()
            print(f'Parameter {name}.bias: dtype={module.bias.dtype}, shape={module.bias.shape}, '
                  f'size={bias_size} bytes')
