import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Устройства для вычислений (GPU если доступен)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Нейронная сеть для PINN
class PINNModel(nn.Module):
    def __init__(self, hidden_size=20):
        super(PINNModel, self).__init__()
        self.layers_stack = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.layers_stack(x)

# Определяем уравнения для разных формулировок

# 1. Для второй формулировки (второй порядок ОДУ)
def pde_second_order(x, t, omega):
    dxdt = torch.autograd.grad(x, t, torch.ones_like(t), create_graph=True)[0]
    d2xdt2 = torch.autograd.grad(dxdt, t, torch.ones_like(t), create_graph=True)[0]
    F = torch.cos(omega * t)  # Внешняя сила
    residual = d2xdt2 + omega**2 * x - F  # Резидуал уравнения
    return residual

# 2. Для первой системы (первый порядок)
def pde_first_order(x, y, t, omega):
    dxdt = y
    dydt = -omega**2 * x + torch.cos(omega * t)
    return dxdt, dydt

# 3. Для альтернативной первой системы
def pde_alternative(x, y, t, omega):
    dxdt = omega * y + (1/omega) * torch.sin(omega * t)
    dydt = -omega * x
    return dxdt, dydt

# Функция потерь для сравнения
def loss_fn(model, t, omega, formulation=1):
    # Получаем решение модели
    if formulation == 1:
        x = model(t)
        res = pde_second_order(x, t, omega)
        return torch.mean(res**2)
    elif formulation == 2:
        y = model(t)
        x = y[:, 0:1]
        res1, res2 = pde_first_order(x, y, t, omega)
        return torch.mean(res1**2 + res2**2)
    elif formulation == 3:
        y = model(t)
        x = y[:, 0:1]
        res1, res2 = pde_alternative(x, y, t, omega)
        return torch.mean(res1**2 + res2**2)

# Обучение модели PINN
def train_model(formulation, omega, t, epochs=1000, learning_rate=0.001):
    model = PINNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, t, omega, formulation)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
    
    return model

# Тестирование модели
def test_model(model, t, omega):
    with torch.no_grad():
        x_pred = model(t)
    return x_pred

# Время и параметры для обучения
t = torch.linspace(0, 10, 1000).reshape(-1, 1).to(device)  # Вектор времени
t.requires_grad = True
omega = 2 * np.pi * 2  # Частота

# Обучаем модели для разных формулировок
model_1 = train_model(1, omega, t)
model_2 = train_model(2, omega, t)
model_3 = train_model(3, omega, t)

# Тестируем модели
x_pred_1 = test_model(model_1, t, omega)
x_pred_2 = test_model(model_2, t, omega)
x_pred_3 = test_model(model_3, t, omega)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(t.cpu().detach().numpy(), x_pred_1.cpu().detach().numpy(), label='Formulation 1')
plt.plot(t.cpu().detach().numpy(), x_pred_2.cpu().detach().numpy(), label='Formulation 2')
plt.plot(t.cpu().detach().numpy(), x_pred_3.cpu().detach().numpy(), label='Formulation 3')
plt.legend()
plt.xlabel('Time (t)')
plt.ylabel('Displacement (x)')
plt.title('Comparison of different formulations')
plt.savefig('disp2time.png')
plt.show()
