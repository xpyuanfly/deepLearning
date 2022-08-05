import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import os
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))    # 待了解
])

train_data = MNIST(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_data, shuffle=True, batch_size=64)
test_data = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(train_data, shuffle=False, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 256)    # imagesize=28*28=784
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 10)      # 0，1，2，3，4，5，6，7，8，9

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 0.2)

if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load("./model/model.pkl"))


def train(epoch):
    for index, data in enumerate(train_loader):
        input, target = data
        optimizer.zero_grad()
        y_predict = model(input)
        loss = criterion(y_predict, target)
        loss.backward()
        optimizer.step()
        if index % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")
            print(loss.item())


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            input, target = data
            output = model(input)
            _, predict = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()
    print(correct / total)


if __name__ == '__main__':
    # img = Image.open("number.png").convert("L")
    # img = transform(img)
    # img.view(-1, 784)
    # result = model(img)
    # a, predict = torch.max(result.data, dim=1)
    # print("预测每个数字的概率：", result)
    # print("预测数字最大的概率：", a)
    # print("预测数字为:", predict.item())
    for i in range(2):
        train(i)
        test()