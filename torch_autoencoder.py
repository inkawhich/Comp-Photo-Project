import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = 8
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
HEIGHT=WIDTH=224


transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
dark_dataset = ImageFolder('./Dark-Image-Data/',transform=transform)
light_dataset = ImageFolder('./Dark-Image-Data_lime/',transform=transform)

dark_loader=DataLoader(dataset=dark_dataset,batch_size=BATCH_SIZE,shuffle=False)
light_loader=DataLoader(dataset=light_dataset,batch_size=BATCH_SIZE,shuffle=False)



class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

model = AutoEncoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE, weight_decay=1e-5)

for epoch in range(NUM_EPOCHS):
    for dark_data, light_data in zip(dark_loader,light_loader):
        dark_img, _ = dark_data
        light_img, _= light_data
        dark_img = Variable(dark_img, requires_grad=True).cuda()
        light_img = Variable(light_img, requires_grad=True).cuda()
        decoded = model(dark_img)
        loss = criterion(decoded,light_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Loss = %.3f" % loss.data[0])


torch.save(model.state_dict(),'./torch_model/')
