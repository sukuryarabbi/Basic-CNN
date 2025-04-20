from torch import nn

class CNN(nn.Module):
    def __init__(self,img_size,img_dim=1,hidden_layer=16,num_classes=10):
        super().__init__()
        self.conv1 = self.make_cnn_block(img_dim,hidden_layer)
        self.conv2 = self.make_cnn_block(hidden_layer,hidden_layer*2)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(hidden_layer*2*7*7,10)
        
        
    def make_cnn_block(self,input_dim,output_dim,kernel=3,stride=1,padding=1):
        return nn.Sequential(
            nn.Conv2d(input_dim,output_dim,kernel,stride,padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU())
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
        
        
        
        