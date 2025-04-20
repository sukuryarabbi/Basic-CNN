from torch import nn


class CNN(nn.Module):
    def __init__(self,img_size,img_dim=3,hidden_layer=64,num_classes=10):
        super().__init__()
        
        self.model = nn.Sequential(
            self.make_cnn_block(img_dim,hidden_layer),
            self.make_cnn_block(hidden_layer,hidden_layer*2),
            self.make_cnn_block(hidden_layer*2,hidden_layer*4),
            self.make_cnn_block(hidden_layer*4,hidden_layer*8),
            self.make_cnn_block(hidden_layer*8,hidden_layer*16,final_layer=True)
            )
        
        
    def make_cnn_block(self,input_dim,output_dim,kernel=3,stride=1,padding=1,final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_dim,output_dim,kernel,stride,padding),
                nn.BatchNorm2d(output_dim),
                nn.ReLU())
        else:
            return nn.Sequential(
                nn.Conv2d(input_dim,output_dim,kernel,stride,padding),
                nn.Sigmoid()
                )
    
    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0),-1)
        return x
        
        
        
        