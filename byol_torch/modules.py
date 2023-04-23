from torchvision import models
from torch import nn
import torch.nn.functional as F
import math
from sklearn.manifold import TSNE
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

##########
# Losses #
##########
def BYOL_loss(p, z):
    p = F.normalize(p, dim=1, p=2)
    z = F.normalize(z, dim=1, p=2)
    return 2 - 2 * (p * z).sum(dim=1)

def Classifier_loss(preds, labels):
    return nn.CrossEntropyLoss(reduction="mean")(preds, labels)

###########
# Helpers #
###########
def tsne_plot(embeddings, labels, filename, num_classes = 10):
    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    plt.figure(1, figsize=(8, 8))
    for lab in range(num_classes):
        indices = labels == lab
        plt.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab, alpha=1)
    
    plt.legend(markerscale=1)
    plt.savefig(filename, dpi=800)

class AverageMeter():
    def __init__(self):
        self.val = 0.0
        self.cnt = 0
    def update(self, x):
        self.cnt += 1
        self.val += x
    def avg(self):
        return self.val / self.cnt
    def reset(self):
        self.val = 0.0
        self.cnt = 0

class EMAHelper():
    # Usage: 
    #   network
    #   ema_helper = EMAHelper(epochs, tau_base)
    #   ema_helper.register(target_network)
    # For update:
    #   ema_helper.update(online_network, epoch)
    #   ema_helper.ema(target_network) # updates network in-place
    def __init__(self, epochs, tau_base=0.99):
        self.epochs = epochs
        self.tau_base = tau_base

        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, module, epoch):
        tau =  1 - (1 - self.tau_base) * (math.cos(math.pi * epoch / self.epochs) + 1) / 2
        for name, param in module.named_parameters():
            self.shadow[name].data = (1 - tau) * param.data + tau * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            param.data.copy_(self.shadow[name].data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

############
# Networks #
############
class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, output_dim):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim, bias=False), # BN에 bias가 이미 존재 - bias = False 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class LinearClassifier(nn.Module):
    def __init__(self, in_features, num_classes = 10):
        super(LinearClassifier, self).__init__()
        
        self.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.classifier(x)
                
class ResNetModel(nn.Module):
    def __init__(self, args, name):
        super(ResNetModel, self).__init__()
        self.name = name
        self.backbone, self.projector, self.predictor = self.build_model(args)
        
    def build_model(self, args):
        backbone = models.__dict__[args.model](pretrained=args.pretrained)
        in_features = backbone.fc.in_features
        self.embedding_dim = in_features
        backbone.fc = nn.Identity()
        projector = MLP(in_features, args.hidden_dim, args.proj_dim)
        predictor = MLP(args.proj_dim, args.hidden_dim, args.pred_dim)
        return backbone, projector, predictor
    
    def forward(self, x, return_embedding=False, return_projection=False, return_prediction=True):
        if return_embedding:
            y = self.backbone(x)
            return y
        
        if return_projection:
            # target network만 projection 필요
            assert self.name == "target"
            y = self.backbone(x)
            z = self.projector(y)
            return z
        
        if return_prediction:
            # online network만 prediction 필요
            assert self.name == "online"
            y = self.backbone(x)
            z = self.projector(y)
            q_z = self.predictor(z)
            return q_z
        
        
        
        

        