import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


# Verifica che tutte le immagini siano valide
def check_images(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                print(f"Errore con l'immagine {img_path}: {e}")



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Peso per le classi difficili
        self.gamma = gamma  # Parametro per penalizzare gli esempi facili
        self.weight = weight  # Pesi personalizzati per le classi
        self.reduction = reduction  # Tipo di riduzione ('mean', 'sum', 'none')
    
    def forward(self, inputs, targets):
        # Calcolare la probabilità predetta usando softmax
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Recupera il logaritmo delle probabilità per le classi corrette
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.view(-1, 1), 1)
        
        # Calcolare il termine (1 - p_t)^gamma
        p_t = torch.sum(targets_one_hot * probs, dim=1)  # p_t per ogni esempio
        focal_factor = (1 - p_t) ** self.gamma
        
        # Calcolare la Focal Loss
        loss = -self.alpha * focal_factor * log_probs.gather(1, targets.view(-1, 1))
        
        # Se ci sono pesi per le classi, applicali
        if self.weight is not None:
            loss = loss * self.weight[targets]

        # Riduzione (media o somma)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")