# Crée un fichier check_data.py dans src/models/
import sys
sys.path.append('.')
from dataset import get_dataloaders, IDX_TO_CLASS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

train_loader, _, _ = get_dataloaders(batch_size=16)
images, labels = next(iter(train_loader))

# Dénormaliser
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
img = images[0].clone()
for i in range(3):
    img[i] = img[i] * std[i] + mean[i]
img = img.permute(1, 2, 0).numpy().clip(0, 1)

plt.imshow(img)
plt.title(f"Classe : {IDX_TO_CLASS[labels[0].item()]}")
plt.savefig('check_image.png')
print("Image sauvegardée : check_image.png")
print(f"Distribution labels batch : {[IDX_TO_CLASS[l.item()] for l in labels]}")