
from dataset import ImageDataset
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.notebook import tqdm
import numpy as np
from scipy import linalg
import torchvision.models as models

inception_model = models.inception_v3( weights='IMAGENET1K_V1', transform_input=False)
inception_model.fc = torch.nn.Identity()  # Remove the classification head
inception_model.dropout = torch.nn.Identity()  # Remove the classification head

inception_model.eval()  # Set the model to evaluation mode
inception_model.to('cuda:0')
def FID_score(real_images_dir, gen_images_dir, inception_model):
    

    image_dataset = ImageDataset(real_images_dir)
    trainloader = DataLoader(image_dataset, batch_size=4)

    image_dataset = ImageDataset(gen_images_dir)
    genloader = DataLoader(image_dataset, batch_size=4)
    def compute_embeddings(dataloader, model):
        model.eval()
        image_embeddings = []

        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(dataloader):
                batch = batch.to('cuda')
                embeddings = model(batch)
    #             print(embeddings)
                image_embeddings.extend(embeddings.cpu().numpy())

        return np.array(image_embeddings)

    # Assuming dataloaders are properly defined with the appropriate transformations
    real_image_embeddings = compute_embeddings(trainloader, inception_model)
    generated_image_embeddings = compute_embeddings(genloader, inception_model)[:5]
    
    
    def calculate_fid(real_embeddings, generated_embeddings):
        mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
        mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
         # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        return fid


    fid = round(calculate_fid(real_image_embeddings, generated_image_embeddings), 3)
    return fid