
from dataset import ImageDataset
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
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
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch = batch.to('cuda')
                embeddings = model(batch)
    #             print(embeddings)
                image_embeddings.extend(embeddings.cpu().numpy())

        return np.array(image_embeddings)

    # Assuming dataloaders are properly defined with the appropriate transformations
    real_image_embeddings = compute_embeddings(trainloader, inception_model)
    generated_image_embeddings = compute_embeddings(genloader, inception_model)
    
    print(real_image_embeddings.shape)
    def calculate_fid(real_embeddings, generated_embeddings):
        mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
        mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings,  rowvar=False)

        diff = mu1 - mu2
        try:
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        except:
            print(sigma1)
            print(sigma2)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        eps = 1e-6
        if not np.isfinite(covmean).all():
            tr_covmean = np.sum(np.sqrt(((np.diag(sigma1) * eps) * (np.diag(sigma2) * eps)) / (eps * eps)))

        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


    fid = round(calculate_fid(real_image_embeddings, generated_image_embeddings), 3)
    return fid
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate VNFASHIONDIFF")
    parser.add_argument("--path_gen", type=str, help="Path to folder images gen")
    parser.add_argument("--path_origin", type=str, help="Path to folder image real")

    args = parser.parse_args()
    score = FID_score(gen_images_dir=args.path_gen, real_images_dir=args.path_origin, inception_model =inception_model)
    print(args.path_gen)
    print(f"Score FID: {score}")