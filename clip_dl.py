from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

class Flickr30kDataset(Dataset):
    def __init__(self, split):
        #self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")
        self.dataset = load_dataset("AnyModal/flickr30k", cache_dir="./huggingface_data")[split]

        self.transform = transforms.Compose(
            [
                #transforms.Resize((224, 224)),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ]
        )
        self.cap_per_image = 1

    def __len__(self):
        return self.dataset.num_rows * self.cap_per_image

    def __getitem__(self, idx):
        # image_path = self.dataset[idx]["image_path"]
        image = self.dataset[idx]["image"].convert("RGB")
        image = self.transform(image)

        # You might need to adjust the labels based on your task
        caption = self.dataset[idx]["alt_text"][0]


        return {"image": image, "caption": caption}