from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def download_dataset():
    download_root = 'MNIST_data/' # 데이터 다운로드 경로

    train_dataset = datasets.MNIST(root=download_root,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True) # 학습 dataset 정의
                            
    test_dataset = datasets.MNIST(root=download_root,
                            train=False,
                            transform=transforms.ToTensor(), 
                            download=True) # 평가 dataset 정의

    batch_size = 100 # 배치 사이즈 정의. 데이터셋을 잘개 쪼개서 묶음으로 만드는 데 기여한다.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 학습 데이터셋을 배치 사이즈 크기만큼씩 잘라서 묶음으로 만든다. 묶음의 개수는 train_dataset / batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) # train_dataloader와 마찬가지

    print(len(train_loader),len(test_loader))

    return train_loader,test_loader

if __name__ == '__main__':
    download_dataset()
