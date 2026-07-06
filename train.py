import torch
import torch.nn as nn
from dataset import DatasetAnomaly, class_dict
import os
import numpy as np
from torch.utils.data import DataLoader
from model import SSFGNN


def main():
    # 1. load dataset
    train_path = "E:/datasets/SEN2MHD/train"
    test_path = "E:/datasets/SEN2MHD/test"
    num_class = 7
    binary = True if num_class == 2 else False
    batch_size = 64
    train_dataset = DatasetAnomaly(train_path, transform=True, load_in_memory=False, use_indices=True, binary_classify=binary)
    test_dataset = DatasetAnomaly(test_path, transform=False, load_in_memory=False, use_indices=True, binary_classify=binary)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False)
    valid_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 2. load model
    model = SSFGNN(num_class,5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 3. prepare super parameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    epoch = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. train
    val_acc_list = []
    out_dir = "checkpoints_new/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for epoch in range(0, epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        total_samples = 0
        confusion = torch.zeros(num_class, num_class, dtype=torch.int64, device=device)

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)  # [B, C, H, W] 或 [B, C]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # loss
            sum_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # GPU prediction
            predicted = outputs.argmax(1)

            # Confusion matrix (GPU)
            labels_flat = labels.view(-1)
            preds_flat = predicted.view(-1)
            inds = num_class * labels_flat + preds_flat
            cm = torch.bincount(inds, minlength=num_class ** 2).reshape(num_class, num_class)
            confusion += cm
        # Calculate evaluation metrics
        confusion_np = confusion.cpu().numpy()
        iou = np.diag(confusion_np) / (confusion_np.sum(1) + confusion_np.sum(0) - np.diag(confusion_np))
        miou = np.nanmean(iou)
        oa = np.trace(confusion_np) / confusion_np.sum()
        avg_loss = sum_loss / total_samples
        print('[epoch:%d, iter:%d] Loss: %.03f | mIoU: %.3f%% | OA: %.3f%%'
              % (epoch + 1, batch_idx, avg_loss, 100. * miou, 100. * oa))
        print(class_dict)
        print("IoU:", np.round(iou * 100, 3))

        # scheduler.step()


        # get the ac with testdataset in each epoch
        print('Waiting Val...')
        model.eval()
        confusion = torch.zeros(num_class, num_class, dtype=torch.int64, device=device)
        sum_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_dataloader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                # loss = criterion(outputs, labels)
                loss = criterion(outputs, labels)
                sum_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # GPU prediction
                predicted = outputs.argmax(1)

                # Confusion matrix (GPU)
                labels_flat = labels.view(-1)
                preds_flat = predicted.view(-1)
                inds = num_class * labels_flat + preds_flat
                cm = torch.bincount(inds, minlength=num_class ** 2).reshape(num_class, num_class)
                confusion += cm

        # Calculate evaluation metrics
        confusion_np = confusion.cpu().numpy()
        iou = np.diag(confusion_np) / (confusion_np.sum(1) + confusion_np.sum(0) - np.diag(confusion_np))
        miou = np.nanmean(iou)
        oa = np.trace(confusion_np) / confusion_np.sum()
        avg_loss = sum_loss / total_samples
        print('[epoch:%d, iter:%d] Loss: %.03f | mIoU: %.3f%% | OA: %.3f%%'
              % (epoch + 1, batch_idx, avg_loss, 100. * miou, 100. * oa))
        print(class_dict)
        print("IoU:", np.round(iou * 100, 3))

        val_acc_list.append(oa)

        if oa == max(val_acc_list):
            torch.save(model.state_dict(), out_dir + "best.pt")
            print("save epoch {} model".format(epoch + 1))

        torch.save(model.state_dict(), out_dir + "last.pt")


    print("Final highest accuracy: {}".format(max(val_acc_list)))


if __name__ == "__main__":
    main()