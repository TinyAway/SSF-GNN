import torch
import torch.nn as nn

from dataset import DatasetAnomaly, class_dict

import numpy as np
from torch.utils.data import DataLoader
from model import SSFGNN



if __name__ == "__main__":

    num_class = 7
    binary = True if num_class == 2 else False
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    test_path = "E:/datasets/anomaly_dataset_new/test"
    test_dataset = DatasetAnomaly(test_path, transform=False, load_in_memory=False, use_indices=True, binary_classify=binary)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = SSFGNN(num_class)
    model.load_state_dict(torch.load('checkpoints_new/D5_3.pt'))
    model.to(device)
    model.eval()

    confusion = torch.zeros(num_class, num_class, dtype=torch.int64, device=device)
    sum_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # torch.size([batch_size, num_class])

            predicted = outputs.argmax(1)

            # Confusion metrix (GPU)
            labels_flat = labels.view(-1)
            preds_flat = predicted.view(-1)
            inds = num_class * labels_flat + preds_flat
            cm = torch.bincount(inds, minlength=num_class ** 2).reshape(num_class, num_class)
            confusion += cm

    confusion = confusion.cpu().numpy()

    print('confusion matrix', confusion)

    precision = np.diag(confusion) / np.sum(confusion, axis=0)
    print('precision', precision)
    recall = np.diag(confusion) / np.sum(confusion, axis=1)
    print('recall', recall)
    oa = np.sum(np.diag(confusion)) / np.sum(confusion)
    print('oa', oa)
    F1 =  2 * precision * recall / (precision + recall)
    print('F1 Score', F1)
    F2 = 5 * precision * recall / (4 * precision + recall)
    print('F2 Score', F2)


