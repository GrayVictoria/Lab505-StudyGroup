import os
import torch
import torch.optim
from torch import nn
from models.ClassifierNet import Net
from data_loader import build_datasets
import args_parser
import numpy as np

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)

def calc_loss(outputs, labels, f1, f2, f3):

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)

    criterion = nn.TripletMarginLoss()
    f1 = f1.view(-1, args.patch_size * args.patch_size)
    f2 = f2.view(-1, args.patch_size * args.patch_size)
    f3 = f3.view(-1, args.patch_size * args.patch_size)

    # d(f1,f2)=d(f1,f3)
    l1 = criterion(f1, f2, f3)
    l2 = criterion(f1, f3, f2)

    #d(f2,f1)=d(f2,f3)
    l3 = criterion(f2, f1, f3)
    l4 = criterion(f2, f3, f1)

    triplet_loss = l1 + l2 + l3 + l4
    return loss + triplet_loss

def L1_penalty(var):
    return torch.abs(var).sum()

def train(model, device, train_loader, optimizer, epoch, slim_params):
    model.train()
    total_loss = 0
    for i, (inputs_1, inputs_2, inputs_3, labels) in enumerate(train_loader):
        inputs_1, inputs_2, inputs_3 = inputs_1.to(device), inputs_2.to(device), inputs_3.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, f1, f2, f3 = model(inputs_1, inputs_2, inputs_3)
        #print(f1.shape)
        loss = calc_loss(outputs, labels, f1, f2, f3)
        L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
        loss += 0.05 * L1_norm

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Epoch %d' % epoch, flush=True, end= " ")
    print('  [loss avg: %.4f]   [current loss: %.4f]' %( total_loss/(epoch+1), loss.item()))

def test(model, device, test_loader):
    model.eval()
    count = 0
    for inputs_1, inputs_2, inputs_3, labels in test_loader:
        inputs_1, inputs_2, inputs_3 = inputs_1.to(device), inputs_2.to(device), inputs_3.to(device)
        outputs, _, _, _ = model(inputs_1, inputs_2, inputs_3)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            test_labels = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            test_labels = np.concatenate((test_labels, labels))
    a = 0
    for c in range(len(y_pred_test)):
        if test_labels[c]==y_pred_test[c]:
            a = a + 1
    acc = a/len(y_pred_test)*100
    print (' [The verification accuracy is: %.2f]' %(acc))
    return acc


def main():

    train_loader, test_loader = build_datasets(args.root, args.dataset, args.patch_size, args.batch_size, args.test_ratio)

    if args.dataset == 'Berlin':
        args.hsi_bands = 244
        args.sar_bands = 4
        args.num_class = 8
    elif args.dataset == 'Augsburg':
        args.hsi_bands = 180
        args.sar_bands = 4
        args.dsm_bands = 1
        args.num_class = 7
    elif args.dataset == 'HHK':
        args.hsi_bands = 166
        args.msi_bands = 8
        args.sar_bands = 3
        args.num_class = 5


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(args.hsi_bands,  args.sar_bands, args.msi_bands, args.hidden_size, args.patch_size, args.num_class).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    Net_params, slim_params = [], []

    for name, param in model.named_parameters():  
        if param.requires_grad and name.endswith('weight') and 'bn2' in name:
            if len(slim_params) % 2 == 0:
                slim_params.append(param[:len(param) // 2])
            else:
                slim_params.append(param[len(param) // 2:])
            
    best_acc = 0
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch, slim_params)
        if (epoch+1)%2 == 0:
            acc = test(model, device, test_loader)
            if acc >= best_acc:
                  best_acc = acc
                  print("save model")
                  torch.save(model.state_dict(),'./checkpoints/model.pth')


if __name__ == '__main__':
    main()
