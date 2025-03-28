import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class  Conv1DModel(nn.Module):
    def __init__(self):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same')
        self.bn5 = nn.BatchNorm1d(256)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.cl_1 = nn.Linear(256, 128)
        self.cl_2 = nn.Linear(128, 4)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = F.pad(x, (0, 3), mode='constant', value=0)
        x = self.relu(x)
        x = self.bn1(x)    #192,96
        x = self.pool1(x)
        x = F.pad(x, (48, 48), mode='constant', value=0)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.pad(x, (48, 48), mode='constant', value=0)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = F.pad(x, (48, 48), mode='constant', value=0)
        x = self.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)
        x = F.pad(x, (48, 48), mode='constant', value=0)
        x = self.relu(self.conv5(x))
        x = self.bn5(x)
        x = self.pool5(x)
        x = F.pad(x, (48, 48), mode='constant', value=0)
        x = self.global_avg_pool(x).squeeze(-1)

        y1 = self.relu(self.cl_1(x))
        y2 = self.cl_2(y1)

        return y1, y2


# Initialize model






def class_margin(m, a, la):
    if len(la) == 0:
        return a
    else:
        index = torch.argmax(la[0], dim=-1)

    new_tensors = []
    for i in range(len(a)):
        c = a[i]
        part1 = c[:index]
        part2 = c[index + 1:]
        val = torch.tensor([c[index] - m], device=a.device)

        new_tensor = torch.cat([part1, val, part2], dim=0)
        new_tensors.append(new_tensor)

    return torch.stack(new_tensors)


def adamargin(data):
    pca = PCA(n_components=1, random_state=1000)
    data = pca.fit_transform(data).squeeze()

    s1 = Fitter(data[0:1000], distributions=['t', 'norm'])
    s2 = Fitter(data[1000:2000], distributions=['t', 'norm'])
    s3 = Fitter(data[2000:3000], distributions=['t', 'norm'])
    s4 = Fitter(data[3000:4000], distributions=['t', 'norm'])
    # s5 = Fitter(data[4000:5000], distributions=['t', 'norm'])
    # s6 = Fitter(data[5000:6000], distributions=['t', 'norm'])
    # s7 = Fitter(data[6000:7000], distributions=['t', 'norm'])

    s1.fit()
    s2.fit()
    s3.fit()
    s4.fit()
    # s5.fit()
    # s6.fit()
    # s7.fit()

    mean = [s1.fitted_param['t'][1], s2.fitted_param['t'][1], s3.fitted_param['t'][1],
            s4.fitted_param['t'][1],
            # s5.fitted_param['t'][1],
            # s6.fitted_param['t'][1],
            # s7.fitted_param['t'][1]
            ]
    sta = [s1.fitted_param['t'][2], s2.fitted_param['t'][2], s3.fitted_param['t'][2],
           s4.fitted_param['t'][2],
           # s5.fitted_param['t'][2],
           # s6.fitted_param['t'][2],
           # s7.fitted_param['t'][2]
           ]

    margin_list = []
    num = len(mean)
    for i in range(num):
        margin = []
        for j in range(num):
            if j != i:
                # condition = 6 * (sta[i] + sta[j]) - abs(mean[i] - mean[j])
                condition = 3 * (sta[i] + sta[j]) - abs(mean[i] - mean[j])   #类别-1？
                if condition < 0:
                    margin.append(0.)
                elif condition > 0:
                    margin.append(condition)
        margin_list.append(margin)
    print(margin_list)
    margin_list = np.max(np.array(margin_list), axis=1)
    return margin_list


def split(source_out1, source_output1, source_label):
    label_argmax = torch.argmax(source_label, dim=-1)
    # label_argmax = source_label
    # la, lb, lc, ld, le, lf, lg = [], [], [], [], [], [], []
    # a, b, c, d, e, f, g = [], [], [], [], [], [], []
    # da, db, dc, dd, de, df, dg = [], [], [], [], [], [], []
    la, lb, lc, ld = [], [], [], []
    a, b, c, d = [], [], [], []
    da, db, dc, dd = [], [], [], []

    for i in range(source_label.shape[0]):
        if label_argmax[i].item() == 0:
            a.append(source_output1[i])
            la.append(source_label[i])
            da.append(source_out1[i])
        elif label_argmax[i].item() == 1:
            b.append(source_output1[i])
            lb.append(source_label[i])
            db.append(source_out1[i])
        elif label_argmax[i].item() == 2:
            c.append(source_output1[i])
            lc.append(source_label[i])
            dc.append(source_out1[i])
        elif label_argmax[i].item() == 3:
            d.append(source_output1[i])
            ld.append(source_label[i])
            dd.append(source_out1[i])
        # elif label_argmax[i] == 4:
        #     e.append(source_output1[i])
        #     le.append(source_label[i])
        #     de.append(source_out1[i])
        # elif label_argmax[i] == 5:
        #     f.append(source_output1[i])
        #     lf.append(source_label[i])
        #     df.append(source_out1[i])
        # elif label_argmax[i] == 6:
        #     g.append(source_output1[i])
        #     lg.append(source_label[i])
        #     dg.append(source_out1[i])

    # return (a, b, c, d, e, f, g,
    #         torch.stack(la) if la else None,
    #         torch.stack(lb) if lb else None,
    #         torch.stack(lc) if lc else None,
    #         torch.stack(ld) if ld else None,
    #         torch.stack(le) if le else None,
    #         torch.stack(lf) if lf else None,
    #         torch.stack(lg) if lg else None
    #         )
    return (a, b, c, d,
            torch.stack(la) if la else None,
            torch.stack(lb) if lb else None,
            torch.stack(lc) if lc else None,
            torch.stack(ld) if ld else None
            )


def DG_Softmax(mar, source_out1, source_output1, source_label):
    # a, b, c, d, e, f, g, la, lb, lc, ld, le, lf, lg = split(source_out1, source_output1, source_label)
    # m, n, p, q, k, x, y = mar
    a, b, c, d, la, lb, lc, ld = split(source_out1, source_output1, source_label)
    m, n, p, q = mar

    a = class_margin(m, torch.stack(a), la) if len(a) > 0 else None
    b = class_margin(n, torch.stack(b), lb) if len(b) > 0 else None
    c = class_margin(p, torch.stack(c), lc) if len(c) > 0 else None
    d = class_margin(q, torch.stack(d), ld) if len(d) > 0 else None
    # e = class_margin(k, torch.stack(e), le) if len(e) > 0 else None
    # f = class_margin(x, torch.stack(f), lf) if len(f) > 0 else None
    # g = class_margin(y, torch.stack(g), lg) if len(g) > 0 else None

    data_set = []
    label_set = []

    if a is not None:
        data_set.append(a)
        label_set.append(la)
    if b is not None:
        data_set.append(b)
        label_set.append(lb)
    if c is not None:
        data_set.append(c)
        label_set.append(lc)
    if d is not None:
        data_set.append(d)
        label_set.append(ld)
    # if e is not None:
    #     data_set.append(e)
    #     label_set.append(le)
    # if f is not None:
    #     data_set.append(f)
    #     label_set.append(lf)
    # if g is not None:
    #     data_set.append(g)
    #     label_set.append(lg)
    if len(data_set) > 0:
        data = torch.cat(data_set, dim=0)
        label = torch.cat(label_set, dim=0)
        loss = loss_func(data.float(), label)
        return data, label, loss
    else:
        return None, None, torch.tensor(0.0)



def accuracy(y_true, y_pred):
    if len(y_true.shape) > 1:  # if one-hot encoded
        y_true = y_true.argmax(dim=1)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(dim=1)
    return (y_true == y_pred).float().mean().item()




def pre_train_step(source_data, target_data, source_label, target_label, model, optimizer, loss_func):
    """
    Single pre-training step
    """
    model.train()
    optimizer.zero_grad()
    source_label = source_label.float()
    target_label = target_label.float()
    # Forward pass
    _, output1 = model(source_data.float())
    _, output2 = model(target_data.float())


    # Calculate loss
    loss = loss_func(output1, torch.argmax(source_label, dim=-1))

    # Backward pass
    loss.backward()
    optimizer.step()

    # Calculate accuracies
    with torch.no_grad():
        train_acc = accuracy(source_label, output1)
        test_acc = accuracy(target_label, output2)

    return loss.item(), train_acc, test_acc


def pre_train(model, train_loader, optimizer, loss_func, num_epochs=100, device='cuda', ):
    """
    Pre-training function
    """
    model.to(device)


    # Tracking metrics
    history = {
        'train_acc': [],
        'test_acc': [],
        'loss': []
    }

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_train_acc = 0.0
        epoch_test_acc = 0.0
        batch_count = 0

        # Wrap loader with tqdm for progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for source_data, target_data, source_label, target_label in loop:
            # Move data to device
            source_data = source_data.to(device)
            target_data = target_data.to(device)
            source_label = source_label.to(device)
            target_label = target_label.to(device)

            # Training step
            batch_loss, batch_train_acc, batch_test_acc = pre_train_step(
                source_data, target_data, source_label, target_label,
                model, optimizer, loss_func
            )

            # Update metrics
            epoch_loss += batch_loss
            epoch_train_acc += batch_train_acc
            epoch_test_acc += batch_test_acc
            batch_count += 1

            # Update progress bar
            loop.set_postfix(loss=batch_loss, train_acc=batch_train_acc, test_acc=batch_test_acc)

        # Calculate epoch averages
        avg_loss = epoch_loss / batch_count
        avg_train_acc = epoch_train_acc / batch_count
        avg_test_acc = epoch_test_acc / batch_count

        # Store history
        history['loss'].append(avg_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_acc'].append(avg_test_acc)

        # Print epoch summary
        print(f'Epoch {epoch + 1}/{num_epochs}: '
              f'Loss: {avg_loss:.5f}, '
              f'Train Acc: {avg_train_acc:.5f}, '
              f'Test Acc: {avg_test_acc:.5f}')

    return model, history


# Helper function (same as before)


def train_step(mar, source_data, target_data, source_label, target_label, model, optimizer, loss_func):
    model.train()

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass
    out1, output1 = model(source_data.float())
    out2, output2 = model(target_data.float())

    # Calculate loss
    data, label, clc_loss_step = DG_Softmax(mar, out1, output1, source_label)

    if data is not None:
        clc_loss_step.backward()
        optimizer.step()

    # Calculate accuracies
    with torch.no_grad():
        train_acc = accuracy(label, data)
        test_acc = accuracy(target_label, output2.argmax(dim=1))

    return clc_loss_step.item(), train_acc, test_acc


# def accuracy(y_true, y_pred):
#     if len(y_true.shape) > 1:  # if one-hot encoded
#         y_true = y_true.argmax(dim=1)
#     if len(y_pred.shape) > 1:
#         y_pred = y_pred.argmax(dim=1)
#     return (y_true == y_pred).float().mean().item()


def train(model, train_loader, optimizer, loss_func, source_data1, num_epochs=50, device='cuda'):
    model.to(device)


    # Tracking metrics
    train_acc_history = []
    test_acc_history = []
    loss_history = []
    margin_history = []

    for epoch in range(num_epochs):
        # Update margin every epoch (changed from every 1 epoch in original)
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                _, feature = model(source_data1.float().to(device))
                mar = adamargin(feature.cpu().numpy())
                ini_margin = mar
        else:
            mar = ini_margin
        # if epoch == 6:
        #     a = 5

        margin_history.append(mar)
        print(f"Current margin: {mar}")

        epoch_loss = 0.
        epoch_train_acc = 0.
        epoch_test_acc = 0.
        batch_count = 0.

        for batch_idx, (source_data, target_data, source_label, target_label) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            # Move data to device
            source_data = source_data.to(device)
            target_data = target_data.to(device)
            source_label = source_label.to(device)
            target_label = target_label.to(device)

            # Train step
            batch_loss, batch_train_acc, batch_test_acc = train_step(
                mar, source_data, target_data, source_label, target_label,
                model, optimizer, loss_func
            )

            # Accumulate metrics
            epoch_loss += batch_loss
            epoch_train_acc += batch_train_acc
            epoch_test_acc += batch_test_acc
            batch_count += 1

        if batch_count > 0:
            # Calculate epoch metrics
            avg_loss = epoch_loss / batch_count
            avg_train_acc = epoch_train_acc / batch_count
            avg_test_acc = epoch_test_acc / batch_count

            # Store history
            loss_history.append(avg_loss)
            train_acc_history.append(avg_train_acc)
            test_acc_history.append(avg_test_acc)


        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}, '
              f'Train Acc: {avg_train_acc:.5f}, Test Acc: {avg_test_acc:.5f}')

    # return {
    #     'model': model,
    #     'train_acc': train_acc_history,
    #     'test_acc': test_acc_history,
    #     'loss': loss_history,
    #     'margin': margin_history
    # }
if __name__ == "__main__":
    import torch
    from torch.utils.data import Dataset
    import numpy as np
    import logging
    logging.disable(30)
    from sklearn.decomposition import PCA
    from fitter import Fitter
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    num_classes = 4
    data0 = np.load('jnu_600_data.npy')
    label0 = np.eye(num_classes)[np.load('jnu_600_label.npy')]
    data1 = np.load('jnu_800_data.npy')
    label1 = np.eye(num_classes)[np.load('jnu_800_label.npy')]
    data2 = np.load('jnu_1000_data.npy')
    label2 = np.eye(num_classes)[np.load('jnu_1000_label.npy')]

    source_data = data0
    source_label = label0
    target_data = data2
    target_label = label2

    print(source_data.shape, source_label.shape, target_data.shape, target_label.shape)

    # 归一化处理
    source_mean = source_data.mean(axis=1).reshape((-1, 1))  # 计算每行的均值
    source_std = source_data.std(axis=1).reshape((-1, 1))  # 计算每行的标准差
    source_data = (source_data - source_mean) / source_std  # 标准化

    # 标准化 target_data
    target_mean = target_data.mean(axis=1).reshape((-1, 1))  # 计算每行的均值
    target_std = target_data.std(axis=1).reshape((-1, 1))  # 计算每行的标准差
    target_data = (target_data - target_mean) / target_std

    # 增加维度 [batch_size, length] -> [batch_size, 1, length]
    source_data = torch.from_numpy(source_data).unsqueeze(1).float()
    target_data = torch.from_numpy(target_data).unsqueeze(1).float()
    source = source_data
    source_label = torch.from_numpy(source_label).float()
    target_label = torch.from_numpy(target_label).float()
    class CustomDataset(Dataset):
        def __init__(self, source_data, target_data, source_label, target_label):
            self.source_data = source_data
            self.target_data = target_data
            self.source_label = source_label
            self.target_label = target_label

        def __len__(self):
            return len(self.source_data)

        def __getitem__(self, idx):
            return self.source_data[idx], self.target_data[idx], self.source_label[idx], self.target_label[idx]

    train_dataset = CustomDataset(source_data, target_data, source_label, target_label)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
    test_loader = DataLoader(train_dataset, batch_size=4000, shuffle=False, drop_last=True)
    model = Conv1DModel()
    pre_optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    pre_train(model, train_loader, optimizer=pre_optimizer, loss_func=loss_func, num_epochs=100, device='cuda')
    train(model, train_loader, optimizer=optimizer, loss_func=loss_func, source_data1=source, num_epochs=50, device='cuda')
