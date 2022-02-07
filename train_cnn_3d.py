import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import torch
import torch.nn as nn
import torch.optim as optim

from data import Betondataset, plot_batch
from paths import *

plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH


"""
Train a 3D CNN for Image Classification
"""


def train_net(net, train, test, load="", checkpoints=True, num_epochs=5):
    """
    Train a Classifier

    :param load: if not "", load Classifier from here and continue training
    :param checkpoints: if state of the net should be saved after each epoch
    :param num_epochs: number of epochs to train
    """

    # Seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load net
    if load != '':
        try:
            net.load_state_dict(torch.load(load))
        except FileNotFoundError:
            print("No parameters loaded")
            pass

    # Loss / Optimizer
    # todo: use CrossEntropyLoss and extra category (e.g. unsure / nothing)
    # pos/all is 2 for semisynth, so basically pos_weight *= 2
    pos_weight = 0.75
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    lr = 0.0001
    weight_decay = 0.1
    beta1 = 0.9
    optimizer = optim.Adam(net.parameters(), betas=(beta1, 0.999), lr=lr, weight_decay=weight_decay)

    # Animate
    fig, anim_ax = plt.subplots(figsize=(8, 5))
    Writer = FFMpegWriter(fps=1)
    Writer.setup(fig, "cnn_progress.mp4", dpi=100)

    # Training loop
    losses = []
    iters = 0
    loss_mean = 0

    print("Starting Training Loop...")
    for epoch in range(1, num_epochs + 1):
        for i, data in enumerate(train, 0):
            inputs, labels = data["X"].to(device), data["y"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Output training stats
            loss_mean += loss.item()
            losses.append(loss.item())
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                      % (epoch, num_epochs, i, len(train), loss_mean))
                loss_mean = 0

            iters += 1

        if checkpoints:
            # do checkpointing
            torch.save(net.state_dict(), '%s_epoch_%d' % (load or "netG", epoch))

        metrics(net, test, plot=epoch == num_epochs, anim=(Writer, anim_ax), criterion=criterion)
        # metrics(net, test, plot=True, criterion=criterion)

    Writer.finish()

    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.savefig("prog.png")
    plt.show()


def metrics(net, testloader, plot=True, anim=None, criterion=None):
    """
    Compute accuracy, recall, precision etc of a net on a test set

    :param anim: (Writer, ax): plot on ax and grab frame.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cutoff = 0.5
    pos_out = []

    neg_out = []
    tp, tn, fp, fn = 0, 0, 0, 0

    # Find 8 samples of FP and FN
    examples_fp = torch.zeros([8, 1, 100, 100, 100])
    examples_fn = torch.zeros([8, 1, 100, 100, 100])
    fp_idx = np.zeros([8])
    fn_idx = np.zeros([8])
    fp_n = 0
    fn_n = 0
    loss = 0

    dur = 0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch["X"].to(device), (batch["y"] > cutoff).float().view(-1)
            start = time.time()
            outputs = torch.sigmoid(net(inputs))
            predicted = (outputs > cutoff).float().view(-1)
            dur += time.time() - start

            if criterion is not None:
                loss += 1 / len(testloader) * criterion(outputs.view(-1), labels).mean().item()

            predicted = predicted.cpu()
            pos_out += outputs[labels == 1].view(-1).tolist()
            neg_out += outputs[labels == 0].view(-1).tolist()
            tp += ((predicted == 1.0) & (labels == 1.0)).sum().item()
            tn += ((predicted == 0.0) & (labels == 0.0)).sum().item()
            fp += ((predicted == 1.0) & (labels == 0.0)).sum().item()
            fn += ((predicted == 0.0) & (labels == 1.0)).sum().item()

            if plot:
                fp_ths = ((predicted == 1.0) & (labels == 0.0))
                fn_ths = ((predicted == 0.0) & (labels == 1.0))
                if fp_n < 8 and fp_ths.sum().item() > 0:
                    examples_fp[fp_n:min(8, fp_n + fp_ths.sum().item())] = inputs[fp_ths][0:8-fp_n]
                    fp_idx[fp_n:min(8, fp_n + fp_ths.sum().item())] = batch["id"][fp_ths][0:8-fp_n]
                    fp_n = min(8, fp_n + fp_ths.sum().item())
                if fn_n < 8 and ((predicted == 0.0) & (labels == 1.0)).sum().item() > 0:
                    examples_fn[fn_n:min(8, fn_n + fn_ths.sum().item())] = inputs[fn_ths][0:8-fn_n]
                    fn_idx[fn_n:min(8, fn_n + fn_ths.sum().item())] = batch["id"][fn_ths][0:8-fn_n]
                    fn_n = min(8, fn_n + fn_ths.sum().item())
    net.train()
    print("Time: %g s (%g s / sample - %.1f s / 2k)" %
          (dur, dur/len(testloader), dur/len(testloader)*(2000/100 * 1.5)**3))

    total = tp + tn + fp + fn
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    tnr = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = 100 * (tp + tn) / total
    print('On the %d test images\nTP|TN|FP|FN: %d %d %d %d\n'
          'Accuracy: %.2f %%\nPrecision: %.2f %%\nRecall: %.2f %%\nTNR: %.2f %%' %
          (total, tp, tn, fp, fn, acc, precision, recall, tnr))
    if criterion is not None:
        print("Loss: %.2f" % loss)

    if anim is not None:
        bins = np.linspace(0, 1, 40)
        Writer, ax = anim
        ax.clear()
        ax.hist(pos_out, bins, alpha=0.5, label="pos")
        ax.hist(neg_out, bins, alpha=0.5, label="neg")
        ax.legend()
        Writer.grab_frame()

    if plot:
        print(fp_idx, fn_idx)
        plot_batch(examples_fn)
        plot_batch(examples_fp)
        plt.show()

        plt.figure()
        bins = np.linspace(0, 1, 40)
        plt.hist(pos_out, bins, alpha=0.5, label="pos")
        plt.hist(neg_out, bins, alpha=0.5, label="neg")
        plt.legend()
        plt.show()


def analyze_net(net, testloader, path, n=100, p=[0, 0.25, 0.5, 0.75, 1]):
    """
    Evaluate a net on a test set an plot groups of data
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(path, map_location=device))

    # Get samples and their output
    samples = []
    dur = 0
    c = 0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            inputs = batch["X"].to(device)
            ids = batch["id"]
            start = time.time()
            out_pre = net(inputs).cpu().view(-1)
            out = torch.sigmoid(out_pre)
            dur += time.time() - start

            for i in range(len(out)):
                c += 1
                labels = batch.get("y", len(batch["X"]) * [None])
                samples += [(out[i], inputs[i], {"id": ids[i], "label": labels[i]}, out_pre[i])]

            if c >= n:
                break
    net.train()
    print("Time: %g s (%g s / sample - %.1f s / 2k)" %
          (dur, dur / c, dur / c * (2000 / 100 * 1.5) ** 3))

    # Sort by output
    samples.sort(key=lambda x: x[0])

    # plot batches of data
    # for tp, tn, fp, fn
    cutoff = 0.5
    tp = [s[1] for s in samples if s[2]["label"] == 1.0 and s[0] > cutoff]
    tn = [s[1] for s in samples if s[2]["label"] == 0.0 and s[0] < cutoff]
    fp = [s[1] for s in samples if s[2]["label"] == 0.0 and s[0] > cutoff]
    fn = [s[1] for s in samples if s[2]["label"] == 1.0 and s[0] < cutoff]
    if len(tp) > 0:
        plot_batch(torch.stack(tp[-8:]), title="True positives")
    if len(tn) > 0:
        plot_batch(torch.stack(tn[:8]), title="True negatives")
    if len(fp) > 0:
        plot_batch(torch.stack(fp[-8:]), title="False positives")
    if len(fn) > 0:
        plot_batch(torch.stack(fn[:8]), title="False negatives")

    out_all = torch.tensor([s[3] for s in samples])
    out_none = torch.tensor([s[3] for s in samples if s[2]["label"] is None])
    out_pos = torch.tensor([s[3] for s in samples if s[2]["label"] == 1.0])
    out_neg = torch.tensor([s[3] for s in samples if s[2]["label"] == 0.0])

    # for out_log close to p
    for p_ in p:
        pos = sorted(sorted(out_all, key=lambda x: abs(torch.sigmoid(x) - p_))[:8])
        plot_batch(torch.stack([s[1] for s in samples if s[3] in pos][:8]),
                   title=r"p $\in$ [%.2f, %.2f]" % (torch.sigmoid(min(pos)), torch.sigmoid(max(pos))))

    # histograms
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    bins = np.linspace(0, 1, 40)
    ax[0].hist(np.array(torch.sigmoid(out_none)), bins, alpha=0.5, label="none")
    ax[0].hist(np.array(torch.sigmoid(out_pos)), bins, alpha=0.5, label="pos")
    ax[0].hist(np.array(torch.sigmoid(out_neg)), bins, alpha=0.5, label="neg")
    ax[0].set_title("Crack probability")

    bins = np.linspace(out_all.min(), out_all.max(), 40)
    ax[1].hist(np.array(out_none), bins, alpha=0.5, label="none")
    ax[1].hist(np.array(out_pos), bins, alpha=0.5, label="pos")
    ax[1].hist(np.array(out_neg), bins, alpha=0.5, label="neg")
    ax[1].set_title("Net output")
    ax[1].legend()
    plt.show()


def animate_dataset(net, testloader, path, n=100):
    """
    Loop through a dataset sorted by predicted crack probability
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(path, map_location=device))

    n = min(n, len(testloader))

    # Get samples and their output
    samples = []
    dur = 0
    c = 0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            inputs = batch["X"].to(device)
            ids = batch["id"]
            start = time.time()
            out = torch.sigmoid(net(inputs)).cpu().view(-1)
            dur += time.time() - start

            for i in range(len(out)):
                c += 1
                labels = batch.get("y", len(batch["X"]) * [None])
                samples += [(out[i], inputs[i].cpu(), {"id": ids[i], "label": labels[i]})]

            if c >= n:
                break
    net.train()
    print("Time: %g s (%g s / sample - %.1f s / 2k)" %
          (dur, dur / c, dur / c * (2000 / 100 * 1.5) ** 3))

    # Sort by output
    samples.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    Writer = FFMpegWriter(fps=5)
    Writer.setup(fig, "dataset_sorted.mp4", dpi=100)

    for p in range(0, n, 8):
        ax.clear()
        plot_batch(torch.stack([s[1] for s in samples[p: p+8]]),
                   ax=ax, title=r"p $\in$ [%.2f, %.2f]" % (samples[p][0], samples[min(n-1, p+8)][0]))
        Writer.grab_frame()

    Writer.finish()


def inspect_net(net, test, path):
    net.load_state_dict(torch.load(path, map_location=device))
    metrics(net, test, plot=True)


if __name__ == "__main__":
    from legacy import LegNet1
    torch.cuda.empty_cache()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    train, val = Betondataset("semisynth-inf", binary_labels=True, confidence=0.9,
                              batch_size=4, shuffle=True, num_workers=1)
    # test = Betondataset("nc-val", test=0, batch_size=4, norm=(0, 1))
    test = Betondataset("nc", test=0, batch_size=4, norm=(0, 255))

    # Net
    net = LegNet1(layers=1).to(device)

    load = "checkpoints/shift_0_11/netcnn_l1p_epoch_5.cp"
    # train_net(net, train, val, load, checkpoints=True, num_epochs=10)
    # inspect_net(net, test, load)
    analyze_net(net, test, load)
    # animate_dataset(net, test, load, n=1000)