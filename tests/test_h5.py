import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
# from scripts.preprocessing.pigs.bpe_calculations import calculate_bpe_mask

path_data = "./dummy_data/side_v3/data_compressed.h5"
path_masks = "./dummy_data/side_v3/masks_compressed.h5"

with h5py.File(path_data, 'r') as f:
    pid = 'Breast_MRI_001_left'
    # print(f[pid].keys())
    
    pre = f[pid]['pre']
    # print(f'pre.shape: {pre.shape}')
    # pre_affine = f[pid]['pre_affine']
    # print(f'pre_affine.shape: {pre_affine.shape}')
    post_1 = f[pid]['post_1']
    # print(f'pre.shape: {pre.shape}, post_1.shape: {post_1.shape}')
    # post_1_affine = f[pid]['post_1_affine']
    # print(f'post_1_affine.shape: {post_1_affine.shape}')
    post_2 = f[pid]['post_2']
    # print(f'post_2.shape: {post_2.shape}')
    post_3 = f[pid]['post_3']
    # print(f'post_3.shape: {post_3.shape}')
    post_4 = f[pid]['post_4']
    # print(f'post_4.shape: {post_4.shape}')
    t1 = f[pid]['T1']
    # print(f't1.shape: {t1.shape}')
    t1_resampled = f[pid]['T1_resampled']
    # print(f't1_resampled.shape: {t1_resampled.shape}')
    sub = f[pid]['sub']
    # print(f'sub.shape: {sub.shape}')

    slice_idx = pre.shape[3] // 4

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 4, 1)
    plt.imshow(pre[0, :, :, slice_idx], cmap='gray')
    plt.title('Pre')
    plt.subplot(2, 4, 2)
    plt.imshow(t1[0, :, :, slice_idx], cmap='gray')
    plt.title('T1')
    plt.subplot(2, 4, 3)
    plt.imshow(t1_resampled[0, :, :, slice_idx], cmap='gray')
    plt.title('T1 Resampled')
    plt.subplot(2, 4, 4)
    plt.imshow(sub[0, :, :, slice_idx], cmap='gray')
    plt.title('Sub')
    plt.subplot(2, 4, 5)
    plt.imshow(post_1[0, :, :, slice_idx], cmap='gray')
    plt.title('Post 1')
    plt.subplot(2, 4, 6)
    plt.imshow(post_2[0, :, :, slice_idx], cmap='gray')
    plt.title('Post 2')
    plt.subplot(2, 4, 7)
    plt.imshow(post_3[0, :, :, slice_idx], cmap='gray')
    plt.title('Post 3')
    plt.subplot(2, 4, 8)
    plt.imshow(post_4[0, :, :, slice_idx], cmap='gray')
    plt.title('Post 4')
    plt.tight_layout()
    plt.show()


with h5py.File(path_masks, 'r') as f:
    # print(f['Breast_MRI_001'].keys())
    breast = f['Breast_MRI_001']['breast']
    fgt = f['Breast_MRI_001']['fgt']

    print(f'breast.shape: {breast.shape}, fgt.shape: {fgt.shape}')

    fgt_mask = np.argmax(fgt, axis=0)
    fgt_mask = (fgt_mask == 2).astype(np.uint8)
    
    # Initial slice index
    slice_index = breast.shape[2] // 2

    # Create figure and two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    img1 = axes[0].imshow(breast[:, :, slice_index], cmap='gray')
    axes[0].set_title('Breast Mask')
    img2 = axes[1].imshow(fgt_mask[:, :, slice_index], cmap='gray')
    axes[1].set_title('FGT Mask')
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Slider axis
    ax_slider = plt.axes([0.2, 0.01, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Slice Index',
        valmin=0,
        valmax=breast.shape[2] - 1,
        valinit=slice_index,
        valfmt='%d',
        valstep=1
    )

    def update(val):
        sl = int(slider.val)
        img1.set_data(breast[:, :, sl])
        img2.set_data(fgt_mask[:, :, sl])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

    # bpe_mask = calculate_bpe_mask(breast, fgt_mask)

