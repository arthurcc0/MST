''' 
Code to open the data_compressed.h5 file and check the shape of the data and mask.
Load one sample and check the keys associated to it. 
'''

import h5py
import matplotlib.pyplot as plt


with h5py.File('dummy_data/side_noN4BC/data_compressed.h5', 'r') as f:
    print('Data_noN4BC (preprocessed_crop-a):')
    # print(f.keys())
    num_patients = len(f.keys())
    print(f"Number of entries: {num_patients}")
    print(f['Breast_MRI_001_left'].keys())
    sub = f['Breast_MRI_001_left']['sub'][()]
    print(sub.shape)
    print(sub.max())
    print(sub.min())
    print(sub.mean())

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(sub[0, :, :, 16], cmap='gray')
    axes[0].set_title('Subtraction')

with h5py.File('dummy_data/side_v3/data_compressed.h5', 'r') as f:
    print('Data_v3:')
    print(f['Breast_MRI_001_left'].keys())
    sub = f['Breast_MRI_001_left']['sub'][()]
    print(sub.shape)
    print(sub.max())
    print(sub.min())
    print(sub.mean())

with h5py.File('dummy_data/side_v3/masks_compressed.h5', 'r') as f:
    print('Masks:')
    # print(f.keys())
    num_patients = len(f.keys())
    print(f"Number of entries: {num_patients}")
    print(f['Breast_MRI_001'].keys())
    breast_mask = f['Breast_MRI_001']['breast'][()]
    print(breast_mask.shape)
    axes[1].imshow(breast_mask[:, :,80], cmap='gray')
    plt.show()