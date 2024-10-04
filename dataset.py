import torch
from torch.utils.data import Dataset

import numpy as np
import h5py

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

class ShapeNetCore(Dataset):  # for single category dataset

    def __init__(self, path, cate, split="train"):
        super().__init__()

        assert split in ["train", "val", "test"]
        synsetid = cate_to_synsetid[cate]
        self.pcds = []
        with h5py.File(path, 'r') as f:
            tgt_datas = f[synsetid][split]
            for pcd in tgt_datas:
                pcd = torch.from_numpy(pcd)
                # shift와 scale은 실질적으로 reshape 할 필요가 없음.
                # 어차피 Broadcast되기 때문에. 단, 명확히 하기위해 일일히 지정한 듯.
                shift = pcd.mean(dim=0).reshape(1, 3)
                scale = pcd.flatten().std().reshape(1, 1)

                pcd = (pcd - shift) / scale

                self.pcds.append(pcd)

    def __len__(self):
        return len(self.pcds)

    def __getitem__(self, idx):
        return self.pcds[idx]


if __name__ == '__main__':
    SEED = 1234
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    path = './data/shapenet.hdf5'
    cate = "airplane"
    split = "train"  # test, train, val

    dataset = ShapeNetCore(path, cate)