import os, pickle
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from ..utils.data_prep import get_fractured

class CustomLoader(data.Dataset):
    def __init__(self, file, out_folder, subset='train', fracture=True, frac_ratio=None, **kwargs):
        self.file = file
        self.out_folder = out_folder
        self.subset = subset
        self.data = np.load(file, allow_pickle=True).item()[self.subset]
        print("Loaded {} data with {} samples".format(self.subset, len(self.data["labels"])))
        # for k in self.data.keys():
        #     self.data[k] = np.repeat(
        #         self.data[k], 
        #         20, 
        #         axis=0,
        #     )
        #     print("Inflated data to: " + str(self.data[k].shape))
        self.load_labels()
        self.fracture_opts = kwargs
        self.fracture = fracture
        self.frac_ratio = frac_ratio
        
    def load_labels(self):
        le_file = os.path.join(self.out_folder, 'label_encoder.pkl')
        if os.path.exists(le_file):
            le = pickle.load(open(le_file, 'rb'))
            if len(le.classes_) == 1:
                assert len(le.classes_) == 1
                new_labels = list(set(self.data['labels']))
                assert len(new_labels) == 1
                if new_labels[0] not in le.classes_:
                    self.data['labels'] = [le.classes_[0] for _ in self.data['labels']]
                    print("Warning: Previously unseen label detected, converting {} to {}".format(
                        new_labels[0], le.classes_[0]
                    ))
            labels = le.transform(self.data['labels'])
        else:
            le = LabelEncoder()
            labels = le.fit_transform(self.data['labels'])
            # assert len(le.classes_) == 1
            pickle.dump(le, open(le_file, 'wb'))
            
        self.le = le
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index]
        shape_target = self.data['data'][index].astype(np.float32)
        if self.fracture:
            if self.frac_ratio is not None:
                # Choose if to pick from pre-fractured or from automatically fractured
                if np.random.random(1) > self.frac_ratio:
                    shape_source = self.data['data_broken'][index].astype(np.float32)
                else:
                    shape_source = get_fractured(shape_target, **self.fracture_opts)
            else:
                shape_source = get_fractured(shape_target, **self.fracture_opts)
        else:
            shape_source = shape_target

        # from [0, 1] space to [-1, 1]
        idxs = np.argwhere(shape_source == 0)
        shape_source[idxs[:,0], idxs[:,1], idxs[:,2]] = -1
        idxs = np.argwhere(shape_target == 0)
        shape_target[idxs[:,0], idxs[:,1], idxs[:,2]] = -1
        
        return shape_source, shape_target, label

    def __len__(self):
        return len(self.data['labels'])
