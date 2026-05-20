import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp



class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, his_len, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)

        # self.data = data

        # 存储 memmap 相关信息
        self.is_memmap = hasattr(data, 'filename')
        if self.is_memmap:
            self.data = data
            logger.info('Using memmap for data loading')
        else:
            self.data = data

        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))

        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon
        self.his_len = his_len
        self.his_offsets = np.arange(-(self.his_len - 1), 1, 1)
        self.his_mask = np.zeros((self.his_len, self.data.shape[1], self.data.shape[2]))

    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                batch_size = len(idx_ind)

                # 直接创建 numpy 数组，避免共享内存的复杂性
                x = np.zeros((batch_size, self.seq_len, self.data.shape[1], self.data.shape[-1]), dtype=np.float32)
                y = np.zeros((batch_size, self.horizon, self.data.shape[1], 1), dtype=np.float32)
                his = np.zeros((batch_size, self.his_len, self.data.shape[1], self.data.shape[-1]), dtype=np.float32)

                # 单线程加载数据（memmap 友好）
                for i in range(batch_size):
                    idx_val = idx_ind[i]

                    # 使用 memmap 按需加载
                    x[i] = self.data[idx_val + self.x_offsets, :, :]
                    y[i] = self.data[idx_val + self.y_offsets, :, :1]

                    if idx_val - self.his_len < 0:
                        his[i] = self.his_mask
                    else:
                        his[i] = self.data[idx_val + self.his_offsets, :, :]

                yield (x, y, his)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# def load_dataset(data_path, args, logger):
#     ptr = np.load(os.path.join(data_path, args.years, 'his.npz'))
#     logger.info('Data shape: ' + str(ptr['data'].shape))
#
#     dataloader = {}
#     for cat in ['train', 'val', 'test']:
#         idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
#         dataloader[cat + '_loader'] = DataLoader(ptr['data'][..., :args.input_dim], idx,
#                                                  args.seq_len, args.horizon, args.his_len, args.bs, logger)
#
#     scaler = StandardScaler(mean=ptr['mean'], std=ptr['std'])
#     return dataloader, scaler

# 1. 修改 load_dataset 函数：用 np.memmap 替代 np.load 加载大文件
def  load_dataset(data_path, args, logger):
    # 读取npz文件的元数据（不加载数据）
    with np.load(os.path.join(data_path, args.years, 'his.npz'), mmap_mode='r') as ptr:
        data_shape = ptr['data'].shape
        data_dtype = ptr['data'].dtype
        mean = ptr['mean']
        std = ptr['std']
    logger.info('Data shape: ' + str(data_shape))

    # 用内存映射加载数据（仅占用元数据内存，数据存在磁盘）
    data_path_full = os.path.join(data_path, args.years, 'his_data_mmap.npy')
    # 若未生成mmap文件，先从npz生成（仅执行一次）
    if not os.path.exists(data_path_full):
        with np.load(os.path.join(data_path, args.years, 'his.npz')) as ptr:
            data = ptr['data'][..., :args.input_dim]
            np.save(data_path_full, data)  # 保存为npy格式（支持memmap）

    # # 用memmap加载：mode='r'表示只读，不加载全量数据
    # data_mmap = np.memmap(data_path_full, dtype=data_dtype, mode='r', shape=data_shape[:-1] + (args.input_dim,))

    # 自动读取文件的真实形状，避免手动构造形状的风险
    data_mmap = np.lib.format.open_memmap(
        data_path_full,
        dtype=data_dtype,
        mode='r'
    )

    # 初始化DataLoader时传入memmap对象
    dataloader = {}
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_path, args.years, 'idx_' + cat + '.npy'))
        dataloader[cat + '_loader'] = DataLoader(
            data=data_mmap,  # 传入memmap而非全量数据
            idx=idx, seq_len=args.seq_len, horizon=args.horizon, his_len=args.his_len, bs=args.bs, logger=logger
        )
    scaler = StandardScaler(mean=mean, std=std)
    return dataloader, scaler

def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


# def get_dataset_info(dataset):
#
#     base_dir = os.getcwd() + '/data/'
#
#     d = {
#         'CA': [base_dir + 'ca', base_dir + 'ca/ca_rn_adj.npy', 8600],
#         'GLA': [base_dir + 'gla', base_dir + 'gla/gla_rn_adj.npy', 3834],
#         'GBA': [base_dir + 'gba', base_dir + 'gba/gba_rn_adj.npy', 2352],
#         'SD': [base_dir + 'sd', base_dir + 'sd/sd_rn_adj.npy', 716],
#     }
#     assert dataset in d.keys()
#     return d[dataset]

def get_dataset_info(dataset):

    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建data目录的绝对路径（根据实际项目结构调整层级）
    # 假设data目录与当前文件所在目录的相对位置是上级目录下的data
    base_dir = os.path.join(current_dir, '../..', 'data')
    # 规范化路径，处理../等相对路径
    base_dir = os.path.normpath(base_dir)

    # base_dir = os.getcwd() + '/data/'

    d = {
        'CA': [base_dir + '/ca', base_dir + '/ca/ca_rn_adj.npy', 8600],
        'GLA': [base_dir + '/gla', base_dir + '/gla/gla_rn_adj.npy', 3834],
        'GBA': [base_dir + '/gba', base_dir + '/gba/gba_rn_adj.npy', 2352],
        'SD': [base_dir + '/sd', base_dir + '/sd/sd_rn_adj.npy', 716],
    }
    assert dataset in d.keys()
    return d[dataset]