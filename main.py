import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model import *
from dataset import ShapeNetCore

def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr

    return LambdaLR(optimizer, lr_lambda=lr_func)


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

path = './data/shapenet.hdf5'
cate = "airplane"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw = False  # 모든 time step의 결과 보기 (저장 포함)

train_dataset = ShapeNetCore(
    path=path,
    cate=cate,
    split='train'
)
val_dataset = ShapeNetCore(
    path=path,
    cate=cate,
    split='val'
)

model = DiffusionPoint(
    SimpleNet(),
    VarianceSchedule()
).to(device)

train_iter = get_data_iterator(DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
))

val_iter = get_data_iterator(DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=True
))

optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0)
schedular = get_linear_scheduler(optimizer, 200000, 400000, 2e-3, 1e-4)

def save_pcds(pcds: dict):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for t, pcl in pcds.items():
        pcl = pcl.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl[0])

        color = np.array([[1, 0, 0]] * pcl.shape[0])
        pcd.colors = o3d.utility.Vector3dVector(color)

        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        view_control = vis.get_view_control()
        view_control.set_up([0, 1, 0])  # 카메라의 상향 벡터를 Y축으로 설정
        view_control.set_front([1, 1, 1])  # 카메라가 Z축 위에서 아래로 향하도록 설정
        view_control.set_lookat([0, 0, 0])  # 포인트 클라우드의 중심을 원점(0, 0, 0)으로 설정
        view_control.set_zoom(0.5)  # 확대/축소 비율 조정
        vis.poll_events()
        vis.update_renderer()
        if os.path.exists("imgs"):
            pass
        else:
            os.mkdir("imgs")

        img_pth = f"imgs/{t}.png"
        vis.capture_screen_image(img_pth)
        vis.clear_geometries()
        print(f"saved image at {t}")

    vis.destroy_window()

def train(it):
    x = next(train_iter)
    x = x.to(device)
    optimizer.zero_grad()
    model.train()

    loss = model.get_loss(x)
    print(it, ":", loss)
    loss.backward()
    optimizer.step()
    schedular.step()

def test(it):  # Visualization
    tgt = next(val_iter)
    tgt = tgt[0]
    num_points, coord = tgt.size()
    tgt = np.asarray(tgt)
    tgt = tgt - np.array([0, 2, 0])

    with torch.no_grad():
        res = model.sample(1, num_points, coord, device, ret_traj=raw)
        if raw:
            save_pcds(res)
            res = res[0]
            points_numpy = res[0].cpu().numpy()
        else:
            points_numpy = res[0].cpu().numpy()

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points_numpy)

        color = np.array([[1, 0, 0]] * points_numpy.shape[0])
        pcd1.colors = o3d.utility.Vector3dVector(color)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(tgt)

        color = np.array([[0, 0, 1]] * points_numpy.shape[0])
        pcd2.colors = o3d.utility.Vector3dVector(color)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2)

        view_control = vis.get_view_control()
        view_control.set_up([0, 1, 0])  # 카메라의 상향 벡터를 Y축으로 설정
        view_control.set_front([1, 1, 1])  # 카메라가 Z축 위에서 아래로 향하도록 설정
        view_control.set_lookat([0, 0, 0])  # 포인트 클라우드의 중심을 원점(0, 0, 0)으로 설정
        view_control.set_zoom(0.5)  # 확대/축소 비율 조정
        vis.run()

if __name__ == '__main__':
    it = 1
    while it <= 400000:
        if it % 10000 == 0:
            test(it)
        train(it)
        it += 1
