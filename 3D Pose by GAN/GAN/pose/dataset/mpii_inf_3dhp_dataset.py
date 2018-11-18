import glob

import numpy
import scipy.io
import collections
import typing
from . import pose_dataset_base


class H36CompatibleJoints(object):
    joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',
                   'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',
                   'left_wrist', 'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',
                   'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
                   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']
    joint_idx = [4, 23, 24, 25, 18, 19, 20, 3, 5, 7, 6, 9, 10, 11, 14, 15, 16]

    @staticmethod
    def convert_points(raw_vector):
        return numpy.array(
            [(int(raw_vector[i * 2]), int(raw_vector[i * 2 + 1])) for i in H36CompatibleJoints.joint_idx])

    @staticmethod
    def convert_points_3d(raw_vector):
        return numpy.array([
            (float(raw_vector[i * 3]), float(raw_vector[i * 3 + 1]), float(raw_vector[i * 3 + 2])) for i in
            H36CompatibleJoints.joint_idx])


class MPII3DDatasetUtil(object):
    mm3d_chest_cameras = [
        0, 2, 4, 7, 8
    ]  # Subset of chest high, used in "Monocular 3D Human Pose Estimation in-the-wild Using Improved CNN supervision"

    @staticmethod
    def read_cameraparam(path):
        params = collections.defaultdict(dict)
        index = 0
        for line in open(path):
            key = line.split()[0].strip()
            if key == "name":
                value = line.split()[1].strip()
                index = int(value)
            if key == "intrinsic":
                values = line.split()[1:]
                values = [float(value) for value in values]
                values = numpy.array(values).reshape((4, 4))
                params[index]["intrinsic"] = values
            if key == "extrinsic":
                values = line.split()[1:]
                values = [float(value) for value in values]
                values = numpy.array(values).reshape((4, 4))
                params[index]["extrinsic"] = values
        return params


MPII3DDatum = typing.NamedTuple('MPII3DDatum', [
    ('annotation_2d', numpy.ndarray),
    ('annotation_3d', numpy.ndarray),
    ('normalized_annotation_2d', numpy.ndarray),
    ('normalized_annotation_3d', numpy.ndarray),
    ('normalize_3d_scale', float),
])


class MPII3DDataset(pose_dataset_base.PoseDatasetBase):
    def __init__(self, annotations_glob="/mnt/dataset/MPII_INF_3DHP/mpi_inf_3dhp/*/*/annot.mat", train=True):
        self.dataset = []
        for annotation_path in glob.glob(annotations_glob):
            print("load ", annotation_path)
            annotation = scipy.io.loadmat(annotation_path)
            for camera in MPII3DDatasetUtil.mm3d_chest_cameras:
                for frame in range(len(annotation["annot2"][camera][0])):
                    annot_2d = H36CompatibleJoints.convert_points(annotation["annot2"][camera][0][frame])
                    annot_3d = H36CompatibleJoints.convert_points_3d(annotation["annot3"][camera][0][frame])
                    annot_3d_normalized, scale = self._normalize_3d(
                        annot_3d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 3))
                    self.dataset.append(MPII3DDatum(
                        annotation_2d=annot_2d,
                        annotation_3d=annot_3d,
                        normalized_annotation_2d=self._normalize_2d(
                            annot_2d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 2)),
                        normalized_annotation_3d=annot_3d_normalized,
                        normalize_3d_scale=scale,
                    ))
        if train == False:  # just small subset
            self.dataset = self.dataset[:1000]

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        return self.dataset[i].normalized_annotation_2d, \
               self.dataset[i].normalized_annotation_3d, \
               self.dataset[i].normalize_3d_scale
