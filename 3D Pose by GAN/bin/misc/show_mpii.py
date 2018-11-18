import os
import cv2
import scipy.io
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import projection_gan

PATH = "/mnt/dataset/MPII_INF_3DHP/mpi_inf_3dhp/S1/Seq1"


def main():
    annotations = scipy.io.loadmat(os.path.join(PATH, "annot.mat"))
    camera = 2
    annotation2d = annotations["annot2"][camera][0]
    annotation3d = annotations["annot3"][camera][0]
    cameraparam = projection_gan.pose.dataset.mpii_inf_3dhp_dataset.MPII3DDatasetUtil.read_cameraparam(
        os.path.join(PATH, "camera.calibration"))
    intr = cameraparam[camera]["intrinsic"][:3, :3]

    for i in range(000, 10, 1):
        frame = cv2.imread(os.path.join(PATH, "imageSequence", "video_{}".format(camera), "{:04}.png".format(i)))
        joints = projection_gan.pose.dataset.mpii_inf_3dhp_dataset.H36CompatibleJoints.convert_points(
            annotation2d[i])

        for pt in joints:
            cv2.circle(frame, tuple(pt), radius=7, color=(0, 0, 255), thickness=-1)
        cv2.imwrite("out.png", frame)

        joints_3d = projection_gan.pose.dataset.mpii_inf_3dhp_dataset.H36CompatibleJoints.convert_points_3d(
            annotation3d[0])

        y = joints_3d.dot(intr.T)
        yy = y[:, :2] / y[:, 2:]


if __name__ == '__main__':
    main()
