from __future__ import print_function
import os
import glob
import dlib
import torch
import torch.nn as nn
from enum import Enum
from skimage import io
from skimage import color
import cv2
import uuid
import torchvision
from torchvision import transforms
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *
#from pose import datasets, hopenet, utils
from .pose.hopenet import Hopenet
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F


class LandmarksType(Enum):
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class FaceAlignment:
    """Initialize the face alignment pipeline

    Args:
        landmarks_type (``LandmarksType`` object): an enum defining the type of predicted points.
        network_size (``NetworkSize`` object): an enum defining the size of the network (for the 2D and 2.5D points).
        enable_cuda (bool, optional): If True, all the computations will be done on a CUDA-enabled GPU (recommended).
        enable_cudnn (bool, optional): If True, cudnn library will be used in the benchmark mode
        flip_input (bool, optional): Increase the network accuracy by doing a second forward passed with
                                    the flipped version of the image
        use_cnn_face_detector (bool, optional): If True, dlib's CNN based face detector is used even if CUDA
                                                is disabled.

    Example:
        >>> FaceAlignment(NetworkSize.2D, flip_input=False)
    """

    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 enable_cuda=True, enable_cudnn=True, flip_input=False,
                 use_cnn_face_detector=False):
        self.enable_cuda = enable_cuda
        self.use_cnn_face_detector = use_cnn_face_detector
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.pose_model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        if enable_cuda:
            self.pose_model.cuda(0)
        self.pose_model.eval()
        base_path = os.path.join(appdata_dir('face_alignment'), "data")

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        if enable_cudnn and self.enable_cuda:
            torch.backends.cudnn.benchmark = True

        # Initialise the face detector
        if self.use_cnn_face_detector:
            path_to_detector = os.path.join(
                base_path, "mmod_human_face_detector.dat")
            if not os.path.isfile(path_to_detector):
                print("Downloading the face detection CNN. Please wait...")

                path_to_temp_detector = os.path.join(
                    base_path, "mmod_human_face_detector.dat.download")

                if os.path.isfile(path_to_temp_detector):
                    os.remove(os.path.join(path_to_temp_detector))

                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/dlib/mmod_human_face_detector.dat",
                    os.path.join(path_to_temp_detector))

                os.rename(os.path.join(path_to_temp_detector),os.path.join(path_to_detector))

            self.face_detector = dlib.cnn_face_detection_model_v1(
                path_to_detector)

        else:
            self.face_detector = dlib.get_frontal_face_detector()

        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(int(network_size))
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(int(network_size)) + '.pth.tar'
        else:
            network_name = '3DFAN-' + str(int(network_size)) + '.pth.tar'
        fan_path = os.path.join(base_path, network_name)

        if not os.path.isfile(fan_path):
            print("Downloading the Face Alignment Network(FAN). Please wait...")

            fan_temp_path = os.path.join(base_path,network_name+'.download')

            if os.path.isfile(fan_temp_path):
                os.remove(os.path.join(fan_temp_path))

            request_file.urlretrieve(
                "https://www.adrianbulat.com/downloads/python-fan/" +
                network_name, os.path.join(fan_temp_path))

            os.rename(os.path.join(fan_temp_path),os.path.join(fan_path))

        fan_weights = torch.load(
            fan_path,
            map_location=lambda storage,
            loc: storage)

        self.face_alignment_net.load_state_dict(fan_weights)

        if self.enable_cuda:
            self.face_alignment_net.cuda()
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()
            depth_model_path = os.path.join(base_path, 'depth.pth.tar')
            if not os.path.isfile(depth_model_path):
                print(
                    "Downloading the Face Alignment depth Network (FAN-D). Please wait...")

                depth_model_temp_path = os.path.join(base_path, 'depth.pth.tar.download')

                if os.path.isfile(depth_model_temp_path):
                    os.remove(os.path.join(depth_model_temp_path))


                request_file.urlretrieve(
                    "https://www.adrianbulat.com/downloads/python-fan/depth.pth.tar",
                    os.path.join(depth_model_temp_path))

                os.rename(os.path.join(depth_model_temp_path),os.path.join(depth_model_path))

            depth_weights = torch.load(
                depth_model_path,
                map_location=lambda storage,
                loc: storage)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            if self.enable_cuda:
                self.depth_prediciton_net.cuda()
            self.depth_prediciton_net.eval()

    def detect_faces(self, image):
        """Run the dlib face detector over an image

        Args:
            image (``ndarray`` object or string): either the path to the image or an image previosly opened
            on which face detection will be performed.

        Returns:
            Returns a list of detected faces
        """
        return self.face_detector(image, 1)

    def get_landmarks(self, input_image, all_faces=False):
        with torch.no_grad():
            if isinstance(input_image, str):
                try:
                    image = io.imread(input_image)
                except IOError:
                    print("error opening file :: ", input_image)
                    return None
            else:
                image = input_image

            # Use grayscale image instead of RGB to speed up face detection
            detected_faces = self.detect_faces(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))            
  
            if image.ndim == 2:
                image = color.gray2rgb(image)

            if len(detected_faces) > 0:
                landmarks = []
                for i, d in enumerate(detected_faces):
                    if i > 0 and not all_faces:
                        break
                    if self.use_cnn_face_detector:
                        d = d.rect

                    center = torch.FloatTensor(
                        [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                         (d.bottom() - d.top()) / 2.0])
                    center[1] = center[1] - (d.bottom() - d.top()) * 0.12
                    scale = (d.right() - d.left() +
                             d.bottom() - d.top()) / 195.0

                    inp = crop(image, center, scale)
                    inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).float().div(255.0).unsqueeze_(0)

                    if self.enable_cuda:
                        inp = inp.cuda()

                    out = self.face_alignment_net(inp)[-1].data.cpu()
                    if self.flip_input:
                        out += flip(self.face_alignment_net(flip(inp))
                                    [-1].data.cpu(), is_label=True)

                    pts, pts_img = get_preds_fromhm(out, center, scale)
                    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

                    if self.landmarks_type == LandmarksType._3D:
                        heatmaps = np.zeros((68, 256, 256))
                        for i in range(68):
                            if pts[i, 0] > 0:
                                heatmaps[i] = draw_gaussian(
                                    heatmaps[i], pts[i], 2)
                        heatmaps = torch.from_numpy(
                            heatmaps).view(1, 68, 256, 256).float()
                        if self.enable_cuda:
                            heatmaps = heatmaps.cuda()
                        depth_pred = self.depth_prediciton_net(
                            torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                        pts_img = torch.cat(
                            (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

                    landmarks.append(pts_img.numpy())
            else:
                print("Warning: No faces were detected.")
                return None

            return landmarks

    def process_folder(self, path, all_faces=False):
        types = ('*.jpg', '*.png')
        images_list = []
        for files in types:
            images_list.extend(glob.glob(os.path.join(path, files)))

        predictions = []
        for image_name in images_list:
            predictions.append((
                image_name, self.get_landmarks(image_name, all_faces)))

        return predictions

    def remove_models(self):
        base_path = os.path.join(appdata_dir('face_alignment'), "data")
        for data_model in os.listdir(base_path):
            file_path = os.path.join(base_path, data_model)
            try:
                if os.path.isfile(file_path):
                    print('Removing ' + data_model + ' ...')
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    def get_default_bounding_box(self,input_image,det):
        x_min = det.left()
        y_min = det.top()
        x_max = det.right()
        y_max = det.bottom()
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        x_min -= bbox_width / 4
        x_max += bbox_width / 4
        y_min -= bbox_height / 4
        y_max += 3 * bbox_height / 4
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(input_image.shape[1], x_max)
        y_max = min(input_image.shape[0], y_max)
        x_min = int(np.round(x_min)); x_max = int(np.round(x_max))
        y_min = int(np.round(y_min)); y_max = int(np.round(y_max))
        return x_min, x_max, y_min, y_max

    def get_head_pose(self, input_image):
        idx_tensor = [idx for idx in range(66)]
        idx_tensor = torch.FloatTensor(idx_tensor)
        if self.enable_cuda:
            idx_tensor = idx_tensor.cuda(0)
        img = Image.fromarray(input_image)
        transformations = transforms.Compose([transforms.Scale(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img)
        if self.enable_cuda:
            img = img.cuda(0)
        yaw, pitch, roll = self.pose_model(img)
        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
        return {'yaw' : yaw_predicted, 'pitch' : pitch_predicted, 'roll' : roll_predicted}

    def get_landmarks_with_rectangles(self, input_image, all_faces=False): ## Returns dictionary w/ uuid's pre-made
        with torch.no_grad():
            if isinstance(input_image, str):
                try:
                    image = io.imread(input_image)
                except IOError:
                    print("error opening file :: ", input_image)
                    return None
            else:
                image = input_image

            # Use grayscale image instead of RGB to speed up face detection
            detected_faces = self.detect_faces(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

            if image.ndim == 2:
                image = color.gray2rgb(image)

            if len(detected_faces) > 0:
                landmarks = {}
                for i, d in enumerate(detected_faces):
                    if i > 0 and not all_faces:
                        break
                    new_uuid = uuid.uuid4().hex
                    rect = d
                    if self.use_cnn_face_detector:
                        d = d.rect

                    center = torch.FloatTensor(
                        [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                         (d.bottom() - d.top()) / 2.0])
                    center[1] = center[1] - (d.bottom() - d.top()) * 0.12
                    scale = (d.right() - d.left() +
                             d.bottom() - d.top()) / 195.0

                    inp = crop(image, center, scale)
                    inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).float().div(255.0).unsqueeze_(0)

                    if self.enable_cuda:
                        inp = inp.cuda()

                    out = self.face_alignment_net(inp)[-1].data.cpu()
                    if self.flip_input:
                        out += flip(self.face_alignment_net(flip(inp))
                                    [-1].data.cpu(), is_label=True)

                    pts, pts_img = get_preds_fromhm(out, center, scale)
                    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

                    if self.landmarks_type == LandmarksType._3D:
                        heatmaps = np.zeros((68, 256, 256))
                        for i in range(68):
                            if pts[i, 0] > 0:
                                heatmaps[i] = draw_gaussian(
                                    heatmaps[i], pts[i], 2)
                        heatmaps = torch.from_numpy(
                            heatmaps).view(1, 68, 256, 256).float()
                        if self.enable_cuda:
                            heatmaps = heatmaps.cuda()
                        depth_pred = self.depth_prediciton_net(
                            torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                        pts_img = torch.cat(
                            (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

                    x_min, x_max, y_min, y_max = self.get_default_bounding_box(image,d)
                    pose = self.get_head_pose(input_image=image[y_min:y_max,x_min:x_max])

                    #landmarks.append(pts_img.numpy())
                    landmarks[new_uuid] = {}
                    landmarks[new_uuid]['rectangle'] = d
                    landmarks[new_uuid]['landmarks'] = pts_img.numpy()
                    landmarks[new_uuid]['confidence'] = rect.confidence
                    landmarks[new_uuid]['pose'] = pose
            else:
                print("Warning: No faces were detected.")
                return {}

            return landmarks
