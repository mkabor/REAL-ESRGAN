import argparse
import cv2
import glob
import os


def main(args):
    txt_file = open(args.meta_info, 'w')
    for folder, root in zip(args.input, args.root):
        img_paths = sorted(glob.glob(os.path.join(folder, '*')))
        for img_path in img_paths:
            status = True
            if args.check:
                # read the image once for check, as some images may have errors
                try:
                    img = cv2.imread(img_path)
                except (IOError, OSError) as error:
                    print(f'Read {img_path} error: {error}')
                    status = False
                if img is None:
                    status = False
                    print(f'Img is None: {img_path}')
            if status:
                # get the relative path
                img_name = os.path.relpath(img_path, root)
                print(img_name)
                txt_file.write(f'{img_name}\n')


if __name__ == '__main__':
    """Generate meta info (txt file) for only Ground-Truth images.

    It can also generate meta info from several folders into one txt file.
    """
    
    # Préparation de train
    # --------------------
    
    # HR images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train/HR', '/content/REAL-ESRGAN/datasets/data_with_da/train/HR_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train', '/content/REAL-ESRGAN/datasets/data_with_da/train'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/train/meta_info/meta_info_trainhrmultiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx2 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train/LR/x2', '/content/REAL-ESRGAN/datasets/data_with_da/train/LR/x2_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/train/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/train/LR/meta_info/meta_info_trainlrx2multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx3 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train/LR/x3', '/content/REAL-ESRGAN/datasets/data_with_da/train/LR/x3_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/train/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/train/LR/meta_info/meta_info_trainlrx3multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx4 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train/LR/x4', '/content/REAL-ESRGAN/datasets/data_with_da/train/LR/x4_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/train/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/train/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/train/LR/meta_info/meta_info_trainlrx4multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # Préparation de val
    # --------------------
    
    # HR images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val/HR', '/content/REAL-ESRGAN/datasets/data_with_da/val/HR_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val', '/content/REAL-ESRGAN/datasets/data_with_da/val'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/val/meta_info/meta_info_valhrmultiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx2 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val/LR/x2', '/content/REAL-ESRGAN/datasets/data_with_da/val/LR/x2_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/val/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/val/LR/meta_info/meta_info_vallrx2multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx3 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val/LR/x3', '/content/REAL-ESRGAN/datasets/data_with_da/val/LR/x3_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/val/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/val/LR/meta_info/meta_info_vallrx3multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx4 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val/LR/x4', '/content/REAL-ESRGAN/datasets/data_with_da/val/LR/x4_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/val/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/val/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/val/LR/meta_info/meta_info_vallrx4multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    
    # Préparation de test
    # --------------------
    
    # HR images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test/HR', '/content/REAL-ESRGAN/datasets/data_with_da/test/HR_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test', '/content/REAL-ESRGAN/datasets/data_with_da/test'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/test/meta_info/meta_info_testhrmultiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx2 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test/LR/x2', '/content/REAL-ESRGAN/datasets/data_with_da/test/LR/x2_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/test/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/test/LR/meta_info/meta_info_testlrx2multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx3 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test/LR/x3', '/content/REAL-ESRGAN/datasets/data_with_da/test/LR/x3_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/test/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/test/LR/meta_info/meta_info_testlrx3multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
    
    # LRx4 images
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test/LR/x4', '/content/REAL-ESRGAN/datasets/data_with_da/test/LR/x4_sub'],
        help='Input folder, can be a list')
    parser.add_argument(
        '--root',
        nargs='+',
        default=['/content/REAL-ESRGAN/datasets/data_with_da/test/LR/', '/content/REAL-ESRGAN/datasets/data_with_da/test/LR/'],
        help='Folder root, should have the length as input folders')
    parser.add_argument(
        '--meta_info',
        type=str,
        default='/content/REAL-ESRGAN/datasets/data_with_da/test/LR/meta_info/meta_info_testlrx4multiscale.txt',
        help='txt path for meta info')
    parser.add_argument('--check', action='store_true', help='Read image to check whether it is ok')
    args = parser.parse_args()

    assert len(args.input) == len(args.root), ('Input folder and folder root should have the same length, but got '
                                               f'{len(args.input)} and {len(args.root)}.')
    os.makedirs(os.path.dirname(args.meta_info), exist_ok=True)
    main(args)
