import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def super_resolution_images(input_path, model_name='RealESRGAN_x4plus', denoise_strength=0.5,
                   outscale=4, model_path=None, face_enhance=False, fp32=False,
                   alpha_upsampler='realesrgan', gpu_id=None):
    """Process an image for super-resolution using Real-ESRGAN and return the processed image.
    
    Args:
        input_path (str): Input image path.
        model_name (str): Model name for super-resolution.
        denoise_strength (float): Denoise strength for the general model.
        outscale (float): Upsampling scale.
        model_path (str): Path to model weights.
        face_enhance (bool): Whether to enhance faces using GFPGAN.
        fp32 (bool): Use full precision for inference.
        alpha_upsampler (str): Upsampler for alpha channels.
        gpu_id (int): GPU device ID.
    
    Returns:
        output (numpy.ndarray): The processed image.
    """
    
    # Determine models according to model names
    model = None
    netscale = None
    file_url = None
    
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # Determine model paths
    if model_path is not None:
        model_path = model_path
    else:
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # Use dni to control the denoise strength
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]

    # Restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,  # no tile during testing
        tile_pad=10,
        pre_pad=0,
        half=not fp32,
        gpu_id=gpu_id)

    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # Read input image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    img_mode = 'RGBA' if len(img.shape) == 3 and img.shape[2] == 4 else None

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error:', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        return None  # Return None on error

    return output  # Return the processed image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image path')
    parser.add_argument('-n', '--model_name', type=str, default='RealESRGAN_x4plus', help='Model name for super-resolution')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5, help='Denoise strength')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--model_path', type=str, default=None, help='Model path (optional)')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference')
    parser.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU device to use')

    args = parser.parse_args()

    # Call process_images with the parsed arguments
    output_image = process_images(args.input, args.model_name, args.denoise_strength, args.outscale,
                                   args.model_path, args.face_enhance, args.fp32, args.gpu_id)

    if output_image is not None:
        # Optionally display or save the output image
        cv2.imshow('Processed Image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
