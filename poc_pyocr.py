from wand.image import Image
from PIL import Image as PI
import pyocr
import pyocr.builders
import io
import numpy as np


PI.MAX_IMAGE_PIXELS = 1000000000  # removes ZIP bomb attack warnings
__DEBUG = True
__DEBUG = False
__MY_THRESHOLD= 190


def apply_rgb_threshold(image, threshold = 170):
     """
     Splits RGB image into components and returns the image composed by points in which
     both R and G and B are above threshold

     :param image:
     :param threshold: from 0 to 255. The lower the threshold, the whiter the image
     :return:
     """

     from hsl_images import img_rgb_to_hls
     return img_rgb_to_hls(image)



     if image.mode[:3] != "RGB":
         return [image,]

     for chan in "RGB":
         pass
         # image.getchannel(chan).show()

     RGB = [np.array(image.getchannel(chan), dtype=np.float32) for chan in "RGB"]
     ret = []
     for channel in RGB:
         channel[channel < threshold] = 0
         ret.append(PI.fromarray(channel.astype('uint8')))

     return ret




def scan_text(image):
    tool = pyocr.get_available_tools()[0]  # tesseract

    #    tool = pyocr.get_available_tools()[1]  # libtesseract
    #    tool = pyocr.get_available_tools()[2]  # cuneiform
    # lang = tool.get_available_languages()[1]
    lang = 'eng'

    builder = pyocr.builders.TextBuilder()
    builder = pyocr.builders.LineBoxBuilder(tesseract_layout=3)
    # builder.tesseract_flags.append("-c")
    # builder.tesseract_flags.append("preserve_interword_spaces=1")

    return tool.image_to_string(
        # image.convert('1', dither = PI.NONE),
        image,
        lang=lang,
        builder=builder
    )

def test_pyocr(filename, tool="tesseract", pdf_resolution=72*5,
               ocr_imageformat="jpeg", enhance_function=None):  # Alternative: jpeg, png

    req_image = []

    image_pdf = Image(filename=filename, resolution=pdf_resolution)

    image_jpeg = image_pdf.convert(ocr_imageformat)

    for idx, img in enumerate(image_jpeg.sequence):
        img_page = Image(image=img)
        if idx == 0 or not __DEBUG:
            req_image.append(img_page.make_blob(ocr_imageformat))

    for img in req_image:
        PImage = PI.open(io.BytesIO(img))
        if enhance_function:
            yield scan_text(enhance_function(PImage))
        #yield scan_text(PImage)
       # continue
        """
        image_parts = crop_image(PImage)
        for idx, part in enumerate(image_parts):
            if idx == 1 and enhance_function:
                image = enhance_function(part)
                yield scan_text(image)
            else:
                yield scan_text(part)
        """

