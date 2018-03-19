from wand.image import Image
from PIL import Image as PI
import pyocr
import pyocr.builders
import io

def test_pyocr(filename, tool="tesseract"):
    tool = pyocr.get_available_tools()[0]  # tesseract
#    tool = pyocr.get_available_tools()[1]  # libtesseract
#    tool = pyocr.get_available_tools()[2]  # cuneiform
    lang = tool.get_available_languages()[1]
    lang = 'eng'

    final_text = []
    req_image = []
    builder = pyocr.builders.TextBuilder()
    builder = pyocr.builders.LineBoxBuilder()
    #builder.tesseract_flags.append("-c")
    #builder.tesseract_flags.append("preserve_interword_spaces=1")

    image_pdf = Image(filename=filename, resolution=340)
    image_jpeg = image_pdf.convert('jpeg')
    for img in image_jpeg.sequence:
        img_page = Image(image=img)
        req_image.append(img_page.make_blob('jpeg'))
    for img in req_image:
            txt = tool.image_to_string(
            PI.open(io.BytesIO(img)),
            lang=lang,
            builder=builder
            )
            final_text.append(txt)
    return final_text
