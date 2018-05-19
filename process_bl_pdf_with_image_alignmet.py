"""
Processes BL documents
"""



import cv2
import numpy as np
from PIL import Image as PI
import io
from wand.image import Image

from poc_pyocr import scan_text
from parse_bill_of_lading import parse_bl, parse_bl_with_counter


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def align_rgb_images(images):
    """
    Adaptation of https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    """

    # Read 8-bit color image.

    if len(images) == 1:
        return images[0]


    # Find the width and height of the color image
    sz = images[0].shape
    #print(sz)

    height = sz[0]
    width = sz[1]

    # Extract the three channels from the gray scale image
    # and merge the three channels into one color image
    im_color = np.zeros((height, width, len(images)), dtype=np.uint8)
    # for i in range(0, 3):
    #    im_color[:, :, i] = im[i * height:(i + 1) * height, :]
    for idx, im in enumerate(images):
        im_color[:, :, idx] = im

    # Allocate space for aligned image
    im_aligned = np.zeros((height, width, len(images)), dtype=np.uint8)

    # The blue and green channels will be aligned to the red channel.
    # So copy the red channel
    im_aligned[:, :, -1] = im_color[:, :, -1]

    # Define motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY
    #warp_mode = cv2.MOTION_AFFINE

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-3)
    #criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-5)

    # Warp the blue and green channels to the red channel
    for i in range(len(images)-1):
        try:
            (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im_color[:, :, -1]), get_gradient(im_color[:, :, i]),
                                                 warp_matrix, warp_mode, criteria)
        except Exception as e:
            break

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use Perspective warp when the transformation is a Homography
            im_aligned[:, :, i] = cv2.warpPerspective(im_color[:, :, i], warp_matrix, (width, height),
                                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use Affine warp when the transformation is not a Homography
            im_aligned[:, :, i] = cv2.warpAffine(im_color[:, :, i], warp_matrix, (width, height),
                                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        #print(warp_matrix)

    return im_aligned


def ong_show(Img):
    """
    Shows a pillow image using the show() command. If the param is a opencv image
    it is converted to a pillow image before being shown
    """
    if isinstance(Img, np.ndarray):
        PI.fromarray(Img.astype('uint8')).show()
    elif isinstance(Img, PI.Image):
        Img.show()
    elif isinstance(Img, tuple):
        ong_show(Img[1])
    else:
        raise Exception("Image type {} not supported in ong_show function".format(type(Img)))

def color_quantize(img):
    """
    Uses k-means to reduce the number of colors of a image...unused but interesting!
    :param img:
    :return:
    """

    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    criteria = (cv2.TERM_CRITERIA_EPS , 50, 1)
    K = 4
    #K = 3
    #K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    # center[center.min(1) > 50, :] = 255   # TODO: deberia filtrar por el minimo de la fila
    #center[center > 180] = 255  # TODO: deberia filtrar por el minimo de la fila
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def crop_place_date_issue_rect(req_image: list) -> list:
    """
    Return a list of images just with the area of the place and date of issue
    Finds lines in the image. Takes the vertical line corresponding to place
    and date of issue an
    :param req_image: a list of images
    :return: a new list of images cropped
    """

    if len(req_image) <=2:
        # if there is just one image is worthless to align it
        return req_image


    horizontal_y = []
    vertical_x = 300 * 12
    min_vertical_y = 300 * 12
    max_vertical_y = 0
    DEBUG_LINES = False

    for img in req_image[:-1]:
        if img.ndim > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        edges = cv2.Canny(gray, 75, 150, apertureSize=3)

        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180,
                                threshold=10, lines=np.array([]),
                                minLineLength=50, maxLineGap=5)

        a, b, c = lines.shape
        vertical_y = []
        for i in range(a):
            x0, y0, x1, y1 = lines[i][0]
            if abs(y0 - y1) < 3:  # horizontal line
                if abs(x1 - x0) < 250:  # Too short, ignore
                    continue
                else:
                    horizontal_y.append(min(y0, y1))
            else:  # non horizontal line
                if abs(x1 - x0) > 10:  # oblicous, ignore
                    continue
                else:  # vertical line
                    if x0 > vertical_x + 10:
                        continue  # only consider leftmost line
                    if x0 < vertical_x - 10:
                        vertical_y = []
                    vertical_y.append(y0)
                    vertical_y.append(y1)
                    vertical_x = min(vertical_x, min(x0, x1))
            if DEBUG_LINES:
                cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3,
                         cv2.LINE_AA)

        if vertical_y:
            min_vertical_y = min(min_vertical_y, min(vertical_y))
            max_vertical_y = max(max_vertical_y, max(vertical_y))

        if DEBUG_LINES:
            cv2.line(img, (vertical_x, min_vertical_y), (vertical_x, max_vertical_y), (255, 0, 0), 3,
                     cv2.LINE_AA)
            ong_show(img)

    # Now find the part of the vertical line to crop. The top is the Y coordinate of the horizontal line
    # closest to the top part of the vertical line and the botton is the Y coordinate of the next
    # horizontal line found
    horizontal_y.sort()
    for y in horizontal_y:
        if min_vertical_y - 10 <= y <= min_vertical_y + 30:
            min_vertical_y = min(min_vertical_y, y)
        elif y > min_vertical_y:
            max_vertical_y = min(max_vertical_y, y)

    if DEBUG_LINES:
        cv2.line(img, (vertical_x, min_vertical_y), (vertical_x, max_vertical_y), (0, 255, 0), 3,
                 cv2.LINE_AA)
        ong_show(img)

    new_req_image = []
    for img in req_image:
        cropped = img[min_vertical_y:max_vertical_y, vertical_x:]
        new_req_image.append(cropped)
        # ong_show(cropped)
    return new_req_image


def process_aligned_image(aligned, min_threshold=220):
    """
    Binarizes an aligned image composed of several channels
    :param aligned: a RGB image, each channel is a BW aligned capture of the same image
    :param min_threshold: min threshold to binarize each BW channel
    :return: a BW image
    """

    __DEBUG_IMG = False
    OFFSET_X = 1
    OFFSET_Y = 0

    if aligned.ndim > 2:
        """
        Slow solution based on pixel matching and some tolerance
        """
        """
        new_aligned = 255 * np.ones(aligned.shape[:2])
        # hypersupermega slow solution
        for x in range(OFFSET_X, aligned.shape[0] - OFFSET_X):
            for y in range(OFFSET_Y, aligned.shape[1] - OFFSET_Y):
                if aligned[x, y, 0] < min_threshold:
                    aligned_ok = True
                    for color in range(1, aligned.shape[2]):
                        if aligned[x - OFFSET_X:x + OFFSET_X + 1,
                                    y - OFFSET_Y:y + OFFSET_Y + 1,
                                    color].min() > min_threshold:
                            aligned_ok = False
                            break
                    if aligned_ok:
                        new_aligned[x, y] = 0
        """
        for color in range(aligned.shape[2]):
            # aligned[:, :, color] = cv2.blur(aligned[:, :, color], (5,5))
            #aligned[:, :, color] = cv2.GaussianBlur(aligned[:, :, color], (0, 0), 0.5)

            _, aligned[:, :, color] = cv2.threshold(aligned[:, :, color],
                                                    min_threshold, 255, cv2.THRESH_BINARY)
        if __DEBUG_IMG:
            ong_show(aligned)

        new_aligned = 0 * np.ones(aligned.shape[:2])
        new_aligned[aligned.max(2) > 250] = 255

    else:
        # _, new_aligned = cv2.threshold(aligned, min_threshold, 255, cv2.THRESH_BINARY)
        new_aligned = aligned

    if __DEBUG_IMG:
        ong_show(new_aligned)

    return new_aligned




def get_botton_right_imgs(req_imgs) -> list:
    """
    Crop the list of images to the botton right part (where place and
    date of issue lays) and return it as a list of opencv images
    :param req_imgs: list of pillow images to crop from pdf
    :return: None if just one image found, the list of opencv if found
    """
    ret = []
    #Last image is the general conditions, so don't take it into account
    for i in range(max(len(req_imgs) - 1, 1)):
        pil_image = req_imgs[i]
        # ong_show(pil_image)
        open_cv_image = np.array(pil_image)

        size = open_cv_image.shape
        w = size[1]
        h = size[0]

        min_w = int(w * .53)
        max_w = int(w * .95)
        min_h = int(h * .72)
        max_h = int(h * .9)

        open_cv_image = open_cv_image[ min_h:max_h, min_w:max_w, :]

        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        ret.append(open_cv_image)
        #ong_show(open_cv_image)


    return ret


def process_date_time_issue(req_pil_images):
    """
    From all (but the last) pages of the B/L pdf: gets get the bottom right
    part, in which looks for the "date and place of issue" box, crops the box,
    align all the cropped images (splitting into 3 horizontal parts to avoid
    missalignments due to poor scanning) and applies OCR to the aligned images
    which are very clear.
    It assumes that a human stamps and signs the document, so it is not located
    exactly in the same position in each copy of the document so the above procedure
    yields a clear image as the clear text without signature nor stamp is easier
    to filfer from the aligned image
    """
    if len(req_pil_images) <= 2:
        return None
    else:
        req_image = get_botton_right_imgs(req_pil_images)

        aligned = align_rgb_images(crop_place_date_issue_rect(req_image))
        # split the image into parts and realigne, to help solving missalignments
        if aligned.ndim > 2:
            parts = 3
            h, w = aligned.shape[:2]
            w_part = int(w / parts)
            for p in range(parts):
                split_aligned = []
                x_min = w_part * p
                x_max = min(x_min + w_part, w)
                for color in range(aligned.shape[2]):
                    im_part = aligned[:, x_min:x_max, color]
                    split_aligned.append(im_part)
                aligned[:, x_min:x_max, :] = align_rgb_images(split_aligned)
        # ong_show(aligned)

        aligned = process_aligned_image(aligned)

        #ong_show(aligned)
        img = PI.fromarray(aligned.astype('uint8'))

        res = [scan_text(img)]
        parsed = parse_bl(res)
        # Get rid of the none values
        parsed = {k: v for k, v in parsed.items() if v is not None}

        #        print(parsed)
        return parsed


def parse_bl_with_alignment(BL_NAME):
        image_pdf = Image(filename=BL_NAME, resolution=310)
        image_jpeg = image_pdf.convert('jpeg')
        req_pil_images = []

        for idx, img in enumerate(image_jpeg.sequence):
            img_page = Image(image=img)
            pil_image = PI.open(io.BytesIO(img_page.make_blob())).convert('RGB')
            req_pil_images.append(pil_image)

        ocr_pages = []
        for img in req_pil_images:
            ocr_pages.append(scan_text(img))
        parsed, counter = parse_bl_with_counter(ocr_pages)
        # Focus on place and date of issue. If in at least two pages the same value
        # was found, do nothing. Otherwise take all the pages, crop the botton right
        # align all of them into a single image, combine it and parse again that part
        if counter.get("Place and date of issue", 0) < 2:
            date_issue = process_date_time_issue(req_pil_images)
            if date_issue:
                parsed.update(date_issue)

        return parsed


if __name__ == '__main__':

    from config_endesa import config_bill_of_lading
    bl_names = config_bill_of_lading.keys()

    #i = 0
    for BL_NAME in bl_names:
        parsed = parse_bl_with_alignment(BL_NAME)
        print(parsed)

        """
        Tratar las imagenes en el espacio LAB
        
        
        for img in req_image[:-1]:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            ong_show(cv2.threshold(l, 160, 255, cv2.THRESH_BINARY))

            dst = cv2.bitwise_and(l, b)
            dst = cv2.bitwise_and(dst, a)
            ong_show(dst)
        """

        """
        Use template matching to find T and inverted T corners
        
        for img in req_image[:-1]:
            if img.ndim > 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)

            #binary = cv2.bitwise_not(gray)
            ong_show(binary)

            fil_T = 5
            template = np.zeros((fil_T * 2 + 1,fil_T * 2 + 1), dtype=np.uint8)

            template[fil_T:fil_T + 2, :] = 255
            template[fil_T:, fil_T:fil_T + 2 ] = 255
            template[fil_T:, fil_T+1] = 255

            #template = cv2.bitwise_not(template)


            w, h = template.shape[::-1]

            res = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

            ong_show(img)
        """

        #continue

        #continue
        #img_to_quantize = req_image[0]
        #img_to_quantize = cv2.cvtColor(img_to_quantize, cv2.COLOR_BGR2GRAY)
        #quantized = color_quantize(img_to_quantize)
        #quantized = img_to_quantize
        #import hsl_images
        #im_hls = hsl_images.img_rgb_to_hls(req_pil_images[0])
        #im_hls.save("hls.jpeg")

        #cv2.imshow("hsl", im_hsl)
        #cv2.waitKey(0)



        """
        from align_images_features import alignImages
        feat_aligned = np.zeros(req_image[0].shape + (len(req_image), ))
        feat_aligned[:, :, 0] = req_image[0]
        for i in range(1, len(req_image)):
            feat_aligned[:, :, i], h = alignImages(req_image[i], req_image[0])
        PI.fromarray(feat_aligned.astype('uint8')).show()
    
        new_aligned = 254 * np.ones(feat_aligned.mean(2).shape[:2])
        # Original : 235
        new_aligned[feat_aligned.mean(2) < 135] = 0
        """







        """
        pil_aligned = PI.fromarray(aligned.astype('uint8'))
        pil_aligned.show()
        temp_file = "ejemplo.png"
        pil_aligned.save(temp_file)
    
        from test_pythonfu import run_gimp_script
        run_gimp_script(temp_file)
        img = PI.open(temp_file + "_out.png")
        """

        """
        # Apply unsharp_mask
        gaussian_3 = cv2.GaussianBlur(new_aligned, (0, 0), 9)
        unsharp_image = cv2.addWeighted(new_aligned, 1.5, gaussian_3, -0.5, 0, new_aligned)
        ong_show(unsharp_image)
        """
#        continue


        #new_aligned = aligned
        #img = PI.fromarray(new_aligned.astype('uint8'))
        """
    #    H, L, S = im_hls.split()
    
        #im_hls.show("HSL")
    #    img = L
    
        #img.show("Solo L")
        img = PI.fromarray(aligned.astype('uint8'))
        #img.show()
        gray = img.convert('L')
        #img = gray.point(lambda x: 0 if x < 160 else 255, '1')
        img.show()
        #img = im_hls
        """
        #for line in res:
        #    print(line.content)