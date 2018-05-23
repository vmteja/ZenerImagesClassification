import os
import sys
import math
import numpy
import random
import string
from PIL import Image, ImageDraw, ImageOps

"""
usage : python zener_generator.py folder_name num_examples
to do : 
    2. change the perspective_factor variable to generate diff datasets req for step-4 

"""

FILE_NAME_LETTER_MAP = {
	"plus": "P",
	"square": "Q",
	"star": "S",
	"waves": "W",
        "circle": "O"
}

# specify maximum rotation in degrees
rotation_range = 45

# dimensions of the output image required 
# if n, generated img dimensions would be nxn
training_dimension = 25

# used for diff sizes and distortion 
perspec_factor = 60
scale_factor = 100.0 * (1.0 / perspec_factor)


# helper func for shifting symbols
def generate_random_shifts(img_size, factor):
    w = img_size[0] / factor
    h = img_size[1] / factor
    shifts = []
    for s in range(0, 4):
        w_shift = (random.random() - 0.5) * w
        h_shift = (random.random() - 0.5) * h
        shifts.append((w_shift, h_shift))
    return shifts

# returns coeff used to transform the entire image
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


# Modifies image by random shifts and changing perspective
def create_perspective(img, factor):
    img_size = img.size
    w = img_size[0]
    h = img_size[1]
    shifts = generate_random_shifts(img_size, factor)
    coeffs = find_coeffs(
        [(shifts[0][0], shifts[0][1]),
         (w + shifts[1][0], shifts[1][1]),
         (w + shifts[2][0], h + shifts[2][1]),
         (shifts[3][0], h + shifts[3][1])], [(0, 0), (w, 0), (w, h), (0, h)])
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


# due to rotation and/or perspective we will need to fill in the background
def mask_image(img):
    mask = Image.new("RGBA", img.size, (255, 255, 255, 255))
    return Image.composite(img, mask, img)

# will adjust the canvas so that respective transforms will not result 
# in the image being cropped
def adjust_image(img, factor):
        # padding to allow more space for image distortion
	padding_factor = 4
	width, height = img.size
	# choose largest dimension
	img_largest_dim = (width, height)[width < height]
	canvas_dim = int(math.floor(img_largest_dim + (padding_factor * (img_largest_dim / factor))))
	canvas_size = (canvas_dim, canvas_dim)
	img_pos = (int(math.floor((canvas_size[0] - width) / 2)), int(math.floor((canvas_size[1] - height) / 2)))
	new_canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
	new_canvas.paste(img, (img_pos[0], img_pos[1], img_pos[0] + width, img_pos[1] + height))
	return new_canvas

# will randomly rotate the image
def rotate_image(img, rotation):
    rotation_factor = math.pow(random.uniform(0.0, 1.0), 4)
    rotation_direction = (1, -1)[random.random() > 0.5]
    rotation_angle = int(math.floor(rotation * rotation_factor * rotation_direction))
    return img.rotate(rotation_angle)


# crop the image to a square that bounds the image using largest bounding-box dimension
# and then resize the image as per given dimensions 
def crop_resize(img, dimension):
	inv_img = ImageOps.invert(img.convert("RGB"))
	# returns left, upper, right, lower
	left, upper, right, lower = inv_img.getbbox()
	width = right - left
	height = lower - upper
	if width > height:
		# we want to add half the difference between width and height
		# to the upper and lower dimension
		padding = int(math.floor((width - height) / 2))
		upper -= padding
		lower += padding
	else:
		padding = int(math.floor((height - width) / 2))
		left -= padding
		right += padding

	img = img.crop((left, upper, right, lower))
	return img.resize((dimension, dimension), Image.LANCZOS)


#fetches sample gener images
def fetch_zener_symbol_images():
    zener_images = {}
    image_path = "./zener_images"
    for root, dirs, files in os.walk(image_path):
        for f in files:
            if f.endswith(".png"):
                image_name = string.split(f, ".")
                image = Image.open(image_path + "/" + f)
                zener_images[FILE_NAME_LETTER_MAP[image_name[0]]] = image
    return zener_images


# uses all the other the methods to distort and finalise the image
def distort_image(img, factor, rotation, dimension):
    img = create_perspective(img, factor)
    img = rotate_image(img, rotation)
    img = mask_image(img)
    img = crop_resize(img, dimension)
    return img

def rm_files(folder):
    """
    deletes files from the path provided ('folder') provided 
    """
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
           if os.path.isfile(file_path):
              os.unlink(file_path)
        except Exception as e:
           print(e)

# threshold value for converting to Black and white image
threshold = 100

# execution starts here 
if __name__ == "__main__":

   # reading the arguments passed
   folder_name = sys.argv[1]
   no_examples = int(sys.argv[2])

   # load the sample zener image symbols
   images = fetch_zener_symbol_images()

   # read the names of zener image symbols in list
   image_symbols = images.keys()

   # directory for saving zener images
   generated_folder = folder_name + '/'
   if not os.path.exists(generated_folder):
      os.makedirs(generated_folder)
   else: 
      # if dir already exists remove the 
      # earlier generated files  
      rm_files(generated_folder)

   print ("generating the training data set .............")
   print ("............................")
   for i in range(1, no_examples+1):
       #randomly choosing a zener image
       picked_symbol = random.choice(image_symbols)
       image_name = "%d_%s"%(i,picked_symbol[0])
       symbol_img = images[picked_symbol]

       adjusted_img = adjust_image(symbol_img, scale_factor)
       modified_image = distort_image(adjusted_img, scale_factor, rotation_range, training_dimension)
       
       # converting to binary image 
       img = modified_image.convert('L')
       img = img.point(lambda p: p > threshold and 255)
       img.save(generated_folder + image_name + ".png")
   print ("dataset generated ")
