import PIL
from PIL import Image

with open('picasso.jpg', 'rb') as f:
    image = f.read()

with open('images/1.jpg', 'wb') as f:
    f.write(image)

with open('images/2.jpg', 'wb') as f:
    f.write(image)