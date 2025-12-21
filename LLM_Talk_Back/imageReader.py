import pytesseract
from PIL import Image

# Path to installed tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open("image.png")
img = img.convert("RGB")
text = pytesseract.image_to_string(img, lang="tam")

print(text)

