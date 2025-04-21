import base64
import hashlib

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def hash_image(image_path):
    with open(image_path, "rb") as image_file:
        return hashlib.sha256(image_file.read()).hexdigest()