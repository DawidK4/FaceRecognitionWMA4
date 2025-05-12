from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import os

download_dir = r'C:\Users\dawid\Desktop\FaceRecognitionWMA4\organized_faces\Aaron_Eckhart'
image_size = (200, 200)
max_images = 200

google_crawler = GoogleImageCrawler(storage={'root_dir': download_dir})
google_crawler.crawl(keyword='Aaron Eckhart face', max_num=max_images)

print("Resizing downloaded images...")
for filename in os.listdir(download_dir):
    file_path = os.path.join(download_dir, filename)
    try:
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img = img.resize(image_size)
            img.save(file_path)
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        os.remove(file_path) 

print("Done! Images are ready in 'faces/Aaron_Eckhart'.")
