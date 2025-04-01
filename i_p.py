import cv2
import easyocr
import numpy as np
import os
from IPython.display import display
import PIL.Image
from ultralytics import YOLO

class LicensePlateRecognition:
    def __init__(self, model_path, output_dir="output_img"):
        """
        Initializes the License Plate Recognition system.

        :param model_path: Path to the trained YOLO model.
        :param output_dir: Directory where processed images will be saved.
        """
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.reader = easyocr.Reader(['en'])

    @staticmethod
    def preprocess_plate(plate):
        """
        Converts the cropped license plate image to a preprocessed format.

        :param plate: Cropped license plate image.
        :return: Preprocessed binary image.
        """
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def show_image(image):
        """
        Displays an image in Jupyter Notebook.

        :param image: Image to display.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        display(PIL.Image.fromarray(image_rgb))

    def process_image(self, image_path):
        """
        Processes a single image: detects license plates, performs OCR, and saves the result.

        :param image_path: Path to the image file.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error loading {image_path}")
            return

        print(f"🔄 Processing: {image_path}")

        # Run YOLO model for detection
        results = self.model(image, conf=0.7)

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)

                # Ignore small detections
                if (x2 - x1) * (y2 - y1) < 500:
                    continue

                # Crop and preprocess license plate
                plate_image = image[y1:y2, x1:x2]
                processed_plate = self.preprocess_plate(plate_image)

                # Perform OCR
                text_results = self.reader.readtext(processed_plate, detail=0)
                plate_text = text_results[0] if text_results else "N/A"

                # Draw bounding box and overlay text
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 225), 3)
                cv2.putText(image, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 2)

        # Display processed image in Jupyter Notebook
        self.show_image(image)

        # Save the processed image
        output_image_path = os.path.join(self.output_dir, f"processed_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, image)
        print(f"✅ Saved: {output_image_path}\n")

    def process_folder(self, folder_path):
        """
        Processes all images in a given folder.

        :param folder_path: Path to the folder containing images.
        """
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print("❌ No images found in the folder.")
            return

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            self.process_image(image_path)

# Only run if this file is executed directly
if __name__ == "__main__":
    model_path = "model/yolov11_model_state_dict.pkl"  # Update this path
    folder_path = "image"  # Update this path

    plate_recognition = LicensePlateRecognition(model_path)
    plate_recognition.process_folder(folder_path)
