import cv2
import easyocr
import os
from ultralytics import YOLO

class LicensePlateRecognitionVideo:
    def __init__(self, model_path, output_dir="output_videos"):
        """
        Initializes the License Plate Recognition for video processing.

        :param model_path: Path to the trained YOLO model.
        :param output_dir: Directory where processed videos will be saved.
        """
        self.model = YOLO(model_path, task="detect")  # Load YOLO model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.reader = easyocr.Reader(['en'])  # Initialize OCR reader

    def process_video(self, input_video, output_video="processed_video.mp4"):
        """
        Processes a video file: detects license plates and overlays detected text.

        :param input_video: Path to the input video file.
        :param output_video: Path where processed video will be saved.
        :return: Path of the saved video.
        """
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print(f"❌ Error: Unable to open video {input_video}")
            return None

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(self.output_dir, output_video)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO model for detection
            results = self.model(frame)

            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)

                    # Crop the detected license plate
                    plate_img = frame[y1:y2, x1:x2]

                    # Perform OCR if the cropped image is valid
                    plate_text = "N/A"
                    if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                        text_results = self.reader.readtext(plate_img, detail=0)
                        plate_text = text_results[0] if text_results else "N/A"

                    # Draw bounding box and overlay detected text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)  # Save processed frame

        cap.release()
        out.release()
        print(f"✅ Processed video saved at: {output_path}")
        return output_path

# Only run if this file is executed directly
if __name__ == "__main__":
    model_path = "model/best.pt"  # Update this with your YOLO model path
    input_video = "videos/input.mp4"  # Update this with your video file
    output_video = "processed_output.mp4"

    video_processor = LicensePlateRecognitionVideo(model_path)
    video_processor.process_video(input_video, output_video)
