from PyQt6.QtCore import QThread, pyqtSignal
import os
import cv2
from main_BGR import process_pframe

class BatchProcessingWorker(QThread):
    progress = pyqtSignal(tuple)
    error = pyqtSignal(str)

    def __init__(self, image_folder, output_folder):
        super().__init__()
        self.image_folder = image_folder
        self.output_folder = output_folder

    def run(self):
        try:
            residuals_folder, reconstructed_folder = self.create_output_directories()

            image_files = os.listdir(self.image_folder)
            try:
                image_files = sorted(
                    image_files, key=lambda x: int(os.path.splitext(x)[0])
                )
            except ValueError:
                self.error.emit(
                    "Image files must be named with numbers (e.g., 0.png, 1.png, etc.)"
                )
                return

            if not image_files:
                self.error.emit("No image files found in the input folder")
                return

            self.progress.emit((0, "Processing first frame..."))
            first_image_path = os.path.join(self.image_folder, image_files[0])
            first_image = cv2.imread(first_image_path)
            if first_image is None:
                self.error.emit(f"Could not read image: {first_image_path}")
                return

            first_reconstructed_path = os.path.join(
                reconstructed_folder, "reconstructed_0.png"
            )
            cv2.imwrite(first_reconstructed_path, first_image)

            total_frames = len(image_files) - 1
            for i in range(total_frames):
                progress = int((i + 1) / total_frames * 100)
                self.progress.emit(
                    (progress, f"Processing frame {i + 1} of {total_frames}...")
                )

                source_image_path = os.path.join(
                    reconstructed_folder, f"reconstructed_{i}.png"
                )
                target_image_path = os.path.join(self.image_folder, image_files[i + 1])

                reconstructed_image, residual_image = process_pframe(
                    source_image_path, target_image_path
                )

                residual_image_8u = cv2.convertScaleAbs(residual_image)
                residual_display = cv2.cvtColor(residual_image_8u, cv2.COLOR_RGB2GRAY)

                residual_path, reconstructed_path = self.generate_file_paths(residuals_folder, reconstructed_folder, i)

                cv2.imwrite(residual_path, residual_display)
                cv2.imwrite(reconstructed_path, reconstructed_image)

            self.progress.emit((100, "Processing completed!"))

        except Exception as e:
            self.error.emit(f"An error occurred: {str(e)}")

    def generate_file_paths(self, residuals_folder, reconstructed_folder, i):
        residual_path = os.path.join(residuals_folder, f"residual_{i+1}.png")
        reconstructed_path = os.path.join(
                    reconstructed_folder, f"reconstructed_{i+1}.png"
                )
        
        return residual_path,reconstructed_path

    def create_output_directories(self):
        residuals_folder = os.path.join(self.output_folder, "residuals")
        reconstructed_folder = os.path.join(self.output_folder, "reconstructed")
        os.makedirs(residuals_folder, exist_ok=True)
        os.makedirs(reconstructed_folder, exist_ok=True)
        return residuals_folder,reconstructed_folder