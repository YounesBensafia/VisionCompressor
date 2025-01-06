import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QProgressBar, QMessageBox, 
                           QFileDialog, QHBoxLayout, QGroupBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap



def mse(imageA, imageB):
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def get_box_cor_from_center(center):
    x, y = center
    return ((int(x - 8), int(y - 8)), (int(x + 8), int(y + 8)))

def process_block(source_image, target_image, block_x, block_y, block_size=16, search_padding=64, threshold=24):
    # Extraire le bloc cible
    target_block = target_image[block_y:block_y + block_size, block_x:block_x + block_size]
    
    # Définir la zone de recherche
    search_area_x1 = max(0, block_x - search_padding)
    search_area_y1 = max(0, block_y - search_padding)
    search_area_x2 = min(source_image.shape[1], block_x + block_size + search_padding)
    search_area_y2 = min(source_image.shape[0], block_y + block_size + search_padding)
    
    search_area = source_image[search_area_y1:search_area_y2, search_area_x1:search_area_x2]
    
    # Convertir en YCrCb
    target_block_ycrcb = cv2.cvtColor(target_block, cv2.COLOR_BGR2YCrCb)
    search_area_ycrcb = cv2.cvtColor(search_area, cv2.COLOR_BGR2YCrCb)
    target_y = target_block_ycrcb[:, :, 0]
    search_area_y = search_area_ycrcb[:, :, 0]
    
    # Recherche du meilleur match
    min_mse = float('inf')
    best_match_position = None
    best_residual = None
    
    step = 32
    h, w = search_area.shape[:2]
    p1 = int(h // 2), int(w // 2)
    
    while step > 1:
        x1, y1 = p1
        points = [
            (x1, y1),
            (x1 - step, y1 - step),
            (x1, y1 - step),
            (x1 + step, y1 - step),
            (x1 - step, y1),
            (x1 + step, y1),
            (x1 - step, y1 + step),
            (x1, y1 + step),
            (x1 + step, y1 + step)
        ]
        
        for p in points:
            box = get_box_cor_from_center(p)
            
            # Vérifier que la boîte est dans les limites
            if (box[0][1] < 0 or box[0][0] < 0 or 
                box[1][1] > search_area.shape[0] or 
                box[1][0] > search_area.shape[1]):
                continue
                
            window_y = search_area_y[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            window = search_area_ycrcb[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            
            if window_y.shape != target_y.shape:
                continue
                
            error = mse(target_y, window_y)
            if error < min_mse:
                min_mse = error
                best_match_position = (
                    box[0][0] + search_area_x1,
                    box[0][1] + search_area_y1
                )
                best_residual = target_block_ycrcb.astype(np.int16) - window.astype(np.int16)
                p1 = p
        
        step = step / 2
    
    if best_match_position is not None:
        motion_vector = (
            best_match_position[0] - block_x,
            best_match_position[1] - block_y
        )
        
        if min_mse < threshold:
            residual = np.zeros_like(best_residual)
        else:
            residual = best_residual
            
        return motion_vector, residual
    
    return None, None

class BlockMatchingWorker(QThread):
    """Worker thread for processing blocks to avoid UI freezing"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(tuple)
    
    def __init__(self, source_image, target_image, block_size, process_all=False):
        super().__init__()
        self.source_image = source_image
        self.target_image = target_image
        self.block_size = block_size
        self.process_all = process_all
        self.selected_point = None
        
    def set_selected_point(self, point):
        self.selected_point = point
        
    def run(self):
        if self.process_all:
            self._process_all_blocks()
        else:
            self._process_single_block()
            
    def _process_single_block(self):
        if self.selected_point is None:
            return
            
        x, y = self.selected_point
        x = (x // self.block_size) * self.block_size
        y = (y // self.block_size) * self.block_size
        
        motion_vector, residual = process_block(self.source_image, self.target_image, x, y)
        
        if motion_vector is not None:
            source_x = x + motion_vector[0]
            source_y = y + motion_vector[1]
            best_match_block = self.source_image[source_y:source_y + self.block_size, source_x:source_x + self.block_size]
            
            best_match_ycrcb = cv2.cvtColor(best_match_block, cv2.COLOR_BGR2YCrCb)
            best_match_y = best_match_ycrcb[:, :, 0]
            
            residual_y = residual[:, :, 0]
            reconstructed_y = np.clip(best_match_y.astype(np.int16) + residual_y, 0, 255).astype("uint8")
            
            reconstructed_ycrcb = best_match_ycrcb.copy()
            reconstructed_ycrcb[:, :, 0] = reconstructed_y
            
            reconstructed_block = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)
            
            self.finished.emit((motion_vector, residual, reconstructed_block, (x, y)))
            
    def _process_all_blocks(self):
        h, w, _ = self.target_image.shape
        
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        
        padded_target = cv2.copyMakeBorder(self.target_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        padded_source = cv2.copyMakeBorder(self.source_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        
        blocks_vertical = padded_target.shape[0] // self.block_size
        blocks_horizontal = padded_target.shape[1] // self.block_size
        
        residual_image = np.zeros_like(padded_target, dtype=np.int16)
        reconstructed_image = np.zeros_like(padded_target)
        
        motion_vectors = []
        
        total_blocks = blocks_vertical * blocks_horizontal
        processed_blocks = 0
        
        for by in range(blocks_vertical):
            for bx in range(blocks_horizontal):
                y = by * self.block_size
                x = bx * self.block_size
                
                motion_vector, residual = process_block(padded_source, padded_target, x, y)
                
                if motion_vector is not None:
                    motion_vectors.append(motion_vector)
                    residual_image[y:y+self.block_size, x:x+self.block_size] = residual
                    
                    motion_vector_x, motion_vector_y = motion_vector
                    source_x = x + motion_vector_x
                    source_y = y + motion_vector_y
                    
                    best_match_block = padded_source[source_y:source_y + self.block_size, source_x:source_x + self.block_size]
                    best_match_ycrcb = cv2.cvtColor(best_match_block, cv2.COLOR_BGR2YCrCb)
                    
                    reconstructed_ycrcb = best_match_ycrcb.astype(np.int16) + residual
                    reconstructed_ycrcb = np.clip(reconstructed_ycrcb, 0, 255).astype("uint8") 
                    reconstructed_block = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)
                    
                    reconstructed_image[y:y+self.block_size, x:x+self.block_size] = reconstructed_block
                
                processed_blocks += 1
                self.progress.emit(int((processed_blocks / total_blocks) * 100))
                
        self.finished.emit((motion_vectors, residual_image, reconstructed_image, None))

class BlockMatchingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.source_image = None
        self.target_image = None
        self.block_size = 16
        self.worker = None
        self.selection_image = None  # Store the image with grid
        self.hover_pos = None  # Store current hover position
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Block Matching")
        self.setMinimumSize(500, 400)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Image selection group
        image_group = QGroupBox("Image Selection")
        image_layout = QHBoxLayout()
        
        self.load_source_btn = QPushButton("Load Source Image")
        self.load_target_btn = QPushButton("Load Target Image")
        self.load_source_btn.clicked.connect(lambda: self.load_image("source"))
        self.load_target_btn.clicked.connect(lambda: self.load_image("target"))
        
        image_layout.addWidget(self.load_source_btn)
        image_layout.addWidget(self.load_target_btn)
        image_group.setLayout(image_layout)
        
        # Processing options group
        process_group = QGroupBox("Processing Options")
        process_layout = QVBoxLayout()
        
        self.single_block_btn = QPushButton("Process Single Block")
        self.all_blocks_btn = QPushButton("Process All Blocks")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.single_block_btn.clicked.connect(self.process_single_block)
        self.all_blocks_btn.clicked.connect(self.process_all_blocks)
        
        process_layout.addWidget(self.single_block_btn)
        process_layout.addWidget(self.all_blocks_btn)
        process_layout.addWidget(self.progress_bar)
        process_group.setLayout(process_layout)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add all widgets to main layout
        main_layout.addWidget(image_group)
        main_layout.addWidget(process_group)
        main_layout.addWidget(self.status_label)
        main_layout.addStretch()
        
        # Initialize buttons as disabled
        self.single_block_btn.setEnabled(False)
        self.all_blocks_btn.setEnabled(False)
        
    def load_image(self, image_type):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {image_type.capitalize()} Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )
        
        if file_name:
            if image_type == "source":
                self.source_image = cv2.imread(file_name)
                self.status_label.setText("Source image loaded successfully")
            else:
                self.target_image = cv2.imread(file_name)
                self.status_label.setText("Target image loaded successfully")
                
            self._check_images()
            
    def _check_images(self):
        """Check if both images are loaded and have the same dimensions"""
        if self.source_image is not None and self.target_image is not None:
            if self.source_image.shape == self.target_image.shape:
                self.single_block_btn.setEnabled(True)
                self.all_blocks_btn.setEnabled(True)
                self.status_label.setText("Ready to process")
            else:
                QMessageBox.warning(self, "Error", "Images must have the same dimensions")
                self.source_image = None
                self.target_image = None
                
    def process_single_block(self):
        if not self._validate_images():
            return
            
        # Create OpenCV window for block selection
        cv2.namedWindow('Select Block')
        cv2.setMouseCallback('Select Block', self._select_block)
        cv2.imshow('Select Block', self.target_image)
        
        self.status_label.setText("Click on the image to select a block...")
        
    def _select_block(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyWindow('Select Block')
            
            self.worker = BlockMatchingWorker(self.source_image, self.target_image, 
                                            self.block_size, False)
            self.worker.set_selected_point((x, y))
            self.worker.finished.connect(self._on_single_block_finished)
            self.worker.start()
            
            self.status_label.setText("Processing selected block...")
            
    def process_all_blocks(self):
        if not self._validate_images():
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing all blocks...")
        
        self.worker = BlockMatchingWorker(self.source_image, self.target_image, 
                                        self.block_size, True)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_all_blocks_finished)
        self.worker.start()
        
        # Disable buttons during processing
        self.single_block_btn.setEnabled(False)
        self.all_blocks_btn.setEnabled(False)
    def draw_grid(self, image):
        """Draw the grid on the image and highlight hovered block"""
        grid_image = image.copy()
        h, w = image.shape[:2]
        
        # Draw vertical lines
        for x in range(0, w, self.block_size):
            cv2.line(grid_image, (x, 0), (x, h), (128, 128, 128), 1)
            
        # Draw horizontal lines
        for y in range(0, h, self.block_size):
            cv2.line(grid_image, (0, y), (w, y), (128, 128, 128), 1)
            
        if self.hover_pos:
            x, y = self.hover_pos
            # Snap to grid
            x = (x // self.block_size) * self.block_size
            y = (y // self.block_size) * self.block_size
            # Draw highlighted block
            cv2.rectangle(grid_image, (x, y), 
                        (x + self.block_size, y + self.block_size), 
                        (0, 255, 0), 2)
            
        return grid_image

    def _mouse_move(self, event, x, y, flags, param):
        """Handle mouse movement to highlight blocks"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_pos = (x, y)
            if self.selection_image is not None:
                display_image = self.draw_grid(self.target_image)
                cv2.imshow('Select Block', display_image)
        elif event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyWindow('Select Block')
            
            self.worker = BlockMatchingWorker(self.source_image, self.target_image, 
                                            self.block_size, False)
            self.worker.set_selected_point((x, y))
            self.worker.finished.connect(self._on_single_block_finished)
            self.worker.start()
            
            self.status_label.setText("Processing selected block...")

    def process_single_block(self):
        if not self._validate_images():
            return
            
        # Create a copy of the target image for selection
        self.selection_image = self.target_image.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('Select Block')
        cv2.setMouseCallback('Select Block', self._mouse_move)
        
        # Show initial grid
        display_image = self.draw_grid(self.target_image)
        cv2.imshow('Select Block', display_image)
        
        self.status_label.setText("Hover over blocks and click to select one...")

    def _validate_images(self):
        if self.source_image is None or self.target_image is None:
            QMessageBox.warning(self, "Error", "Please load both images first")
            return False
        return True

    def _on_single_block_finished(self, result):
        motion_vector, residual, reconstructed_block, coords = result
        if motion_vector is not None:
            x, y = coords
            # Draw the selected block on the original image for visualization
            marked_image = self.target_image.copy()
            cv2.rectangle(marked_image, (x, y), 
                        (x + self.block_size, y + self.block_size), 
                        (0, 255, 0), 2)
            
            self._show_single_block_results(
                self.target_image[y:y + self.block_size, x:x + self.block_size],
                residual,
                reconstructed_block
            )
            self.status_label.setText(f"Motion vector for block ({x}, {y}): {motion_vector}")
        
        self.worker = None
        
    def _on_all_blocks_finished(self, result):
        motion_vectors, residual_image, reconstructed_image, _ = result
        self._show_all_blocks_results(residual_image, reconstructed_image)
        
        self.progress_bar.setVisible(False)
        self.single_block_btn.setEnabled(True)
        self.all_blocks_btn.setEnabled(True)
        self.status_label.setText("Processing completed!")
        self.worker = None
        
    def _show_single_block_results(self, original_block, residual, reconstructed_block):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(original_block, cv2.COLOR_BGR2RGB))
        plt.title("Original Block")
        plt.axis("off")
        
        plt.subplot(132)
        residual_image = cv2.convertScaleAbs(residual)
        residual_display = cv2.cvtColor(residual_image, cv2.COLOR_YCrCb2RGB)
        residual_display = cv2.cvtColor(residual_display, cv2.COLOR_RGB2GRAY)
        plt.imshow(residual_display, cmap='gray')
        plt.title("Residual")
        plt.axis("off")
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(reconstructed_block, cv2.COLOR_BGR2RGB))
        plt.title("Reconstructed Block")
        plt.axis("off")
        
        plt.show()
        
    def _show_all_blocks_results(self, residual_image, reconstructed_image):
        # Save and display residual image
        residual_image_8u = cv2.convertScaleAbs(residual_image)
        residual_display = cv2.cvtColor(residual_image_8u, cv2.COLOR_YCrCb2RGB)
        residual_display = cv2.cvtColor(residual_display, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("residual_image.png", residual_display)
        
        # Save reconstructed image
        cv2.imwrite("reconstructed_image.png", reconstructed_image)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(residual_display, cmap="gray")
        plt.title("Residual Image")
        plt.axis("off")
        
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
        plt.title("Reconstructed Image")
        plt.axis("off")
        plt.show()

def main():
    app = QApplication(sys.argv)
    gui = BlockMatchingGUI()
    gui.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()