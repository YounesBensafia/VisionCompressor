from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QMessageBox,
    QFileDialog,
    QLineEdit,
    QHBoxLayout,
    QStackedWidget,
    QGroupBox,
)
from src.models.block_matching_worker import BlockMatchingWorker
from src.models.batch_processing_worker import BatchProcessingWorker
import cv2
import os
from PyQt6.QtCore import Qt
from src.main_BGR import create_video_from_reconstructed
from matplotlib import pyplot as plt

class BlockMatchingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.source_image = None
        self.target_image = None
        self.block_size = 16
        self.worker = None
        self.selection_image = None
        self.hover_pos = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("VisionCompressor - Block Matching")
        self.setMinimumSize(600, 500)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPushButton {
                background-color: #4a9eff;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 11px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5d8f;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QGroupBox {
                font-size: 12px;
                font-weight: bold;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #4a9eff;
            }
            QLineEdit {
                background-color: #3c3c3c;
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 10px;
            }
            QLineEdit:focus {
                border-color: #4a9eff;
            }
            QProgressBar {
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                text-align: center;
                background-color: #3c3c3c;
            }
            QProgressBar::chunk {
                background-color: #4a9eff;
                border-radius: 4px;
            }
            QLabel {
                color: #ffffff;
            }
        """)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.create_mode_selection_page()
        self.create_single_frame_page()
        self.create_batch_processing_page()

        self.stacked_widget.addWidget(self.mode_selection_page)
        self.stacked_widget.addWidget(self.single_frame_page)
        self.stacked_widget.addWidget(self.batch_processing_page)

        self.stacked_widget.setCurrentIndex(0)

    def create_mode_selection_page(self):
        self.mode_selection_page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)

        title_label = QLabel("🎯 VisionCompressor")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #4a9eff;
                margin-bottom: 10px;
            }
        """)

        subtitle_label = QLabel("Select Processing Mode")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #cccccc;
                margin-bottom: 30px;
            }
        """)

        single_frame_btn = QPushButton("🖼️ Single Frame Processing")
        batch_process_btn = QPushButton("📁 Batch Frame Processing")

        for btn in [single_frame_btn, batch_process_btn]:
            btn.setMinimumHeight(80)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    border: 2px solid #4a4a4a;
                    border-radius: 12px;
                    padding: 20px;
                    font-size: 14px;
                    font-weight: bold;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #4a9eff;
                    border-color: #4a9eff;
                    transform: translateY(-2px);
                }
                QPushButton:pressed {
                    background-color: #357abd;
                }
            """)

        single_frame_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        batch_process_btn.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(2)
        )

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addStretch()
        layout.addWidget(single_frame_btn)
        layout.addWidget(batch_process_btn)
        layout.addStretch()

        self.mode_selection_page.setLayout(layout)

    def create_single_frame_page(self):
        self.single_frame_page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        back_btn = QPushButton("← Back to Mode Selection")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 11px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        image_group = QGroupBox("📷 Image Selection")
        image_layout = QHBoxLayout()
        image_layout.setSpacing(15)

        self.load_source_btn = QPushButton("📁 Load Source Image")
        self.load_target_btn = QPushButton("🎯 Load Target Image")
        
        for btn in [self.load_source_btn, self.load_target_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4a9eff;
                    border-radius: 8px;
                    padding: 15px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #357abd;
                }
            """)
        
        self.load_source_btn.clicked.connect(lambda: self.load_image("source"))
        self.load_target_btn.clicked.connect(lambda: self.load_image("target"))

        image_layout.addWidget(self.load_source_btn)
        image_layout.addWidget(self.load_target_btn)
        image_group.setLayout(image_layout)

        process_group = QGroupBox("⚙️ Processing Options")
        process_layout = QVBoxLayout()
        process_layout.setSpacing(15)

        self.single_block_btn = QPushButton("🔍 Process Single Block")
        self.all_blocks_btn = QPushButton("🚀 Process All Blocks")
        
        for btn in [self.single_block_btn, self.all_blocks_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    border-radius: 8px;
                    padding: 15px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
                QPushButton:disabled {
                    background-color: #555555;
                    color: #888888;
                }
            """)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                text-align: center;
                background-color: #3c3c3c;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 6px;
            }
        """)

        self.single_block_btn.clicked.connect(self.process_single_block)
        self.all_blocks_btn.clicked.connect(self.process_all_blocks)

        process_layout.addWidget(self.single_block_btn)
        process_layout.addWidget(self.all_blocks_btn)
        process_layout.addWidget(self.progress_bar)
        process_group.setLayout(process_layout)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                border-radius: 6px;
                padding: 10px;
                font-size: 11px;
                color: #cccccc;
            }
        """)

        layout.addWidget(back_btn)
        layout.addWidget(image_group)
        layout.addWidget(process_group)
        layout.addWidget(self.status_label)
        layout.addStretch()

        self.single_block_btn.setEnabled(False)
        self.all_blocks_btn.setEnabled(False)

        self.single_frame_page.setLayout(layout)

    def create_batch_processing_page(self):
        self.batch_processing_page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        back_btn = QPushButton("← Back to Mode Selection")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 11px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        folder_group = QGroupBox("📂 Folder Selection")
        folder_layout = QVBoxLayout()
        folder_layout.setSpacing(15)

        input_layout = QHBoxLayout()
        self.input_folder_path = QLineEdit()
        self.input_folder_path.setPlaceholderText("Select input folder containing frames...")
        self.input_folder_path.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                padding: 12px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #4a9eff;
            }
        """)
        
        input_folder_btn = QPushButton("📁 Browse")
        input_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                border-radius: 8px;
                padding: 12px 20px;
                min-width: 100px;
            }
        """)
        input_folder_btn.clicked.connect(self.select_input_folder)
        input_layout.addWidget(self.input_folder_path)
        input_layout.addWidget(input_folder_btn)

        output_layout = QHBoxLayout()
        self.output_folder_path = QLineEdit()
        self.output_folder_path.setPlaceholderText("Select output folder for processed frames...")
        self.output_folder_path.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                padding: 12px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border-color: #4a9eff;
            }
        """)
        
        output_folder_btn = QPushButton("📁 Browse")
        output_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                border-radius: 8px;
                padding: 12px 20px;
                min-width: 100px;
            }
        """)
        output_folder_btn.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_folder_path)
        output_layout.addWidget(output_folder_btn)

        folder_layout.addLayout(input_layout)
        folder_layout.addLayout(output_layout)
        folder_group.setLayout(folder_layout)

        self.batch_progress = QProgressBar()
        self.batch_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                text-align: center;
                background-color: #3c3c3c;
                height: 30px;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 6px;
            }
        """)
        
        self.batch_status = QLabel("Ready to process")
        self.batch_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.batch_status.setStyleSheet("""
            QLabel {
                background-color: #3c3c3c;
                border-radius: 6px;
                padding: 12px;
                font-size: 12px;
                color: #cccccc;
            }
        """)
        
        self.process_frames_btn = QPushButton("🚀 Process All Frames")
        self.process_frames_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        self.process_frames_btn.clicked.connect(self.process_frames)
        self.process_frames_btn.setEnabled(False)

        self.create_video_btn = QPushButton("🎬 Create Video from Reconstructed Frames")
        self.create_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                border-radius: 8px;
                padding: 15px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.create_video_btn.clicked.connect(self.create_video)
        self.create_video_btn.setEnabled(True)

        layout.addWidget(back_btn)
        layout.addWidget(folder_group)
        layout.addWidget(self.batch_progress)
        layout.addWidget(self.batch_status)
        layout.addWidget(self.process_frames_btn)
        layout.addWidget(self.create_video_btn)
        layout.addStretch()

        self.batch_processing_page.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_path.setText(folder)
            self._check_batch_processing_ready()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_path.setText(folder)
            self._check_batch_processing_ready()

    def _check_batch_processing_ready(self):
        input_folder = self.input_folder_path.text()
        output_folder = self.output_folder_path.text()
        self.process_frames_btn.setEnabled(bool(input_folder and output_folder))

    def process_frames(self):
        input_folder = self.input_folder_path.text()
        output_folder = self.output_folder_path.text()

        self.process_frames_btn.setEnabled(False)
        self.batch_status.setText("Processing frames...")
        self.batch_progress.setValue(0)

        self.batch_worker = BatchProcessingWorker(input_folder, output_folder)
        self.batch_worker.progress.connect(self.update_batch_progress)
        self.batch_worker.finished.connect(self.on_batch_processing_finished)
        self.batch_worker.error.connect(self.on_batch_processing_error)
        self.batch_worker.start()

    def update_batch_progress(self, progress_info):
        percentage, status_text = progress_info
        self.batch_progress.setValue(percentage)
        self.batch_status.setText(status_text)

    def on_batch_processing_finished(self):
        self.batch_status.setText("Processing completed successfully!")
        self.batch_progress.setValue(100)
        self.process_frames_btn.setEnabled(True)
        self.create_video_btn.setEnabled(True)
        self.batch_worker = None

    def create_video(self):
        output_folder = self.output_folder_path.text()
        reconstructed_folder = os.path.join(output_folder, "reconstructed")
        output_video = os.path.join(output_folder, "reconstructed_video.avi")

        create_video_from_reconstructed(reconstructed_folder, output_video, fps=30)
        QMessageBox.information(self, "Video Created", f"Video saved as {output_video}")

    def on_batch_processing_error(self, error_message):
        self.batch_status.setText(f"Error: {error_message}")
        self.process_frames_btn.setEnabled(True)
        self.batch_worker = None
        QMessageBox.critical(self, "Error", error_message)

    def load_image(self, image_type):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {image_type.capitalize()} Image",
            "",
            "Images (*.png *.jpg *.jpeg)",
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
        if self.source_image is not None and self.target_image is not None:
            if self.source_image.shape == self.target_image.shape:
                self.single_block_btn.setEnabled(True)
                self.all_blocks_btn.setEnabled(True)
                self.status_label.setText("Ready to process")
            else:
                QMessageBox.warning(
                    self, "Error", "Images must have the same dimensions"
                )
                self.source_image = None
                self.target_image = None

    def process_single_block(self):
        if not self._validate_images():
            return

        cv2.namedWindow("Select Block")
        cv2.setMouseCallback("Select Block", self._select_block)
        cv2.imshow("Select Block", self.target_image)

        self.status_label.setText("Click on the image to select a block...")

    def _select_block(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyWindow("Select Block")

            self.worker = BlockMatchingWorker(
                self.source_image, self.target_image, self.block_size, False
            )
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

        self.worker = BlockMatchingWorker(
            self.source_image, self.target_image, self.block_size, True
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_all_blocks_finished)
        self.worker.start()

        self.single_block_btn.setEnabled(False)
        self.all_blocks_btn.setEnabled(False)

    def draw_grid(self, image):
        grid_image = image.copy()
        h, w = image.shape[:2]

        for x in range(0, w, self.block_size):
            cv2.line(grid_image, (x, 0), (x, h), (128, 128, 128), 1)

        for y in range(0, h, self.block_size):
            cv2.line(grid_image, (0, y), (w, y), (128, 128, 128), 1)

        if self.hover_pos:
            x, y = self.hover_pos
            x = (x // self.block_size) * self.block_size
            y = (y // self.block_size) * self.block_size
            cv2.rectangle(
                grid_image,
                (x, y),
                (x + self.block_size, y + self.block_size),
                (0, 255, 0),
                2,
            )

        return grid_image

    def _mouse_move(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_pos = (x, y)
            if self.selection_image is not None:
                display_image = self.draw_grid(self.target_image)
                cv2.imshow("Select Block", display_image)
        elif event == cv2.EVENT_LBUTTONDOWN:
            cv2.destroyWindow("Select Block")

            self.worker = BlockMatchingWorker(
                self.source_image, self.target_image, self.block_size, False
            )
            self.worker.set_selected_point((x, y))
            self.worker.finished.connect(self._on_single_block_finished)
            self.worker.start()

            self.status_label.setText("Processing selected block...")

    def process_single_block(self):
        if not self._validate_images():
            return

        self.selection_image = self.target_image.copy()

        cv2.namedWindow("Select Block")
        cv2.setMouseCallback("Select Block", self._mouse_move)

        display_image = self.draw_grid(self.target_image)
        cv2.imshow("Select Block", display_image)

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
            marked_image = self.target_image.copy()
            cv2.rectangle(
                marked_image,
                (x, y),
                (x + self.block_size, y + self.block_size),
                (0, 255, 0),
                2,
            )

            self._show_single_block_results(
                self.target_image[y : y + self.block_size, x : x + self.block_size],
                residual,
                reconstructed_block,
            )
            self.status_label.setText(
                f"Motion vector for block ({x}, {y}): {motion_vector}"
            )

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
        plt.imshow(residual_display, cmap="gray")
        plt.title("Residual")
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(cv2.cvtColor(reconstructed_block, cv2.COLOR_BGR2RGB))
        plt.title("Reconstructed Block")
        plt.axis("off")

        plt.show()

    def _show_all_blocks_results(self, residual_image, reconstructed_image):
        residual_image_8u = cv2.convertScaleAbs(residual_image)
        residual_display = cv2.cvtColor(residual_image_8u, cv2.COLOR_YCrCb2RGB)
        residual_display = cv2.cvtColor(residual_display, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("residual_image.png", residual_display)

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