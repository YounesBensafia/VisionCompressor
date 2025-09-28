from PyQt6.QtCore import QThread, pyqtSignal
from src.services.process_block import process_block
import cv2
import numpy as np

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

        motion_vector, residual = process_block(
            self.source_image, self.target_image, x, y
        )

        if motion_vector is not None:
            source_x = x + motion_vector[0]
            source_y = y + motion_vector[1]
            best_match_block = self.source_image[
                source_y : source_y + self.block_size,
                source_x : source_x + self.block_size,
            ]

            best_match_ycrcb = cv2.cvtColor(best_match_block, cv2.COLOR_BGR2YCrCb)
            best_match_y = best_match_ycrcb[:, :, 0]

            residual_y = residual[:, :, 0]
            reconstructed_y = np.clip(
                best_match_y.astype(np.int16) + residual_y, 0, 255
            ).astype("uint8")

            reconstructed_ycrcb = best_match_ycrcb.copy()
            reconstructed_ycrcb[:, :, 0] = reconstructed_y

            reconstructed_block = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)

            self.finished.emit((motion_vector, residual, reconstructed_block, (x, y)))

    def _process_all_blocks(self):
        h, w, _ = self.target_image.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        padded_target = cv2.copyMakeBorder(
            self.target_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )
        padded_source = cv2.copyMakeBorder(
            self.source_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

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

                motion_vector, residual = process_block(
                    padded_source, padded_target, x, y
                )

                if motion_vector is not None:
                    motion_vectors.append(motion_vector)
                    residual_image[y : y + self.block_size, x : x + self.block_size] = (
                        residual
                    )

                    motion_vector_x, motion_vector_y = motion_vector
                    source_x = x + motion_vector_x
                    source_y = y + motion_vector_y

                    best_match_block = padded_source[
                        source_y : source_y + self.block_size,
                        source_x : source_x + self.block_size,
                    ]
                    best_match_ycrcb = cv2.cvtColor(
                        best_match_block, cv2.COLOR_BGR2YCrCb
                    )

                    reconstructed_ycrcb = best_match_ycrcb.astype(np.int16) + residual
                    reconstructed_ycrcb = np.clip(reconstructed_ycrcb, 0, 255).astype(
                        "uint8"
                    )
                    reconstructed_block = cv2.cvtColor(
                        reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR
                    )

                    reconstructed_image[
                        y : y + self.block_size, x : x + self.block_size
                    ] = reconstructed_block

                processed_blocks += 1
                self.progress.emit(int((processed_blocks / total_blocks) * 100))

        self.finished.emit((motion_vectors, residual_image, reconstructed_image, None))
