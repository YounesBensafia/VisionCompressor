import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def mse(imageA, imageB):
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err


def process_pframe(
    source_image_path, target_image_path, threshold=24, block_size=16, padding=64
):
    # Load the images
    # =============================================#
    source_image = cv2.imread(source_image_path)  #
    target_image = cv2.imread(target_image_path)  #
    # ==============================================

    # * Ensure the images are of the same dimensions
    # *==========================================================================================================#
    # *==========================================================================================================#

    # * Pad the target image to make its dimensions divisible by 16 **AJUSTEMENT**
    # *========================================================================================================
    h, w, _ = target_image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded_target = cv2.copyMakeBorder(
        target_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
    )
    h, w, _ = source_image.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded_source = cv2.copyMakeBorder(
        source_image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
    )

    print(
        pad_h,
        pad_w,
        target_image.shape,
        source_image.shape,
        padded_target.shape,
        padded_source.shape,
    )
    assert (
        padded_target.shape == padded_source.shape
    ), "Source and target images must have the same dimensions."

    # * SO HERE WE HAVE AN IMAGE DIVISIBLE BY 16
    # Divide the padded target image into 16x16 blocks
    blocks_vertical = padded_target.shape[0] // block_size
    blocks_horizontal = padded_target.shape[1] // block_size

    residual_image = np.zeros_like(padded_target, dtype=np.int16)
    motion_vectors = []

    def get_box_cor_from_center(center):
        x, y = center
        return (
            (int(x - block_size // 2), int(y - block_size // 2)),
            (int(x + block_size // 2), int(y + block_size // 2)),
        )

    # Process each block
    for by in range(blocks_vertical):
        for bx in range(blocks_horizontal):
            # Extract the target block
            y = by * block_size
            x = bx * block_size
            target_block = padded_target[y : y + block_size, x : x + block_size]

            # Define search area in the source image
            search_area_x1 = max(0, x - padding)
            search_area_y1 = max(0, y - padding)
            search_area_x2 = min(padded_source.shape[1], x + block_size + padding)
            search_area_y2 = min(padded_source.shape[0], y + block_size + padding)

            search_area = padded_source[
                search_area_y1:search_area_y2, search_area_x1:search_area_x2
            ]

            # Convert to YCRCB and use Y channel
            target_block_ycrcb = cv2.cvtColor(target_block, cv2.COLOR_BGR2YCrCb)
            search_area_ycrcb = cv2.cvtColor(search_area, cv2.COLOR_BGR2YCrCb)
            target_y = target_block_ycrcb[:, :, 0]
            search_area_y = search_area_ycrcb[:, :, 0]

            # Sliding window search for best match
            min_mse = float("inf")
            best_match_position = None
            best_residual = None

            step = 32
            h, w = search_area.shape[:2]

            p1 = int(h // 2), int(w // 2)

            while step > 1:
                points = []
                x1, y1 = p1

                points.append((x1, y1))
                points.append((x1 - step, y1 - step))
                points.append((x1, y1 - step))
                points.append((x1 + step, y1 - step))
                points.append((x1 - step, y1))
                points.append((x1 + step, y1))
                points.append((x1 - step, y1 + step))
                points.append((x1, y1 + step))
                points.append((x1 + step, y1 + step))

                for p in points:
                    box = get_box_cor_from_center(p)

                    window_y = search_area_y[
                        box[0][1] : box[1][1], box[0][0] : box[1][0]
                    ]
                    window = search_area_ycrcb[
                        box[0][1] : box[1][1], box[0][0] : box[1][0]
                    ]
                    if window_y.shape != target_y.shape:
                        continue
                    error = mse(target_y, window_y)
                    if error < min_mse:
                        min_mse = error
                        best_match_position = (
                            box[0][0] + search_area_x1,
                            box[0][1] + search_area_y1,
                        )
                        best_residual = target_block_ycrcb.astype(
                            np.int16
                        ) - window.astype(np.int16)
                        p1 = p
                step = step / 2

            # Store the motion vector
            if best_match_position is not None:
                motion_vector_x = best_match_position[0] - x
                motion_vector_y = best_match_position[1] - y
                motion_vectors.append((motion_vector_x, motion_vector_y))

                if min_mse < threshold:
                    residual_image[y : y + block_size, x : x + block_size] = 0
                else:
                    residual_image[y : y + block_size, x : x + block_size] = (
                        best_residual
                    )

    # Display and save the residual image
    # residual_image_8u = cv2.convertScaleAbs(residual_image)
    # residual_display = cv2.cvtColor(residual_image_8u, cv2.COLOR_YCrCb2RGB)
    # residual_display = cv2.cvtColor(residual_display, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite("residual_image.png", residual_display)
    # plt.imshow(residual_display, cmap="gray")
    # plt.title("Residual Image")
    # plt.axis("off")
    # plt.show()

    # Reconstruct the target image using the source image, motion vectors, and residuals
    reconstructed_image = np.zeros_like(padded_source)
    for by in range(blocks_vertical):
        for bx in range(blocks_horizontal):
            y = by * block_size
            x = bx * block_size
            target_block = padded_target[y : y + block_size, x : x + block_size]

            motion_vector_x, motion_vector_y = motion_vectors[
                by * blocks_horizontal + bx
            ]
            source_x = x + motion_vector_x
            source_y = y + motion_vector_y

            best_match_block = padded_source[
                source_y : source_y + block_size, source_x : source_x + block_size
            ]

            # Convert best match block to YCRCB and extract Y channel
            best_match_ycrcb = cv2.cvtColor(best_match_block, cv2.COLOR_BGR2YCrCb)
            best_match_y = best_match_ycrcb[:, :, 0]

            # Add residual to reconstruct the target block
            residual = residual_image[y : y + block_size, x : x + block_size]
            reconstructed_ycrcb = best_match_ycrcb.astype(np.int16) + residual
            reconstructed_ycrcb = np.clip(reconstructed_ycrcb, 0, 255).astype("uint8")

            # Convert back to BGR and place in reconstructed image
            reconstructed_block = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)
            reconstructed_image[y : y + block_size, x : x + block_size] = (
                reconstructed_block
            )

    return reconstructed_image, residual_image
    # Save and display the reconstructed image
    # cv2.imwrite("reconstructed_image.png", reconstructed_image)
    # plt.imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
    # plt.title("Reconstructed Image")
    # plt.axis("off")
    # plt.show()


def process_all_frames(image_folder="images", output_folder="output"):

    image_files = os.listdir(image_folder)
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    # Define output directories
    residuals_folder = os.path.join(output_folder, "residuals")
    reconstructed_folder = os.path.join(output_folder, "reconstructed")
    os.makedirs(residuals_folder, exist_ok=True)
    os.makedirs(reconstructed_folder, exist_ok=True)

    # Save the first frame in the reconstructed folder as it is without processing
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    first_reconstructed_image_path = os.path.join(
        reconstructed_folder, "reconstructed_0.png"
    )

    # Save the first image
    cv2.imwrite(first_reconstructed_image_path, first_image)

    for i in range(len(image_files) - 1):
        source_image_path = os.path.join(
            reconstructed_folder, f"reconstructed_{i+1}.png"
        )
        target_image_path = os.path.join(image_folder, image_files[i + 1])
        reconstructed_image, residual_image = process_pframe(
            source_image_path, target_image_path
        )

        # Display and save the residual image
        residual_image_8u = cv2.convertScaleAbs(residual_image)
        residual_display = cv2.cvtColor(residual_image_8u, cv2.COLOR_YCrCb2RGB)
        residual_display = cv2.cvtColor(residual_display, cv2.COLOR_RGB2GRAY)

        residual_image_path = os.path.join(residuals_folder, f"residual_{i+1}.png")
        reconstructed_image_path = os.path.join(
            reconstructed_folder, f"reconstructed_{i+1}.png"
        )

        # Save the residual image
        cv2.imwrite(residual_image_path, residual_display)

        # Save the reconstructed image
        cv2.imwrite(reconstructed_image_path, reconstructed_image)


def create_video_from_reconstructed(
    reconstructed_folder="output/reconstructed",
    output_video="reconstructed_video.avi",
    fps=30,
):
    image_files = sorted(
        os.listdir(reconstructed_folder),
        key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]),
    )

    if not image_files:
        print("No reconstructed images found.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(reconstructed_folder, image_files[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI format
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(reconstructed_folder, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

        # Display the frame
        cv2.imshow("Reconstructed Video", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):  # Press 'q' to quit early
            break

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_video}")


# Call the function
# create_video_from_reconstructed()
