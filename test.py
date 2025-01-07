import os
import tkinter as tk
from tkinter import filedialog


def rename_images():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")
    if not folder_path:
        print("No folder selected. Exiting...")
        return

    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".png")])

    for index, image in enumerate(images):
        old_path = os.path.join(folder_path, image)
        new_path = os.path.join(folder_path, f"{index}.png")
        os.rename(old_path, new_path)
        print(f"Renamed: {image} -> {index}.png")

    print("Renaming completed!")


if __name__ == "__main__":
    rename_images()
