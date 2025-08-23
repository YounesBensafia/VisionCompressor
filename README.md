# VisionCompressor

<img src="https://github.com/user-attachments/assets/900b26aa-973d-49f9-b092-25836ba542b4" alt="Preview" width="100%" />

A powerful Python-based image compression and reconstruction tool that provides efficient compression algorithms while maintaining image quality.

## Features

- **Multiple Processing Modes**:
  - Standard compression mode
  - BGR (Blue-Green-Red) channel processing
  - Y-channel only processing for enhanced efficiency
- **Interactive GUI** for easy image manipulation
- **Batch Processing** capability for multiple images
- **Visual Feedback** with residual image generation
- **Customizable Parameters** for compression quality

## Quick Start

### Prerequisites

- Python 3.x
- Required Python packages (install via requirements.txt)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YounesBensafia/VisionCompressor.git
   cd VisionCompressor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### GUI Interface

Launch the graphical interface for interactive compression:

```bash
python src/gui.py
```

### Command Line Interface

1. Standard compression:

   ```bash
   python src/main.py
   ```

2. BGR channel processing:

   ```bash
   python src/main_BGR.py
   ```

3. Y-channel only processing:
   ```bash
   python src/main_with_y_channel_only.py
   ```

## Project Structure

```
VisionCompressor/
├── src/               # Source code
│   ├── gui.py        # GUI implementation
│   ├── main.py       # Standard compression
│   ├── main_BGR.py   # BGR processing
│   └── main_with_y_channel_only.py
├── images/           # Input images directory
├── output/           # Output directory
│   ├── reconstructed/  # Reconstructed images
│   └── residuals/     # Residual images
└── requirements.txt  # Python dependencies
```

## Input/Output

- **Input Images**: Place your images in the `images/` directory
- **Output**:
  - Reconstructed images are saved in `output/reconstructed/`
  - Residual images are saved in `output/residuals/`

## Contributing

Contributions are welcome! Feel free to:

- Open issues for bugs or enhancement suggestions
- Submit pull requests with improvements
- Share feedback about the tool

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
