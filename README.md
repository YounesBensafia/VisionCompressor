# VisionCompressor

<img src="https://github.com/user-attachments/assets/a5f4dafc-6901-44d7-be20-d9fa1ad045e9" 
     alt="Preview" 
     width="100%" 
     height="40%" />
     
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
python main.py
```

## Project Structure

```
VisionCompressor/
├── src/              # Source code
│   ├── gui.py        # GUI implementation
│   ├── main.py       # Standard compression
│   ├── main_BGR.py   # BGR processing
│   └── main_with_y_channel_only.py
├── images/           # Input images directory
│   ├── reconstructed/  # Reconstructed images
│   └── residuals/     # Residual images
├── tests/
│
└── pyproject.toml 
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
