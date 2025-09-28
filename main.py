import sys
from PyQt6.QtWidgets import (QApplication)
from src.models.block_matching_gui import BlockMatchingGUI


def main():
    app = QApplication(sys.argv)
    gui = BlockMatchingGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
