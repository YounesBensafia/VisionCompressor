import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def mse(imageA, imageB):
    err = np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)
    return err

def get_box_cor_from_center(center):
    x, y = center
    return ((int(x - 8), int(y - 8)), (int(x + 8), int(y + 8)))

def process_block(source_image, target_image, block_x, block_y, block_size=16, search_padding=64, threshold=24):
    """
    Traite un seul bloc et retourne le vecteur de mouvement et le résiduel
    """
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

class BlockMatchingGUI:
    def __init__(self):
        self.source_image = None
        self.target_image = None
        self.block_size = 16
        self.selected_point = None
        
        # Créer la fenêtre principale
        self.root = tk.Tk()
        self.root.title("Block Matching")
        self.root.geometry("300x200")
        
        # Créer les widgets
        ttk.Label(self.root, text="Choisissez le mode:").pack(pady=20)
        
        ttk.Button(self.root, text="1. Bloc unique", 
                  command=self.process_single_block).pack(pady=10)
        
        ttk.Button(self.root, text="2. Tous les blocs", 
                  command=self.process_all_blocks).pack(pady=10)
        
        ttk.Button(self.root, text="Quitter", 
                  command=self.root.destroy).pack(pady=10)
        
    def select_block(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x = x
            self.mouse_y = y
            self.selected_point = (x, y)
            cv2.destroyWindow('Select Block')

    def process_single_block(self):
        if not self.load_images():
            return
            
        self.root.withdraw()
        
        cv2.namedWindow('Select Block')
        cv2.setMouseCallback('Select Block', self.select_block)
        
        print("Cliquez sur l'image pour sélectionner un bloc...")
        cv2.imshow('Select Block', self.target_image)
        
        while self.selected_point is None:
            if cv2.waitKey(1) & 0xFF == 27:  # Échap pour quitter
                cv2.destroyAllWindows()
                self.root.destroy()
                return
                
        x, y = self.selected_point
        x = (x // self.block_size) * self.block_size
        y = (y // self.block_size) * self.block_size
        
        motion_vector, residual = process_block(self.source_image, self.target_image, x, y)
        
        if motion_vector is not None:
            print(f"Vecteur de mouvement pour le bloc ({x}, {y}): {motion_vector}")
            
            # Reconstruction du bloc
            source_x = x + motion_vector[0]
            source_y = y + motion_vector[1]
            best_match_block = self.source_image[source_y:source_y + self.block_size, source_x:source_x + self.block_size]

            # Convertir le bloc en YCrCb pour ajouter le résiduel
            best_match_ycrcb = cv2.cvtColor(best_match_block, cv2.COLOR_BGR2YCrCb)
            best_match_y = best_match_ycrcb[:, :, 0]
            
            residual_y = residual[:, :, 0]
            reconstructed_y = np.clip(best_match_y.astype(np.int16) + residual_y, 0, 255).astype("uint8")
            
            reconstructed_ycrcb = best_match_ycrcb.copy()
            reconstructed_ycrcb[:, :, 0] = reconstructed_y
            
            reconstructed_block = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)
            
            # Afficher le bloc original, l'image résiduelle et l'image reconstruite
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(self.target_image[y:y + self.block_size, x:x + self.block_size], cv2.COLOR_BGR2RGB))
            plt.title("Bloc Original")
            plt.axis("off")
            
            plt.subplot(132)
            residual_image = cv2.convertScaleAbs(residual)
            residual_display = cv2.cvtColor(residual_image, cv2.COLOR_YCrCb2RGB)
            residual_display = cv2.cvtColor(residual_display, cv2.COLOR_RGB2GRAY)
            

            plt.imshow(residual_display, cmap='gray')
            plt.title("Résiduel")
            plt.axis("off")
            
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(reconstructed_block, cv2.COLOR_BGR2RGB))
            plt.title("Bloc Reconstruit")
            plt.axis("off")
            
            plt.show()
        
        cv2.destroyAllWindows()
        self.root.destroy()

        
        
    

    def load_images(self):
        try:
            self.source_image = cv2.imread("images/1.jpg")
            self.target_image = cv2.imread("images/2.jpg")
            
            if self.source_image is None or self.target_image is None:
                messagebox.showerror("Erreur", "Impossible de charger les images")
                return False
                
            if self.source_image.shape != self.target_image.shape:
                messagebox.showerror("Erreur", "Les images doivent avoir les mêmes dimensions")
                return False
                
            return True
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement des images: {str(e)}")
            return False

    # Ajout de la reconstruction dans process_all_blocks
    def process_all_blocks(self):
        """
        Traite tous les blocs de l'image et reconstruit l'image
        """
        if not self.load_images():
            return
            
        self.root.withdraw()  # Cache la fenêtre principale
        
        print("Traitement de tous les blocs en cours...")
        
        h, w, _ = self.target_image.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        padded_target = cv2.copyMakeBorder(self.target_image, 0, pad_h, 0, pad_w, 
                                        cv2.BORDER_CONSTANT, value=0)
        padded_source = cv2.copyMakeBorder(self.source_image, 0, pad_h, 0, pad_w, 
                                        cv2.BORDER_CONSTANT, value=0)
        
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

                    # Reconstruction du bloc cible
                    motion_vector_x, motion_vector_y = motion_vector
                    source_x = x + motion_vector_x
                    source_y = y + motion_vector_y
                    
                    best_match_block = padded_source[
                        source_y : source_y + self.block_size, source_x : source_x + self.block_size
                    ]
                    best_match_ycrcb = cv2.cvtColor(best_match_block, cv2.COLOR_BGR2YCrCb)
                    
                    # Ajouter le résiduel pour reconstruire le bloc
                    reconstructed_ycrcb = best_match_ycrcb.astype(np.int16) + residual
                    reconstructed_ycrcb = np.clip(reconstructed_ycrcb, 0, 255).astype("uint8")
                    reconstructed_block = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)
                    
                    reconstructed_image[y:y+self.block_size, x:x+self.block_size] = reconstructed_block
                
                processed_blocks += 1
                print(f"Progression: {processed_blocks}/{total_blocks} blocs traités", end='\r')
        
        print("\nTraitement terminé!")
        
        # Sauvegarder et afficher l'image résiduelle
        residual_image_8u = cv2.convertScaleAbs(residual_image)
        residual_display = cv2.cvtColor(residual_image_8u, cv2.COLOR_YCrCb2RGB)
        residual_display = cv2.cvtColor(residual_display, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("residual_image.png", residual_display)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(residual_display, cmap="gray")
        plt.title("Image Résiduelle")
        plt.axis("off")
        
        # Sauvegarder et afficher l'image reconstruite
        cv2.imwrite("reconstructed_image.png", reconstructed_image)
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
        plt.title("Image Reconstituée")
        plt.axis("off")
        plt.show()
        
        cv2.destroyAllWindows()
        self.root.destroy()


    def run(self):
        """
        Lance l'interface graphique
        """
        self.root.mainloop()

def main():
    try:
        app = BlockMatchingGUI()
        app.run()
    except Exception as e:
        print(f"Une erreur s'est produite: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()