import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading

class ModernGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x600")
        self.root.title("Modern Interface")
        self.root.configure(bg='#f0f4f8')
        
        # Center window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2
        self.root.geometry(f"800x600+{x}+{y}")
        
        # Custom styles
        self.style = ttk.Style()
        self.style.theme_use('default')
        
        # Configure styles
        self.style.configure('Modern.TFrame', background='#f0f4f8')
        self.style.configure('Modern.TButton',
                           padding=20,
                           font=('Helvetica', 12),
                           background='#4a90e2',
                           foreground='white')
        self.style.configure('Title.TLabel',
                           font=('Helvetica', 24, 'bold'),
                           background='#f0f4f8',
                           foreground='#2c3e50')
        self.style.configure('Subtitle.TLabel',
                           font=('Helvetica', 16),
                           background='#f0f4f8',
                           foreground='#34495e')
        
        # Mouse hover effects
        self.style.map('Modern.TButton',
                      background=[('active', '#357abd')],
                      relief=[('pressed', 'groove'),
                             ('!pressed', 'ridge')])
        
        self.setup_frames()
        self.show_welcome()
    
    def setup_frames(self):
        # Welcome Frame
        self.welcome_frame = ttk.Frame(self.root, style='Modern.TFrame')
        ttk.Label(self.welcome_frame,
                 text="Welcome",
                 style='Title.TLabel').pack(pady=40)
        ttk.Label(self.welcome_frame,
                 text="Click Start to begin your journey",
                 style='Subtitle.TLabel').pack(pady=20)
        ttk.Button(self.welcome_frame,
                  text="Start",
                  command=self.show_options,
                  style='Modern.TButton',
                  width=20).pack(pady=30)
        
        # Options Frame
        self.options_frame = ttk.Frame(self.root, style='Modern.TFrame')
        ttk.Label(self.options_frame,
                 text="Choose Your Option",
                 style='Title.TLabel').pack(pady=40)
        
        buttons_frame = ttk.Frame(self.options_frame, style='Modern.TFrame')
        buttons_frame.pack(pady=20)
        
        for option, text in [('A', 'Process Data'), ('B', 'View Statistics')]:
            button_container = ttk.Frame(buttons_frame, style='Modern.TFrame')
            button_container.pack(side=tk.LEFT, padx=20)
            
            ttk.Button(button_container,
                      text=option,
                      command=self.process_option_a if option == 'A' else None,
                      style='Modern.TButton',
                      width=15).pack(pady=5)
            ttk.Label(button_container,
                     text=text,
                     style='Subtitle.TLabel').pack()
        
        # Results Frame
        self.results_frame = ttk.Frame(self.root, style='Modern.TFrame')
        self.result_var = tk.StringVar()
        
        ttk.Label(self.results_frame,
                 text="Results",
                 style='Title.TLabel').pack(pady=40)
        
        self.result_label = ttk.Label(self.results_frame,
                                    textvariable=self.result_var,
                                    style='Subtitle.TLabel')
        self.result_label.pack(pady=20)
        
        ttk.Button(self.results_frame,
                  text="Try Again",
                  command=self.show_welcome,
                  style='Modern.TButton',
                  width=20).pack(pady=30)
        
        # Progress bar for processing
        self.progress = ttk.Progressbar(self.results_frame,
                                      mode='indeterminate',
                                      length=300)
    
    def show_frame(self, frame):
        for f in (self.welcome_frame, self.options_frame, self.results_frame):
            f.pack_forget()
        frame.pack(expand=True, fill='both')
    
    def show_welcome(self):
        self.show_frame(self.welcome_frame)
    
    def show_options(self):
        self.show_frame(self.options_frame)
    
    def process_option_a(self):
        self.show_frame(self.results_frame)
        self.result_var.set("Processing...")
        self.progress.pack(pady=20)
        self.progress.start(10)
        
        def process():
            time.sleep(2)  # Simulate processing
            result = random.randint(1000, 9999)
            self.root.after(0, self.show_result, result)
        
        threading.Thread(target=process, daemon=True).start()
    
    def show_result(self, result):
        self.progress.stop()
        self.progress.pack_forget()
        self.result_var.set(f"Processing complete!\nGenerated ID: {result}")
    
    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    app = ModernGUI()
    app.run()