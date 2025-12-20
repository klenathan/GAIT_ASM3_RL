import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import os

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RL Project Launcher")
        self.root.geometry("400x500")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Gridworld Section
        frame_gw = ttk.LabelFrame(root, text="Part I: Gridworld", padding=10)
        frame_gw.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(frame_gw, text="Level:").grid(row=0, column=0, sticky="w")
        self.gw_level = ttk.Combobox(frame_gw, values=[0, 1, 2, 3, 4, 5, 6], state="readonly")
        self.gw_level.current(0)
        self.gw_level.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_gw, text="Algorithm:").grid(row=1, column=0, sticky="w")
        self.gw_algo = ttk.Combobox(frame_gw, values=["q_learning", "sarsa"], state="readonly")
        self.gw_algo.current(0)
        self.gw_algo.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(frame_gw, text="Run Gridworld", command=self.run_gridworld).grid(row=2, column=0, columnspan=2, pady=10)

        # Arena Section
        frame_ar = ttk.LabelFrame(root, text="Part II: Arena (Deep RL)", padding=10)
        frame_ar.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(frame_ar, text="Control Style:").grid(row=0, column=0, sticky="w")
        self.ar_style = ttk.Combobox(frame_ar, values=["1 (Rot/Thrust)", "2 (Directional)"], state="readonly")
        self.ar_style.current(0)
        self.ar_style.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(frame_ar, text="Algorithm:").grid(row=1, column=0, sticky="w")
        self.ar_algo = ttk.Combobox(frame_ar, values=["ppo", "dqn"], state="readonly")
        self.ar_algo.current(0)
        self.ar_algo.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Button(frame_ar, text="Train Model", command=self.train_arena).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(frame_ar, text="Evaluate Model", command=self.eval_arena).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(root, textvariable=self.status_var, relief="sunken").pack(side="bottom", fill="x")

    def run_command(self, cmd):
        def thread_target():
            self.status_var.set(f"Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                self.status_var.set("Finished successfully")
            except subprocess.CalledProcessError as e:
                self.status_var.set(f"Error: {e}")
            except Exception as e:
                self.status_var.set(f"Failed: {e}")

        threading.Thread(target=thread_target, daemon=True).start()

    def run_gridworld(self):
        level = self.gw_level.get()
        algo = self.gw_algo.get()
        # Assume python3 is in path
        cmd = ["python3", "-m", "gridworld.main", "--level", str(level), "--algo", algo]
        self.run_command(cmd)

    def train_arena(self):
        style = self.ar_style.get().split()[0]
        algo = self.ar_algo.get()
        cmd = ["python3", "-m", "arena.train", "--algo", algo, "--style", style, "--steps", "10000"]
        self.run_command(cmd)

    def eval_arena(self):
        style = self.ar_style.get().split()[0]
        algo = self.ar_algo.get()
        cmd = ["python3", "-m", "arena.evaluate", "--algo", algo, "--style", style, "--episodes", "3"]
        self.run_command(cmd)

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()
