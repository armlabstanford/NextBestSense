# ARMLab 2024
import subprocess
from base_splatfacto import ROSSplatfacto


class Splatfacto3D(ROSSplatfacto):
    def __init__(self, data_dir='bunny_blender_dir', render_uncertainty=False,
                 train_split_fraction=0.5, depth_uncertainty_weight=1.0, rgb_uncertainty_weight=1.0):
        """
        initialize 3DGS. Calls the ns-train cmd to avoid manual copy paste.

        When the model trains, it will train to 2K steps,
        """
        super(Splatfacto3D, self).__init__(data_dir, render_uncertainty, train_split_fraction, depth_uncertainty_weight, rgb_uncertainty_weight)

    def start_training(self, steps=15000):
        """
        Start training Gaussian Splatting 
        """
        print("Starting training")
        
        command = f"""ns-train depth-splatfacto --data {self.data_dir} --pipeline.model.render_uncertainty {self.render_uncertainty} --viewer.quit-on-train-completion True nerfstudio-data --train-split-fraction 0.5"""
        # Open terminal and run GS training
        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])

        print("Starting GS Training.")

    
if __name__ == "__main__":
    # train on bunny blender example
    data_dir = '/home/user/touch-gs-data/bunny_blender_data'
    splatfacto = Splatfacto3D(data_dir=data_dir)
    splatfacto.start_training()