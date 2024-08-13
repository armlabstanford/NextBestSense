# ARMLab 2024
import subprocess
from gaussian_splatting_py.base_splatfacto import ROSSplatfacto


class Splatfacto3D(ROSSplatfacto):
    def __init__(self, data_dir='bunny_blender_dir', render_uncertainty=False,
                 train_split_fraction=0.5, depth_uncertainty_weight=1.0, rgb_uncertainty_weight=1.0):
        """
        initialize 3DGS. Calls the ns-train cmd to avoid manual copy paste.

        When the model trains, it will train to 2K steps,
        """
        super(Splatfacto3D, self).__init__(data_dir, render_uncertainty, train_split_fraction, depth_uncertainty_weight, rgb_uncertainty_weight)

    def start_training(self, data_dir, steps=15000):
        """
        Start training Gaussian Splatting 
        """
        print("Starting training")
        # outputs dir is the same as the data dir with outputs
        outputs_dir = f'{data_dir}/outputs'
        
        command = f"""ns-train depth-splatfacto --data {data_dir} --output-dir {outputs_dir} --pipeline.model.render_uncertainty {self.render_uncertainty} --viewer.quit-on-train-completion True --pipeline.model.depth-loss-mult 0.1 nerfstudio-data --train-split-fraction 1"""
        # Open terminal and run GS training
        print(command)
        
        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', command])
        print("Starting GS Training.")

    
if __name__ == "__main__":
    # train on bunny blender example
    data_dir = '/home/user/NextBestSense/data/2024-07-28-02-02-26'
    splatfacto = Splatfacto3D(data_dir=data_dir)
    splatfacto.start_training(data_dir)