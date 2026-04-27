import subprocess
import sys

kl_coef = 1
lr = 1e-4
latent_dim=[512]
#scheme=['d']
scheme=[  'c']
#embedding=[128,512]


for i in range(len(latent_dim)):


        lrD = lr/ 10
        lrG= lr
        kl=kl_coef
        lat=latent_dim

        cmd = [
            sys.executable,
            "main.py",
            "--kl", str(kl),
            "--lrG", str(lrG),
            "--lrD", str(lrD),
            "--reconst_vis", "1",
            "--multiphase", "1",
            "--latent_dim", str(lat[i]),
            "--scheme", scheme[0],
            "--resume_phase", str(0),
            "--resume_epoch", str(None),
            "--batch_size", str(1),
        ]


        print("Running:", " ".join(cmd))

        subprocess.run(cmd)