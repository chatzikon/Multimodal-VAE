import subprocess
import sys

kl_coef = 1
lr = 1e-2
latent_dim=64
#scheme=['d']
scheme=['a', 'b', 'c', 'd']




for i in range(len(scheme)):


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
            "--multiphase", "0",
            "--latent_dim", str(lat),
            "--scheme", scheme[i],
            "--resume_phase", str(None),
            "--resume_epoch", str(None)
        ]


        print("Running:", " ".join(cmd))

        subprocess.run(cmd)