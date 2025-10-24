#!/bin/bash
#!/bin/bash
#SBATCH -p orfoz
#SBATCH -C weka
#SBATCH -A muhammedkaya
#SBATCH -J bigru_rev
#SBATCH -o bigru_rev.out
#SBATCH -N 1
#SBATCH -n 55
#SBATCH --time=1-00:00:00


# Conda ortamını etkinleştirme
eval "$(/arf/home/muhammedkaya/miniconda3/bin/conda shell.bash hook)"
conda activate dl-env

# Çalıştırılacak Python dosyası
python bigru_revision.py