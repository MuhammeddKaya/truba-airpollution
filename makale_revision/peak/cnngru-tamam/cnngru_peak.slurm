#!/bin/bash
#!/bin/bash
#SBATCH -p hamsi
#SBATCH -C weka
#SBATCH -A muhammedkaya
#SBATCH -J cnngru_peak
#SBATCH -o cnngru_peak.out
#SBATCH -N 1
#SBATCH -n 54
#SBATCH --time=2-00:00:00


# Conda ortamını etkinleştirme
eval "$(/arf/home/muhammedkaya/miniconda3/bin/conda shell.bash hook)"
conda activate dl-env

# Çalıştırılacak Python dosyası
python cnngru_peak.py
