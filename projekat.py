import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import iradon

# Parametri
R1 = 100  # cm
R2 = 25  # cm
R3 = 12  # cm
mu1 = 1  # cm^-1
mu2 = 2  # cm^-1
mu3 = 4  # cm^-1
s = 0.1  # cm
detector_size = 256
N = 180  # Broj projekcija

# Kreirajte matricu
fantom = np.zeros((256, 256))

# Postavite vrednosti za prvi krug u centru
center1 = (256 // 2, 256 // 2)
for (i, j), value in np.ndenumerate(fantom):
    distance_squared = (i - center1[0])**2 + (j - center1[1])**2
    if distance_squared <= R1**2:
        fantom[i, j] = 0.1

# Postavite vrednosti za drugi krug pomereno na levo
center2 = (256 // 3, 256 // 3)
for (i, j), value in np.ndenumerate(fantom):
    distance_squared = (i - center2[0])**2 + (j - center2[1])**2
    if distance_squared <= R2**2:
        fantom[i, j] += 0.2

# Postavite vrednosti za treći krug pomereno na levo
center3 = ((256 // 3) * 2, 256 // 3 * 2)
for (i, j), value in np.ndenumerate(fantom):
    distance_squared = (i - center3[0])**2 + (j - center3[1])**2
    if distance_squared <= R3**2:
        fantom[i, j] += 0.4

# Prikaz rezultujuće matrice kao slike
plt.imshow(fantom, cmap='gray', origin='lower', vmin=0, vmax=1)
plt.title('Matrica sa tri kruga')
plt.show()

# Generisanje sinograma pomoću Radon transformacije
theta = np.linspace(180., 0., max(fantom.shape), endpoint=True)
sinogram = np.zeros((len(theta), max(fantom.shape)))

for i, angle in enumerate(theta):
    rotated_image = rotate(fantom, angle, mode='constant', cval=0, order=2, reshape=False)
    sinogram[:, i] = np.sum(rotated_image, axis=0)

# Normalizacija projekcija


sinogram_origigi = np.copy(sinogram)
sinogram /= np.max(sinogram)



# Prikaz sinograma
plt.imshow(sinogram.T, cmap='gray', extent=(0, max(fantom.shape), 0, 180), aspect='auto')
plt.ylabel('Ugao (stepeni)')
plt.xlabel('Projekcija')
plt.title('Sinogram')
plt.show()

selected_angles = theta[::1]

# Projektovanje unazad sa odabranim uglovima
reconstructed_backprojection_selected = np.zeros_like(fantom)



for i, angle in enumerate(selected_angles):
    jedan_niz = np.zeros_like(sinogram)
    jedan_niz[: , (255//2)] = sinogram[:, i]
    rotated_projection = rotate(jedan_niz, angle, reshape=False)
    reconstructed_backprojection_selected += rotated_projection.squeeze()
    #plt.imshow(reconstructed_backprojection_selected.T, cmap='gray', origin='lower', vmin=0, vmax=1)
    #plt.show()
# Prikaz rekonstruisane slike sa odabranim uglovima
plt.imshow(reconstructed_backprojection_selected, cmap='gray', origin='lower', vmin=0, vmax=1)
plt.title('Rekonstruisana slika pomoću projektovanja unazad')
plt.show()



reconstructed_iradon = iradon(sinogram_origigi, theta=theta, circle=True)
reconstructed_iradon = np.flipud(reconstructed_iradon)

# Prikaz rekonstruisane slike
plt.imshow(reconstructed_iradon, cmap='gray', origin='lower', vmin=0, vmax=1)
plt.title('Rekonstruisana slika pomoću iradon')
plt.show()




def apply_filter(projection, filter_type):

    # Implementacija različitih filtera
    if filter_type == 'ram_lak':
        ram_lak_filter = np.abs(np.fft.fftshift(np.fft.fft(projection)))
        filtered_projection = np.fft.ifft(np.fft.ifftshift(ram_lak_filter)).real
    elif filter_type == 'shepp_logan':
        r = np.abs(np.fft.fftshift(np.fft.fft(projection)))
        shepp_logan_filter = np.abs(r) * np.sinc(r)
        filtered_projection = np.fft.ifft(np.fft.ifftshift(r*shepp_logan_filter)).real
    
    elif filter_type == 'hann':
        r = np.abs(np.fft.fftshift(np.fft.fft(projection)))
        hann_filter = 0.5 * (1 + np.cos(2 * np.pi * r))
        filtered_projection = np.fft.ifft(np.fft.ifftshift(r*shepp_logan_filter)).real

        pass
    elif filter_type == 'hamming':
        # Implementacija Hamming filtera
        r = np.abs(np.fft.fftshift(np.fft.fft(projection)))
        hamming_filter = 0.54 - 0.46 * np.cos(2 * np.pi * r)
        # hamming_filter = np.hamming(r.size)
        filtered_projection = np.fft.ifft(np.fft.ifftshift(r * hamming_filter)).real
        pass
    else:
        raise ValueError("Nepodržan tip filtera")

    return filtered_projection

filtered_sinogram = np.zeros_like(sinogram)
filter_type = 'shepp_logan'  # Možete promeniti tip filtera

for i in range(sinogram.shape[1]):
    filtered_sinogram[:, i] = apply_filter(sinogram[:, i], filter_type)

# Filtrirane projekcije
plt.imshow(filtered_sinogram.T, cmap='gray', extent=(0, 180, 0, max(fantom.shape)), aspect='auto')
plt.xlabel('Ugao (stepeni)')
plt.ylabel('Projekcija')
plt.title('Filtrirani Sinogram')
plt.show()

# Filtrirana rekonstrukcija
filtered_backprojection = np.zeros_like(fantom)

for i, angle in enumerate(selected_angles):
    jedan_niz = np.zeros_like(filtered_sinogram)
    jedan_niz[: , (255//2)] = filtered_sinogram[:, i]
    rotated_projection = rotate(jedan_niz, angle, reshape=False)
    reconstructed_backprojection_selected += rotated_projection.squeeze()

# Prikaz rekonstruisane slike
plt.imshow(reconstructed_backprojection_selected, cmap='gray', origin='lower', vmin=0, vmax=1)
plt.title('Filtrirana Rekonstruisana Slika')