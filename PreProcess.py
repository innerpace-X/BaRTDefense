from PIL import Image
import skimage
from skimage import color,transform,filters,morphology,restoration,exposure
import numpy as np
from scipy import fftpack
from scipy._lib.six import BytesIO
import os
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf

def randUnifC(low, high, params=None):
    p = np.random.uniform()
    if params is not None:
        params.append(p)
    return (high-low)*p + low

def randUnifI(low, high, params=None):
    p = np.random.uniform()
    if params is not None:
        params.append(p)
    return round((high-low)*p + low)

def randLogUniform(low, high, base=np.exp(1)):
    div = np.log(base)
    return base**np.random.uniform(np.log(low)/div,np.log(high)/div)


#input [-1,1]
def preprocess(img):
    choices = np.zeros(shape = [25])
    #ABOUT TRAINING: slowly increase nums here
    nums = np.random.random_integers(0,10)
    picked = np.random.random_integers(0,24,nums)
    choices[picked] = 1
    #print(picked)
    #Color reduction
    if(choices[0]==1):
        scales = [np.asscalar(np.random.random_integers(8, 200)) for x in range(3)]
        multi_channel = np.random.choice(2) == 0
        params = [multi_channel] + [s / 200.0 for s in scales]
        if multi_channel:
            img = np.round(img * scales[0]) / scales[0]
        else:
            for i in range(3):
                img[:, :, i] = np.round(img[:, :, i] * scales[i]) / scales[i]

    #JPEG Noise
    if (choices[1] == 1):
        quality = np.asscalar(np.random.random_integers(55,95))
        params = [quality / 100.0]
        pil_image = Image.fromarray((img * 255.0).astype(np.uint8) )
        f = BytesIO()
        pil_image.save(f, format='jpeg', quality=quality)
        jpeg_image = np.asarray(Image.open(f),).astype(np.float32) / 255.0
        img = jpeg_image

    #Swirl
    if (choices[2] == 1):
        strength = (2.0 - 0.01) * np.random.random(1)[0] + 0.01
        c_x = np.random.random_integers(1, 256)
        c_y = np.random.random_integers(1, 256)
        radius = np.random.random_integers(10, 200)
        params = [strength / 2.0, c_x / 256.0, c_y / 256.0, radius / 200.0]
        img = transform.swirl(img, rotation=0,
                                      strength=strength, radius=radius, center=(c_x,
                                                                                c_y))

    #Noise Injection
    if (choices[3] == 1):
        params = []
        options = ['gaussian', 'poisson', 'salt', 'pepper', 's&p', 'speckle']
        noise_type = np.random.choice(options, 1)[0]
        params.append(options.index(noise_type) / 6.0)
        per_channel = np.random.choice(2) == 0
        params.append(per_channel)
        if per_channel:
            for i in range(3):
                img[:, :, i] = skimage.util.random_noise(img[:, :, i], mode = noise_type )
            else:
                img = skimage.util.random_noise(img,mode = noise_type)

    # FFT Perturbation
    if (choices[4] == 1):
        r, c, _ = img.shape
        point_factor = (1.02 - 0.98) * np.random.random((r, c)) + 0.98
        randomized_mask = [np.random.choice(2) == 0 for x in range(3)]
        keep_fraction = [(0.95 - 0.0) * np.random.random(1)[0] + 0.0
        for x in range(3)]
        params = randomized_mask + keep_fraction
        for i in range(3):
            im_fft = fftpack.fft2(img[:, :, i])
        r, c = im_fft.shape
        if randomized_mask[i]:
            mask = np.ones(im_fft.shape[:2]) > 0
            im_fft[int(r * keep_fraction[i]):int(r * (1 - keep_fraction[i]))] = 0
            im_fft[:, int(c * keep_fraction[i]):int(c * (1 - keep_fraction[i]))] = 0
            mask = ~mask
            mask = mask * ~(np.random.uniform(size=im_fft.shape[:2] ) <keep_fraction[i])
            mask = ~mask
            im_fft = np.multiply(im_fft, mask)
        else:
            im_fft[int(r * keep_fraction[i]):int(r*(1-keep_fraction[i]))] = 0
            im_fft[:, int(c*keep_fraction[i]):int(c*(1-keep_fraction[i]))] = 0
        im_fft = np.multiply(im_fft, point_factor)
        im_new = fftpack.ifft2(im_fft).real
        im_new = np.clip(im_new, 0, 1)
        img[:, :, i] = im_new

    #Zooms
    if (choices[5] == 1):
        h, w, _ = img.shape
        i_s = np.random.random_integers(10, 50)
        i_e = np.random.random_integers(10, 50)
        j_s = np.random.random_integers(10, 50)
        j_e = np.random.random_integers(10, 50)
        params = [i_s / 50, i_e / 50, j_s / 50, j_e / 50]
        i_e = h - i_e
        j_e = w - j_e
        # Crop the image...
        img = img[i_s:i_e, j_s:j_e, :]
        # ...now scale it back up
        img = skimage.transform.resize(img, (h, w, 3))
    if(choices[11] == 1):
        #DO NOTHING HERE BECAUSE SEAM_CARVE REMOVED
        """
        h, w, _ = img.shape
        both_axis = np.random.choice(2) == 0
        toRemove_1 = np.random.random_integers(10, 50)
        toRemove_2 = np.random.random_integers(10, 50)
        params = [both_axis, toRemove_1 / 50, toRemove_2 / 50]
        if both_axis:
            # First remove from vertical
            eimg = skimage.filters.sobel(skimage.color.rgb2gray(img) )
            img = transform.seam_carve(img, eimg, 'vertical', toRemove_1)
            # Now from horizontal
            eimg = skimage.filters.sobel(skimage.color.rgb2gray(img) )
            img = transform.seam_carve(img, eimg,'horizontal', toRemove_2)
        else:
            eimg = skimage.filters.sobel(skimage.color.rgb2gray(img) )
            direction = 'horizontal'
            if toRemove_2 < 30:
                direction = 'vertical'
                
            img = transform.seam_carve(img, eimg,direction, toRemove_1)
            # Now scale it back up
            img = skimage.transform.resize(img, (h, w, 3))
        """

    #Color space processes
    if(choices[6]==1):
        img = color.rgb2hsv(img)
        params = []
        # Hue
        img[:, :, 0] += randUnifC(-0.05, 0.05, params=params)
        # Saturation
        img[:, :, 1] += randUnifC(-0.25, 0.25, params=params)
        # Value
        img[:, :, 2] += randUnifC(-0.25, 0.25, params=params)
        img = np.clip(img, 0, 1.0)
        img = color.hsv2rgb(img)
        img = (img*2.0)-1.0
        img = np.clip(img,-1.0,1.0)

    if (choices[8] == 1):
        img = (img+1.0)*0.5
        img = color.rgb2xyz(img)
        params = []
        # X
        img[:, :, 0] += randUnifC(-0.05, 0.05, params=params)
        # Y
        img[:, :, 1] += randUnifC(-0.05, 0.05, params=params)
        # Z
        img[:, :, 2] += randUnifC(-0.05, 0.05, params=params)
        img = np.clip(img, 0, 1.0)
        img = color.xyz2rgb(img)
        img = (img * 2.0) - 1.0
        img = np.clip(img, -1.0, 1.0)


    if (choices[9] == 1):
        img = (img + 1.0) * 0.5
        img = color.rgb2lab(img)
        params = []
        # L
        img[:, :, 0] += randUnifC(-5.0, 5.0, params=params)
        # a
        img[:, :, 1] += randUnifC(-2.0, 2.0, params=params)
        # b
        img[:, :, 2] += randUnifC(-2.0, 2.0, params=params)
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 100.0)
        img = color.lab2rgb(img)
        img = (img * 2.0) - 1.0
        img = np.clip(img, -1.0, 1.0)
    if(choices[10] == 1):
        img = (img + 1.0) * 0.5
        img = color.rgb2yuv(img)
        params = []
        # Y
        img[:, :, 0] += randUnifC(-0.05, 0.05, params=params)
        # U
        img[:, :, 1] += randUnifC(-0.02, 0.02, params=params)
        # V
        img[:, :, 2] += randUnifC(-0.02, 0.02, params=params)
        # U & V channels can have negative values; clip only Y
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 1.0)
        img = color.yuv2rgb(img)
        img = (img * 2.0) - 1.0
        img = np.clip(img, -1.0, 1.0)
    if(choices[7]==1):
        nbins = np.random.random_integers(40, 256)
        params = [nbins / 256.0]
        for i in range(3):
            img[:, :, i] = skimage.exposure.equalize_hist(img[:, :, i], nbins = nbins)
        img = (img * 2.0) - 1.0
        img = np.clip(img, -1.0, 1.0)
    #if(choices[12]==1):
    if(choices[13]==1):
        per_channel = np.random.choice(2) == 0
        params = [per_channel]
        low_precentile = [randUnifC(0.01, 0.04, params=params) for x in range(3)]
        hi_precentile = [randUnifC(0.96, 0.99, params=params)for x in range(3)]
        if per_channel:
            for i in range(3):
                p2, p98 = np.percentile(img[:, :, i],
                                        (low_precentile[i] * 100,
                                         hi_precentile[i] * 100))
                img[:, :, i] =skimage.exposure.rescale_intensity(img[:, :, i], in_range=(p2, p98))
        else:
            p2, p98 = np.percentile(img, (low_precentile[0] * 100, hi_precentile[0] * 100))
            img = skimage.exposure.rescale_intensity(img, in_range = (p2, p98) )
        img = (img * 2.0) - 1.0
        img = np.clip(img, -1.0, 1.0)
    if(choices[14]==1):
        ratios = np.random.rand(3)
        ratios /= ratios.sum()
        params = [x for x in ratios]
        img_g = img[:, :, 0] * ratios[0] + img[:, :, 1] * ratios[1] + img[:, :, 2] * ratios[2]
        for i in range(3):
            img[:, :, i] = img_g
        img = np.clip(img, -1.0, 1.0)
    if(choices[15]==1):
        ratios = np.random.rand(3)
        ratios /= ratios.sum()
        prop_ratios = np.random.rand(3)
        params = [x for x in ratios] + [x for x in prop_ratios]
        img_g = img[:, :, 0] * ratios[0] + img[:, :, 1] * ratios[1]+ img[:, :, 2] * ratios[2]
        for i in range(3):
            p = max(prop_ratios[i], 0.2)
        img[:, :, i] = img[:, :, i] * p + img_g * (1.0 - p)
        img = np.clip(img, -1.0, 1.0)
    if(choices[16]==1):
        params = []
        channels = [0, 1, 2]
        remove_channel = np.random.choice(3)
        channels.remove(remove_channel)
        params.append(remove_channel)
        ratios = np.random.rand(2)
        ratios /= ratios.sum()
        params.append(ratios[0])
        img_g = img[:, : ,channels[0]] * ratios[0] + img[:, :, channels[1]] * ratios[1]
        for i in channels:
            img[:, :, i] = img_g
        img = np.clip(img, -1.0, 1.0)
    if(choices[17]==1):
        params = []
        channels = [0, 1, 2]
        to_alter = np.random.choice(3)
        channels.remove(to_alter)
        params.append(to_alter)
        ratios = np.random.rand(2)
        ratios /= ratios.sum()
        params.append(ratios[0])
        img_g = img[:, :, channels[0]] * ratios[0] + img[:, :, channels[1]] * ratios[1]
        # Lets mix it back in with the original channel
        p = (0.9 - 0.1) * np.random.random(1)[0] + 0.1
        params.append(p)
        img[:, :, to_alter] = img_g * p + img[:, :, to_alter] *(1.0 - p)
        img = np.clip(img, -1.0, 1.0)
    if(choices[18]==1):
        if randUnifC(0, 1) > 0.5:
            sigma = [randUnifC(0.1, 3)] * 3
        else:
            sigma = [randUnifC(0.1, 3), randUnifC(0.1, 3), randUnifC(0.1, 3)]
        img[:, :, 0] = skimage.filters.gaussian(img[:, :, 0],sigma = sigma[0])
        img[:, :, 1] = skimage.filters.gaussian(img[:, :, 1], sigma = sigma[1])
        img[:, :, 2] = skimage.filters.gaussian(img[:, :, 2],sigma = sigma[2])
        img = np.clip(img, -1.0, 1.0)
    if(choices[19]==1):
        if randUnifC(0, 1) > 0.5:
            radius = [randUnifI(2, 5)] * 3
        else:
            radius = [randUnifI(2, 5), randUnifI(2, 5), randUnifI(2, 5)]
        # median blur - different sigma for each channel
        for i in range(3):
            mask = skimage.morphology.disk(radius[i])
            img[:, :, i] = skimage.filters.rank.median( np.clip(img[:, :, i],-1.0,1.0), mask)/ 255.0
        img = np.clip(img, -1.0, 1.0)
    if(choices[20]==1):
        if randUnifC(0, 1) > 0.5:
            radius = [randUnifI(2, 3)] * 3
        else:
            radius = [randUnifI(2, 3), randUnifI(2, 3), randUnifI(2, 3)]
        # mean blur w/ different sigma for each channel
        for i in range(3):
            mask = skimage.morphology.disk(radius[i])
            img[:, :, i] = skimage.filters.rank.mean(np.clip(img[:, :, i],-1.0,1.0), mask) / 255.0
        img = np.clip(img, -1.0, 1.0)
    if(choices[21]==1):
        params = []
        radius = []
        ss = []
        for i in range(3):
            radius.append(randUnifI(2, 20, params=params))
            ss.append(randUnifI(5, 20, params=params))
            ss.append(randUnifI(5, 20, params=params))
        for i in range(3):
            mask = skimage.morphology.disk(radius[i])
            img[:, :, i] = skimage.filters.rank.mean_bilateral( np.clip(img[:, :, i],-1.0,1.0), mask, s0 = ss[i], s1 = ss[3 + i]) / 255.0
        img = np.clip(img, -1.0, 1.0)
    if(choices[22]==1):
        params = []
        weight = (0.25 - 0.05) * np.random.random(1)[0] + 0.05
        params.append(weight)
        multi_channel = np.random.choice(2) == 0
        params.append(multi_channel)
        img = skimage.restoration.denoise_tv_chambolle(img, weight = weight, multichannel = multi_channel)
        img = np.clip(img, -1.0, 1.0)
    if(choices[23]==1):
        wavelets = ['db1','db2', 'haar', 'sym9']
        convert2ycbcr = np.random.choice(2) == 0
        wavelet = np.random.choice(wavelets)
        mode_ = np.random.choice(["soft", "hard"])
        denoise_kwargs = dict(multichannel=True,
                              convert2ycbcr=convert2ycbcr, wavelet=wavelet,
                              mode=mode_)
        max_shifts = np.random.choice([0, 1])
        params = [convert2ycbcr, wavelets.index(wavelet) /
                  float(len(wavelets)), max_shifts / 5.0,
                  (mode_ == "soft")]
        img = skimage.restoration.cycle_spin(img,
                                             func=skimage.restoration.denoise_wavelet,
                                             max_shifts=max_shifts, func_kw=denoise_kwargs)
        img = np.clip(img, -1.0, 1.0)
    if(choices[24]==1):
        wavelets = ['db1', 'db2', 'haar', 'sym9']
        convert2ycbcr = np.random.choice(2) == 0
        wavelet = np.random.choice(wavelets)
        mode_ = np.random.choice(["soft", "hard"])
        denoise_kwargs = dict(multichannel=True,
                              convert2ycbcr=convert2ycbcr, wavelet=wavelet,
                              mode=mode_)
        max_shifts = np.random.choice([0, 1])
        params = [convert2ycbcr, wavelets.index(wavelet) /
                  float(len(wavelets)), max_shifts / 5.0,
                  (mode_ == "soft")]
        img = skimage.restoration.cycle_spin(img,
                                             func=skimage.restoration.denoise_wavelet,
                                             max_shifts=max_shifts, func_kw=denoise_kwargs)
        img = np.clip(img, -1.0, 1.0)

    return img

a=0;

if __name__ == '__main__':
    for filenames, image in load_images(input_dir):
        a=a+1
        img=preprocess(image)
        print(a)
        save_images(img, filenames, output_dir)
