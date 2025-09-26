import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import scipy.constants as cst
import pyhdf.SD as SD
import scipy.signal as sig
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy import ndimage
from matplotlib.colors import LogNorm
from GPR_background import GP_beam_fitter
from Ratiocurve_calculation import Setup

Parent_loc = os.path.join('/', 'Users', 'hpoole', 'Library', 'CloudStorage', 'Box-Box', 'TDYNO_NLUF', 'OMEGA')
Save_info = False
Shot_numbers = [108616]

## XRFC Information (this info should be stored in hdf files)
Pixel_size = 9 * 1e-3 # mm
Magnification = 2
CCD_binning = 2
mm_per_pixel = CCD_binning*Pixel_size/Magnification

Image_resolution = 50e-3 / Magnification # mm
Resolution_pixels = Image_resolution / mm_per_pixel

class Read_image:
    def __init__(self, Shot_number):
        self.shot_number = Shot_number
        self.shot_day, self.time, self.ots_time, self.bias, self.signal_frac, self.ne = self.get_shot_info()

        self.file_loc = os.path.join(Parent_loc, self.shot_day, 'Data', str(self.shot_number), 'XRFC3')
        self.processed_file = os.path.join(self.file_loc, 'XRFC_processed_{}.txt'.format(self.shot_number))
        if os.path.exists(self.processed_file):
            self.image = np.genfromtxt(self.processed_file)
            print('Image read')
        else:
            self.raw_image = self.read_hdf()
            PROCESSING = Process_image(self.raw_image, self.shot_number)
            self.image = PROCESSING.processed_image
            if Save_info:
                np.savetxt(os.path.join(self.file_loc, 'XRFC_processed_{}.txt'.format(self.shot_number)), self.image)

    def get_shot_info(self):
        if self.shot_number == 108611:
            Shot_day = 'OMEGA_Jun2023'
            XRFC_time = 22.5
            Bias = 150
            OTS_time = 27
            Signal_frac = 0.15
            ne = '2e20'
        elif self.shot_number == 108613:
            Shot_day = 'OMEGA_Jun2023'
            XRFC_time = 24.5
            Bias = 150
            OTS_time = 19
            Signal_frac = 0.15
            ne = '3e20'
        elif self.shot_number == 108614:
            Shot_day = 'OMEGA_Jun2023'
            XRFC_time = 22.5
            Bias = 150
            OTS_time = 21
            Signal_frac = 0.15
            ne = '2e20'
        elif self.shot_number == 108615:
            Shot_day = 'OMEGA_Jun2023'
            XRFC_time = 18.5
            Bias = 50
            OTS_time = 23
            Signal_frac = 0.15
            ne = '5e19'
        elif self.shot_number == 108616:
            Shot_day = 'OMEGA_Jun2023'
            XRFC_time = 20.5
            Bias = 100
            OTS_time = 25
            Signal_frac = 0.15
            ne = '1e20'
        elif self.shot_number == 108617:
            Shot_day = 'OMEGA_Jun2023'
            XRFC_time = 24.5
            Bias = 150
            OTS_time = 19
            Signal_frac = 0.15
            ne = '3e20'

        return Shot_day, XRFC_time, OTS_time, Bias, Signal_frac, ne

    def read_hdf(self, plot=False):

        File_name = 'xrfccd_xrfc3_t1_{}.hdf'.format(self.shot_number)
        File = os.path.join(self.file_loc, File_name)

        ## Load hdf5 file using standard hdf5 reader
        hdf = SD.SD(File, SD.SDC.READ)
        hdf_object = hdf.datasets()
        # print(hdf_object, hdf.attributes()), hdf.info())

        ## Get list of data groups stored in draco hdf file
        groups = [k for k in hdf_object.keys()]
        # print(groups)

        ## Real data
        data = hdf.select(groups[0])
        array = np.array(data.get())
        raw_image = array[0]
        background = array[1]

        image = raw_image - background
        image[image > 4e4] = 0
        original = image

        if plot:
            plt.clf()
            plt.figure()
            plt.imshow(image+1, norm=LogNorm(vmin=10), cmap='inferno')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.suptitle('{} Data'.format(self.shot_number))
            plt.show()
        return np.asarray(image, dtype=np.float64)

class Process_image:
    def __init__(self, raw_image, shot_number):
        self.shot_number = shot_number
        self.raw_image = raw_image
        self.image_center = tuple([int(self.raw_image.shape[0]/2), int(self.raw_image.shape[1]/2)])
        self.crop_image()
        self.processed_image = self.background_removal(self.cropped_image)

    def crop_image(self, plot=False):
        self.cropped_image = self.raw_image[250:1850, self.image_center[0]-400:self.image_center[0]+400]
        if plot:
            plt.figure()
            plt.imshow(self.cropped_image+1, norm=LogNorm(vmin=10), cmap='inferno')
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.suptitle('Cropped Data s{}'.format(self.shot_number))
            plt.show()

    def background_removal(self, image, N_samples=1000, N_pred=(300, 700), plot=True):
        mask_image = np.asarray(image.copy(), dtype=np.float64)
        mask_image = self.salt_and_pepper(mask_image)
        mask_image = self.gaussian_smoothing(mask_image, res=25)
        mask_image[mask_image >= 0.1 * np.nanmax(mask_image)] = np.nan
        mask_image[np.isfinite(mask_image)] = 1
        background_plasma = image*mask_image
        plasma = image*np.abs(np.nan_to_num(mask_image)-1)


        ## First remove the broad background
        bf = GP_beam_fitter(np.isfinite(background_plasma) * 1.0, N_samples=N_samples, N_pred=N_pred)
        bkg_img, bkg_unc = bf.fit_beam(image)
        broad_bkgd_removed_image = image - bkg_img

        plasma_img = broad_bkgd_removed_image
        plasma_img = self.salt_and_pepper(plasma_img)
        plasma_img = self.gaussian_smoothing(plasma_img)
        plasma_img[plasma_img < 1] = 0

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            im = axs[0].imshow(image+1, norm=LogNorm(vmin=10), cmap='inferno', origin='lower')
            plt.colorbar(im, ax=axs[0])
            axs[0].set_title('Original')
            im = axs[1].imshow(plasma_img+1, norm=LogNorm(vmin=10), cmap='inferno', origin='lower')
            plt.colorbar(im, ax=axs[1])
            axs[1].set_title('Background Removed')
            plt.suptitle('s{}'.format(self.shot_number))
            plt.show()

        return plasma_img

    def salt_and_pepper(self, Image):
        def snp(image):
            image = sig.medfilt2d(image, kernel_size=(3, 3))
            return image

        for i in range(0, 5, 1):
            Image = snp(Image)
        return Image

    def gaussian_smoothing(self, Image, res=None):
        def fwhm2sigma(fwhm):
            return fwhm / np.sqrt(8 * np.log(2))

        if res is None:
            res = Resolution_pixels
        Sigma = fwhm2sigma(res)
        Smoothed_image = ndimage.gaussian_filter(Image, sigma=Sigma, order=0)
        return Smoothed_image

class Generate_ratio_image:
    def __init__(self, image, READ):
        self.image = image
        self.shot_number, self.time, self.file_loc, self.signal_frac = READ.shot_number, READ.time, READ.file_loc, READ.signal_frac
        self.images = self.get_individual_images(self.image)
        self.intensity_image = self.images[0].copy()
        self.corr_images = self.correlate_filters(*self.images)
        self.ratio_image = self.create_ratio_map(*self.corr_images)
        if Save_info:
            np.savetxt(os.path.join(self.file_loc, 'XRFC_ratiomap_{}.txt'.format(self.shot_number)), self.ratio_image)

    def get_individual_images(self, Image, plot=False):
        Image_center = [int(np.shape(Image)[1] / 2), int(np.shape(Image)[0] / 2)]

        Top_Image = Image.copy()[Image_center[1] + 100:, :]
        Bottom_Image = Image.copy()[:Image_center[1] - 100, :]

        def gaus(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        def image_crop(img, plot=plot):
            x = np.arange(0, np.shape(img)[1], 1)
            y = np.arange(0, np.shape(img)[0], 1)
            Sum_Y = np.nansum(img, axis=0)
            Sum_Y = Sum_Y/np.nanmax(Sum_Y)
            Sum_X = np.nansum(img, axis=1)
            Sum_X = Sum_X/np.nanmax(Sum_X)
            popty, pcov = curve_fit(gaus, x, Sum_Y, p0=[1, x[-1] / 2, 3])
            Peak_x = int(popty[1])
            poptx, pcov = curve_fit(gaus, y, Sum_X, p0=[1, y[-1] / 2, 3])
            Peak_y = int(poptx[1])
            # print(Peak_x, Peak_y)
            crop_img = img.copy()[Peak_y-240:Peak_y+240, Peak_x-240:Peak_x+240]

            if plot:
                plt.figure()
                plt.imshow(img+1, norm=LogNorm(vmin=10), cmap='inferno')
                plt.plot(x, 100*Sum_Y+400, 'w-', alpha=0.6)
                plt.plot(x, 100*gaus(x, *popty)+400, 'r-.')
                plt.axvline(Peak_x, linestyle='--', color='w')

                plt.plot(100*Sum_X+400, y, 'w-', alpha=0.6)
                plt.plot(100*gaus(y, *poptx)+400, y, 'r-.')
                plt.axhline(Peak_y, linestyle='--', color='w')
                plt.gca().invert_yaxis()
                plt.suptitle(self.shot_number)
                plt.show()

                plt.figure()
                plt.imshow(crop_img+1, norm=LogNorm(vmin=10), cmap='inferno')
                plt.gca().invert_yaxis()
                plt.suptitle(self.shot_number)
                plt.show()

            return crop_img

        Bottom_Image = image_crop(Bottom_Image)
        Top_Image = image_crop(Top_Image)

        return Top_Image, Bottom_Image

    def correlate_filters(self, Top_Image, Bottom_Image, plot=False):

        Norm_top_image = Top_Image / np.nanmax(Top_Image)
        Norm_bottom_image = Bottom_Image / np.nanmax(Bottom_Image)

        ## 2D takes a long time...
        # Cross_correlation = sig.correlate2d(Norm_top_image, Norm_bottom_image, boundary='symm', mode='same')
        Cross_correlation = sig.correlate(Norm_top_image, Norm_bottom_image, mode='same', method='fft')
        y, x = np.unravel_index(np.argmax(Cross_correlation), Cross_correlation.shape)

        Top_Image_centre = [int(np.shape(Norm_bottom_image)[1] / 2), int(np.shape(Norm_bottom_image)[0] / 2)]
        Top_Image_yshift = y - Top_Image_centre[1]
        Image_centre = [int(np.shape(Norm_bottom_image)[1] / 2), int(np.shape(Norm_bottom_image)[0] / 2)]


        New_top_image = Norm_top_image[y - 200:y + 200, x - 200:x + 200]
        New_bottom_image = Norm_bottom_image[Image_centre[1] - 200:Image_centre[1] + 200,
                           Image_centre[0] - 200:Image_centre[0] + 200]

        Sum_image = New_top_image + New_bottom_image
        Sum_image = Sum_image / np.nanmax(Sum_image)

        Final_top_image = Top_Image[y - 200:y + 200, x - 200:x + 200]
        Final_bottom_image = Bottom_Image[Image_centre[1] - 200:Image_centre[1] + 200,
                             Image_centre[0] - 200:Image_centre[0] + 200]

        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(Final_top_image+1, norm=LogNorm(vmin=10), cmap='inferno')
            axs[0].set_title('Top filter')
            axs[1].imshow(Final_bottom_image+1, norm=LogNorm(vmin=10), cmap='inferno')
            axs[1].set_title('Bottom filter')
            for a in axs.flat:
                a.invert_yaxis()
            plt.show()


        return Final_top_image, Final_bottom_image

    def create_ratio_map(self, Image_1, Image_2, flatfield_corr=1, plot=True):

        def remove_outliers(a, bins=100, plot=False):
            values = (a[~np.isnan(a)])
            hist, bin_edges = np.histogram(values, bins=bins)
            hist = hist / np.nanmax(hist)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            edge = 0.01
            try:
                low_edge = bin_centers[np.where((hist <= edge) & (bin_centers <= bin_centers[np.argmax(hist)]))[0][-1]]
            except:
                low_edge = 0
            try:
                high_edge = bin_centers[np.where((hist <= edge) & (bin_centers >= bin_centers[np.argmax(hist)]))[0][0]]
            except:
                high_edge = 100
            a[a <= low_edge] = np.nan
            a[a >= high_edge] = np.nan

            if plot:
                plt.figure()
                plt.hist(bin_centers, weights=hist, bins=bin_centers)
                plt.axvline(high_edge, color='k', linestyle='--')
                plt.axvline(low_edge, color='k', linestyle='--')
                plt.show()
            return a

        Image_1[Image_1 <= self.signal_frac*np.nanmax(Image_1)] = np.nan
        Image_2[Image_2 <= self.signal_frac*np.nanmax(Image_2)] = np.nan
        Image_2 = Image_2*flatfield_corr

        Ratio = Image_1/Image_2
        Ratio = remove_outliers(Ratio, bins=100)

        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(8, 7))
            # plt.tight_layout()
            # plt.subplots_adjust(hspace=0)

            def plot_image(image, ax, vmin=None, vmax=None, cmap='binary'):
                if vmin is None:
                    vmin = np.nanmin(image)
                if vmax is None:
                    vmax = np.nanmax(image)
                im = ax.imshow(image,
                          vmin=vmin, vmax=vmax, cmap=cmap)
                plt.colorbar(im, ax=ax)

            plot_image(Image_1, axs[0, 0])
            plot_image(Image_2, axs[0, 1])
            plot_image(Ratio, axs[1, 0], cmap='jet')
            axs[1, 1].hist(Ratio.flatten(), bins=50, color='blue')

            for a in [axs[0, 0], axs[0, 1], axs[1, 0]]:
                a.set_aspect('equal', adjustable='box')
                a.invert_yaxis()

            for a in axs.flat:
                a.tick_params(axis='both', which='major', length=10, direction='in', top=True, right=True)
                a.tick_params(axis='both', which='minor', length=5, direction='in', top=True, right=True)
                a.minorticks_on()

            # axs[0, 0].set_title(self.filters[0].replace('_', ' '), fontsize=20)
            # axs[0, 1].set_title(self.filters[1].replace('_', ' '), fontsize=20)

            plt.suptitle('Ratio Images s{} t = {} ns'.format(self.shot_number, self.time), fontsize=22)
            plt.show()

        return Ratio

class Find_filter_thickness:
    def __init__(self):
        self.shot_number, self.time, self.file_loc, self.ne = READ.shot_number, READ.time, READ.file_loc, READ.ne
        self.ratio_map = np.genfromtxt(os.path.join(self.file_loc, 'XRFC_ratiomap_{}.txt'.format(self.shot_number)))
        self.optimise_filters()
        # self.ratiocurve_run([-0.2, 0.05, 0.1, 0.2])

    def ratiocurve_run(self, filter_errs=[0.0, 0.0, 0.0, 0.0], plot=False):
        filt1_arr = ['Mylar 1 {}'.format(filter_errs[0]), 'V 0.2 {}'.format(filter_errs[2]), 'Al 0.8 {}'.format(filter_errs[3])]
        filt2_arr = ['Mylar 2 {}'.format(filter_errs[1]), 'V 0.2 {}'.format(filter_errs[2]), 'Al 0.8 {}'.format(filter_errs[3])]
        filt_arr = np.array([filt1_arr, filt2_arr])

        Run_Ratiocurve = Setup('CHCl', Filters=filt_arr, Electron_dens=[float(self.ne)])
        data = Run_Ratiocurve.ratiocurve_info
        self.ratio_te_org, self.ratio_org = data[0], data[1]

        ratio_te = self.ratio_te_org.copy()[np.argmax(self.ratio_org):]
        ratio = self.ratio_org.copy()[np.argmax(self.ratio_org):]

        self.ratiocurve_te, self.ratiocurve = ratio_te[ratio.argsort()], ratio[ratio.argsort()]
        if plot:
            plt.figure()
            plt.plot(self.ratio_te_org, self.ratio_org)
            plt.xlim(0, 500)
            plt.show()
        return

    def optimise_filters(self, norm=False):

        def ratiocurve_optimisation(filt1_err, filt2_err, filt3_err, filt4_err, plot=False):
            self.ratiocurve_run([filt1_err, filt2_err, filt3_err, filt4_err])
            if norm:
                ratiocurve = self.ratiocurve / np.nanmax(self.ratiocurve)
                dif = np.abs(np.nanmin(ratiocurve) - np.nanmin(self.ratio_map))
            else:
                ratiocurve = self.ratiocurve
                # dif = np.sum([np.abs(np.nanmin(self.ratiocurve) - np.nanmin(self.ratio_map)),
                #               np.abs(np.nanmax(self.ratiocurve) - np.nanmax(self.ratio_map))])
                dif = np.abs(np.nanmax(self.ratiocurve) - np.nanmax(self.ratio_map))
            if plot:
                plt.figure()
                plt.plot(self.ratiocurve_te, ratiocurve)
                plt.xlim(0, 500)
                plt.show()
            return dif

        # Define ranges for the four input values
        jump = 0.05
        filt1_err_values = np.arange(-0.25, 0.2501, jump) # Mylar 1
        filt2_err_values = np.arange(-0.25, 0.2501, jump)  # Mylar 2
        filt3_err_values = np.arange(-0.1, 0.101, jump) # V 0.2
        filt4_err_values = np.arange(-0.2, 0.201, jump)  # Al 0.8

        # Store the best values
        best_values = None
        min_diff = float('inf')

        print('Optimising...')
        for filt1_err, filt2_err, filt3_err, filt4_err in itertools.product(filt1_err_values, filt2_err_values, filt3_err_values, filt4_err_values):
            # Compute the difference using your function
            Ratiocurve_min_difference = ratiocurve_optimisation(filt1_err, filt2_err, filt3_err, filt4_err)

            # Check if this is the smallest difference so far
            if Ratiocurve_min_difference < min_diff:
                min_diff = Ratiocurve_min_difference
                best_values = (filt1_err, filt2_err, filt3_err, filt4_err)
        print("Optimal values:", best_values)
        print("Minimum difference:", min_diff)
        self.ratiocurve_run([best_values[0], best_values[1], best_values[2], best_values[3]])
        print(np.nanmax(self.ratio_map), np.nanmax(self.ratiocurve))

        return [best_values[0], best_values[1], best_values[2], best_values[3]]

class Generate_Te_map:
    def __init__(self, READ, ratio_image, filter_errs=[0, 0, 0, 0]): #[-0.2, 0.05, 0.1, 0.2]
        self.filter_errs = filter_errs
        self.shot_number, self.time, self.file_loc, self.ne = READ.shot_number, READ.time, READ.file_loc, READ.ne

        self.ratio_map = ratio_image
        # self.ratio_map = np.genfromtxt(os.path.join(self.file_loc, 'XRFC_ratiomap_{}.txt'.format(self.shot_number)))
        self.ratiocurve_run(self.filter_errs)
        self.te_map = self.get_te_map()
        if Save_info:
            np.savetxt(os.path.join(self.file_loc, 'XRFC_temap_{}.txt'.format(self.shot_number)), self.te_map)

    def ratiocurve_run(self, filter_errs=[0.0, 0.0, 0.0, 0.0], plot=False):
        filt1_arr = ['Mylar 1 {}'.format(filter_errs[0]), 'V 0.2 {}'.format(filter_errs[2]),
                     'Al 0.8 {}'.format(filter_errs[3])]
        filt2_arr = ['Mylar 2 {}'.format(filter_errs[1]), 'V 0.2 {}'.format(filter_errs[2]),
                     'Al 0.8 {}'.format(filter_errs[3])]
        filt_arr = np.array([filt1_arr, filt2_arr])

        Run_Ratiocurve = Setup('CHCl', Filters=filt_arr, Electron_dens=[float(self.ne)])
        data = Run_Ratiocurve.ratiocurve_info
        self.ratio_te_org, self.ratio_org = data[0], data[1]

        ratio_te = self.ratio_te_org.copy()[np.argmax(self.ratio_org):]
        ratio = self.ratio_org.copy()[np.argmax(self.ratio_org):]

        self.ratiocurve_te, self.ratiocurve = ratio_te[ratio.argsort()], ratio[ratio.argsort()]
        if plot:
            plt.figure()
            plt.plot(self.ratio_te_org, self.ratio_org)
            plt.xlim(0, 500)
            plt.show()
        return

    def get_te_map(self, plot=True):
        x_pix, y_pix = np.arange(-np.shape(self.ratio_map)[1]/2, np.shape(self.ratio_map)[1]/2, 1), np.arange(-np.shape(self.ratio_map)[0]/2, np.shape(self.ratio_map)[0]/2, 1)
        x, y = x_pix*mm_per_pixel, y_pix*mm_per_pixel

        Temp = self.ratiocurve_te.copy()[self.ratiocurve.argsort()]
        RatioFit = self.ratiocurve.copy()[self.ratiocurve.argsort()]
        Te_map = np.interp(self.ratio_map, RatioFit, Temp, left=np.nan, right=np.nan)

        if plot:
            def plot_image(image, ax, vmin=None, vmax=None, cmap='jet'):
                if vmin is None:
                    vmin = np.nanmin(image)
                if vmax is None:
                    vmax = np.nanmax(image)
                im = ax.imshow(image, extent=[x[0], x[-1], y[0], y[-1]], origin='lower',
                               vmin=vmin, vmax=vmax, cmap=cmap)
                plt.colorbar(im, ax=ax)


            fig, axs = plt.subplots(2, 2, figsize=(8, 8))

            plot_image(self.ratio_map, axs[0, 0], vmin=0)
            plot_image(Te_map, axs[1, 0], vmin=0, vmax=400)

            axs[0, 1].plot(self.ratio_te_org, self.ratio_org)
            axs[1, 1].hist(Te_map.flatten(), bins=np.arange(0, 500, 5))

            for a in axs[:, -1].flat:
                a.set_xlim(0, 500)

            for a in axs.flat:
                a.tick_params(axis='both', which='major', length=10, direction='in', top=True, right=True)
                a.tick_params(axis='both', which='minor', length=5, direction='in', top=True, right=True)
                a.minorticks_on()
            plt.suptitle('Te Images s{} t = {} ns'.format(self.shot_number, self.time), fontsize=22)

            plt.show()

        return Te_map


for Shot_number in Shot_numbers:
    READ = Read_image(Shot_number)

    #%%
    RATIO = Generate_ratio_image(READ.image, READ)

    ## This takes time..
    # Find_filter_thickness()

    ## I have a feeling this won't work for you..
    ## Basically because I run the ratiocurve calculation instead of just reading a file in
    ## Same would be true for the Find_filter_thickness class
    
    #%%
    TE_GEN = Generate_Te_map(READ, RATIO.ratio_image)
    # Te_map = TE_GEN.te_map
    # Te_mean, Te_std = np.nanmean(Te_map), np.nanstd(Te_map)
# %%
