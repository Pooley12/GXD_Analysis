import numpy as np
import os
import subprocess
import scipy.constants as cst
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import ndimage
from multiprocessing import Pool, cpu_count
import time
import sys
import matplotlib as mpl

class Setup:
    ## This is just for when it's imported into other scripts
    def __init__(self, Material, Filters=None, Electron_dens = None):
        if Material == 'CHCl':
            PARENT_loc = os.path.join(os.getcwd())
            PROPACEOS_loc = os.path.join(PARENT_loc, 'EOS')
            FILTER_loc = os.path.join(PARENT_loc, 'Filters')
            DETECTOR_SENSITIVITY_loc = os.path.join(PARENT_loc, 'Sensitivities', 'GXD_Sensitivity.csv')

            Spect3D_program_loc = os.path.join('/', 'Applications', 'Spect3D', 'Spect3D.app', 'Contents', 'MacOS', 'Spect3D')
            Spect3D_workspace = os.path.join(PARENT_loc, 'single_cell.spw')            
            
            Foil_composition = ['C 0.499', 'H 0.438', 'Cl 0.063']
            OUTPUT_file_loc = os.path.join(PARENT_loc, 'Ratiocurves', 'CHCl')
            PROPACEOS_file = os.path.join(PROPACEOS_loc, 'C49_9H43_8Cl6_3.prp')

        Electron_temps = np.concatenate([np.arange(20, 150, 1), np.arange(150, 1000, 10), np.arange(1000, 3000, 100), np.arange(3000, 5010, 1000)])  # eV
        if Electron_dens is None:
            Electron_dens = [2e20]
        Spect3D_areal_size = 0.1  # cm
        
        if Filters is None:
            Filters = [['Mylar 1 0', 'V 0.2 0', 'Al 0.8 0'], ['Mylar 2 0', 'V 0.2 0', 'Al 0.8 0']]

        CALCS = Scattering_Calculations(Atomic_makeup=Foil_composition, OUTPUT_loc=OUTPUT_file_loc,
                Electron_temperature=Electron_temps, Electron_density=Electron_dens,
                Size=Spect3D_areal_size, Spect3D_workspace=Spect3D_workspace, Spect3D_program=Spect3D_program_loc,
                PROPACEOS_file=PROPACEOS_file)
        
        CREATE = Create_Ratiocurves(Filters=Filters, Filter_loc=FILTER_loc, Detector_Sensitivity_loc=DETECTOR_SENSITIVITY_loc,
                    SCATTERING_CALC=CALCS, type='dat', Add_errors=False)
        # self.signals = CREATION.filter_signals
        self.ratiocurve_info = CREATE.get_ratiocurves(plot=True)

    def __getattr__(self, name):
        return getattr(self, name)

def filter_string(FILTER, Math=True):
    Filter_string = ''
    if np.shape(FILTER):
        for j in range(0, len(FILTER), 1):
            if j == 0:
                if Math:
                    Filter_string += '\\mathrm{{{:.4g}\\,\\mu m\\,\\,{}'.format(
                        float(FILTER[j].split()[1]) * (1 + float(FILTER[j].split()[2])),
                        FILTER[j].split()[0])
                else:
                    Filter_string += '{:.4g}um_{}'.format(
                        float(FILTER[j].split()[1]) * (1 + float(FILTER[j].split()[2])),
                        FILTER[j].split()[0])
            else:
                if Math:
                    Filter_string += ' + '
                    Filter_string += '{:.4g}\\,\\mu m\\,\\,{}'.format(
                        float(FILTER[j].split()[1]) * (1 + float(FILTER[j].split()[2])),
                        FILTER[j].split()[0])
                else:
                    Filter_string += '_+_'
                    Filter_string += '{:.4g}um_{}'.format(
                        float(FILTER[j].split()[1]) * (1 + float(FILTER[j].split()[2])),
                        FILTER[j].split()[0])
            if j == len(FILTER) - 1:
                if Math:
                    Filter_string += '}'
    else:
        if Math:
            Filter_string += '\\mathrm{{{:.4g}\\,\\mu m\\,\\,{}}}'.format(
                float(FILTER.split()[1]) * (1 + float(FILTER.split()[2])),
                FILTER.split()[0])
        else:
            Filter_string += '{:.4g}um_{}'.format(
                float(FILTER.split()[1]) * (1 + float(FILTER.split()[2])),
                FILTER.split()[0])
    return Filter_string

class Get_Atomic_info:
    def __init__(self, Atomic_makeup, Electron_densities, Size):
        self.atm = Atomic_makeup
        self.size = Size
        self.ne = Electron_densities
        self.Mean_atomic_weight = self.Atomic_masses()
        self.Mass_densities, self.Areal_densities = self.Spect3D_mass_inputs()

    def __getattr__(self, name):
        return getattr(self, name)

    def Atomic_masses(self):
        ########## CALCULATE MEAN WEIGHT AND IONISATION OF FOIL ##########
        ## This calculates the mean weight and ionisation of
        ## the foil composition using 'Atomic_data.txt' file
        ATOMIC_DATA = np.genfromtxt(os.path.join('.', 'Atomic_data.txt'), skip_header=2, usecols=(0, 1, 3), dtype=None,
                                    encoding='utf-8')
        AMU = cst.physical_constants['atomic mass constant'][0] * 1e3  # g
        Mean_atomic_weight = 0
        Mean_ionisation = 0
        for Element in self.atm:
            Element_weight = None
            Chemical_name = Element.split()[0]
            Fraction = float(Element.split()[1])
            for a in ATOMIC_DATA:
                if a[1] == Chemical_name:
                    Element_weight = a[2]
                    Element_ionisation = a[0]
            try:
                Mean_atomic_weight += Fraction * Element_weight
                Mean_ionisation += Fraction * Element_ionisation
            except:
                print('Error in Get_Atomic_info\n'
                      '\tAtomic_Masses => Error extracting atomic information.\n'
                      '\tCheck Element {} in Atomic Data file'.format(Chemical_name))
        Mass = Mean_atomic_weight * AMU  # g
        self.weight = Mean_atomic_weight
        self.ionisation = Mean_ionisation
        self.mass = Mass
        return Mean_atomic_weight

    def Spect3D_mass_inputs(self):
        ########## CALCULATE MASS AND AREAL DENSITIES ##########
        ## This calculates the mass and areal densities to input
        ## in SPECT3D based on the user-specified electron
        ## densities and areal size
        Mass_densities = (self.ne * self.mass) / self.ionisation
        Areal_densities = Mass_densities * self.size
        return Mass_densities, Areal_densities

class Create_Detector_Response:
    def __init__(self, Filters, Filter_loc, Detector_Sensitivity_loc, Add_errors=False, err=0.2):
        self.Filter_loc = Filter_loc
        self.Detector_sensitivity = Detector_Sensitivity_loc
        self.Filters = Filters
        self.add_errors = Add_errors
        self.error = err
        self.get_detector_responses()

    def __getattr__(self, name):
        return getattr(self, name)

    def get_filter_transmission(self):
        ########## EXTRACT FILTER TRANSMISSION ##########
        ## This extracts the filter transmission (from CXRO
        ## website) from Filter_loc folder
        # Thick = Thickness * (1+Error)
        File = os.path.join(self.Filter_loc, '{}um_{}.txt'.format(self.t, self.filter))
        try:
            data = np.genfromtxt(File, skip_header=2)
            # data = np.genfromtxt(File, skip_header=2, delimiter=',')
            E = data[:, 0]
            T = data[:, 1]
            T = T ** (float(self.t) * (1 + float(self.err)) / float(self.t))
        except:
            print('Error in Create_Ratiocurves\n'
                  '\tGet_filter_transmission => Error finding filter transmission\n'
                  '\tCheck Filter transmission file {} exists'.format(File))
            sys.exit()
        return E, T

    def get_detector_responses(self, plot=False):

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        ## The high temperature resolution is really required for the NIF platform
        Detector_data = np.genfromtxt(self.Detector_sensitivity, delimiter=',', usecols=(0,1), skip_header=1)
        E, Sensitivity = Detector_data[:, 0], Detector_data[:, 1]
        self.Photon_energy = np.arange(np.min(E), 10000, 1)
        # self.Photon_energy = np.arange(np.min(E), np.max(E) + 1, 1)
        # idx = find_nearest(E, 2500)  # This normalises the sensitivity
        # Sensitivity = Sensitivity / Sensitivity[idx]
        self.Sensitivity = np.interp(self.Photon_energy, E, Sensitivity)

        Responses = []
        Filter_transmissions = []
        for i in range(0, len(self.Filters), 1):
            if np.shape(self.Filters[i]):
                Ts = []
                for j in range(0, len(self.Filters[i]), 1):
                    self.filter, self.t, self.err = self.Filters[i][j].split()
                    E, ts = self.get_filter_transmission()
                    Ts.append(np.interp(self.Photon_energy, E, ts))
                T = np.prod(Ts, axis=0)
            else:
                self.filter, self.t, self.err = self.Filters[i].split()
                E, ts = self.get_filter_transmission()
                T = np.interp(self.Photon_energy, E, ts)
            Filter_transmission = T
            Filter_transmissions.append(Filter_transmission)
            Detector_response = T * self.Sensitivity
            Responses.append(Detector_response)
        if self.add_errors:
            for i in range(0, len(self.Filters), 1):
                if np.shape(self.Filters[i]):
                    Ts_plus = []
                    Ts_minus = []
                    for j in range(0, len(self.Filters[i]), 1):
                        self.filter, self.t, Error = self.Filters[i][j].split()
                        self.err = self.error
                        E, ts = self.get_filter_transmission()
                        Ts_plus.append(np.interp(self.Photon_energy, E, ts))
                        self.err = -self.error
                        E, ts = self.get_filter_transmission()
                        Ts_minus.append(np.interp(self.Photon_energy, E, ts))
                    T_plus = np.prod(Ts_plus, axis=0)
                    T_minus = np.prod(Ts_minus, axis=0)
                else:
                    self.filter, self.t, Error = self.Filters[i].split()
                    self.err = self.error
                    E, ts = self.get_filter_transmission()
                    T_plus = np.interp(self.Photon_energy, E, ts)
                    self.err = -self.error
                    E, ts = self.get_filter_transmission()
                    T_minus = np.interp(self.Photon_energy, E, ts)
                Detector_response_plus = T_plus * self.Sensitivity
                Detector_response_minus = T_minus * self.Sensitivity
                Responses.append(Detector_response_plus)
                Responses.append(Detector_response_minus)
        self.Detector_responses = Responses
        self.Filter_transmissions = Filter_transmissions

        if plot:
            fig, ax = plt.subplots(figsize=(8, 7))
            for i in range(0, len(self.Filters), 1):
                Filter_string = filter_string(self.Filters[i])
                Ar = np.array([self.Photon_energy, Responses[i]]).T
                # np.savetxt('./Detector_responses/{}.csv'.format(filter_string(self.Filters[i], Math=False)), Ar)
                plt.plot(self.Photon_energy, Responses[i], label=r'${}$'.format(Filter_string))
            plt.xlabel('Photon energy (eV)', fontsize=20)
            plt.ylabel('Detector Response (arb. u.)', fontsize=20)
            plt.suptitle('Detector response for filter packs', fontsize=22)
            plt.xlim(0, 5000)
            plt.minorticks_on()
            ax.tick_params(which='major', length=5)
            # ax.grid(color='grey', alpha=0.15)
            ax.tick_params(axis='both', which='major', length=11, direction='in', color='black', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=5, direction='in', color='black', top=True, right=True)
            plt.legend()
            plt.show()

            fig, ax = plt.subplots(figsize=(8, 7))
            for i in range(0, len(self.Filters), 1):
                Filter_string = filter_string(self.Filters[i])
                Ar = np.array([self.Photon_energy, Responses[i]]).T
                plt.plot(self.Photon_energy, Filter_transmissions[i], label=r'${}$'.format(Filter_string))
            plt.xlabel('Photon energy (eV)', fontsize=20)
            plt.ylabel('Filter transmission', fontsize=20)
            plt.suptitle('Filter transmissions for filter packs', fontsize=22)
            plt.xlim(0, 5000)
            plt.minorticks_on()
            ax.tick_params(which='major', length=5)
            # ax.grid(color='grey', alpha=0.15)
            ax.tick_params(axis='both', which='major', length=11, direction='in', color='black', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=5, direction='in', color='black', top=True, right=True)
            plt.legend()
            plt.show()

        return

class Run_single_SPECT3D:
    def __init__(self, Atomic_makeup, Electron_temperature, Electron_density, Size, Spect3D_workspace, Spect3D_program_loc,
                 Propaceos_file, OUTPUT_loc=None, Ion_temperature=None, core=1):
        self.core = core
        self.spect3D_program = Spect3D_program_loc
        self.etemp = np.asarray(Electron_temperature)
        if not Ion_temperature:
            self.itemp = self.etemp
        else:
            self.itemp = np.asarray(Ion_temperature)
        if not OUTPUT_loc:
            self.OUTPUT_loc = os.path.join(os.getcwd(), 'SPECT3D_OUTPUTS')
        else:
            self.OUTPUT_loc = OUTPUT_loc
        # os.system('rm -r {}/*'.format(self.OUTPUT_loc))
        spec_file = os.path.basename(Spect3D_workspace)
        self.spw = os.path.join(self.OUTPUT_loc, spec_file.replace('.spw', f'_{self.core}.spw'))
        os.system(f'scp {Spect3D_workspace} {self.spw}')

        self.atom = Atomic_makeup
        self.nes = np.asarray(Electron_density)
        self.size = np.asarray(Size)
        ATOMIC = Get_Atomic_info(Atomic_makeup=self.atom, Electron_densities=self.nes, Size=self.size)
        self.den = ATOMIC.Mass_densities
        self.areal = ATOMIC.Areal_densities
        self.initial_setup(PRP_file=Propaceos_file, MAU=ATOMIC.Mean_atomic_weight)
        self.photon_energy, self.intensity = self.run()

        os.system(f'rm -r {self.spw}')

    # def __getattr__(self, name):
    #     return getattr(self, name)

    def initial_setup(self, PRP_file, MAU):
        ########## INITIAL SETUP OF WORKSPACE ##########
        ## These parameters remain fixed across
        ## Temperature, Density and Size
        with open(self.spw, 'r') as f:
            Lines = f.readlines()
            New_lines = Lines
            for i in range(0, len(Lines), 1):
                if 'Opacity filepath' in Lines[i]:
                    Line_inputs = Lines[i].rsplit(maxsplit=1)
                    Line_inputs[-1] = PRP_file
                    New_lines[i] = (' '.join(Line_inputs) + '\n')
                elif 'Mean atomic weight' in Lines[i]:
                    Line_inputs = Lines[i].rsplit(maxsplit=1)
                    Line_inputs[-1] = str(MAU)
                    New_lines[i] = (' '.join(Line_inputs) + '\n')
        with open(self.spw, 'w') as w:
            w.writelines(New_lines)
            w.close()
        return

    def input_workspace(self):  # , Electron_temperature = 1, Ion_temperature = 1, Mass_density = 1, Size = 1):
        ########## INPUT Ts, RHO AND SIZE ##########
        ## These parameters are what are changing with
        ## each iteration
        with open(self.spw, 'r') as f:
            Lines = f.readlines()
            New_lines = Lines
            for i in range(0, len(Lines), 1):
                if 'Uniform Temperature' in Lines[i]:
                    if 'Units' in Lines[i]:
                        pass
                    else:
                        Line_inputs = Lines[i].rsplit(maxsplit=1)
                        Line_inputs[-1] = str(self.etemp)
                        New_lines[i] = (' '.join(Line_inputs) + '\n')
                elif 'Ion Temperature' in Lines[i]:
                    if 'Units' in Lines[i]:
                        pass
                    else:
                        Line_inputs = Lines[i].rsplit(maxsplit=1)
                        Line_inputs[-1] = str(self.itemp)
                        New_lines[i] = (' '.join(Line_inputs) + '\n')
                elif 'Uniform Density' in Lines[i]:
                    if 'Units' in Lines[i]:
                        pass
                    else:
                        Line_inputs = Lines[i].rsplit(maxsplit=1)
                        Line_inputs[-1] = str(self.den)
                        New_lines[i] = (' '.join(Line_inputs) + '\n')
                elif 'Characteristic Size' in Lines[i]:
                    Line_inputs = Lines[i].rsplit(maxsplit=1)
                    Line_inputs[-1] = str(self.areal)
                    New_lines[i] = (' '.join(Line_inputs) + '\n')
        with open(self.spw, 'w') as w:
            w.writelines(New_lines)
            w.close()
        return

    def run_workspace(self):
        ########## RUNNING SPECT3D ##########
        ## This runs the Spect3D program (which is in
        ## Spect3D_program_loc location) with the spw
        ## workspace. The output folder is saved to
        ## location 'CWD' and called 'runname'

        if not os.path.exists(self.OUTPUT_loc):
            os.makedirs(self.OUTPUT_loc)
        try:
            ## LINUX Setup
            # Workspace = subprocess.run(['LD_LIBRARY_PATH="$LD_LIBRARY_PATH:{}" {} -b -x -i {} -d {} -o {}'.format(
            #     Library_loc, Spect3D_program_loc,
            #     self.spw, os.path.join(self.OUTPUT_loc), self.runname)], shell=True, stdout=subprocess.PIPE)
            ## Mac Setup
            Workspace = subprocess.run(['{} -b -x -i {} -d {} -o {}'.format(
                self.spect3D_program,
                self.spw, os.path.join(self.OUTPUT_loc), self.runname)], shell=True, stdout=subprocess.PIPE)
        except:
            print('Issue running Spect3D program')
        return

    def run(self):
        ########## RUNNING SPECT3D ##########
        ## This runs the Spect3D program
        self.input_workspace()
        # self.runname = f'Spect3D_output{self.core}'
        self.runname = '{}eV_{}_{}mm'.format(self.etemp, str(self.nes).replace('+', ''), int(self.size*10))
        self.output = os.path.join(self.OUTPUT_loc, self.runname)
        self.run_workspace()
        E, I = self.get_spectra()
        # os.system('rm -r {}'.format(self.output))
        return E, I

    def get_spectra(self):
        ########## EXTRACT SPECT3D SPECTRUM ##########
        ## This extracts the SPECT3D spectrum (.ses files)
        ## from Spect3D_outputs folder
        Spectrum_file = os.path.join(self.output, 'results_01', 't0001', '{}_0001.sis'.format(self.runname))
        Data = np.genfromtxt(Spectrum_file, delimiter='', skip_header=3)
        Photon_energy, Intensity = Data[:, 0], Data[:, 1]
        return Photon_energy, Intensity

class Scattering_Calculations:

    def __init__(self, Atomic_makeup, Electron_temperature, Electron_density, Size, 
                 Spect3D_workspace, Spect3D_program, PROPACEOS_file, OUTPUT_loc):

        self.Atomic_makeup = Atomic_makeup
        self.OUTPUT_loc = OUTPUT_loc
        self.PROPACEOS_file = PROPACEOS_file
        self.Spect3D_workspace = Spect3D_workspace
        self.Spect3D_program = Spect3D_program

        self.Electron_temps = Electron_temperature
        self.Electron_dens = Electron_density
        self.Size = Size

        self.DAT_loc = os.path.join(self.OUTPUT_loc, 'DAT_files')

    def run_parallel_calcs(self):
        self.te_size, self.ne_size = len(self.Electron_temps), len(self.Electron_dens)

        tasks = []
        n_cores = min(cpu_count(), self.te_size * self.ne_size)

        for i, (t, n) in enumerate(np.ndindex(self.te_size, self.ne_size)):
            core_id = i#(i % n_cores) + 1
            tasks.append((
                self.Electron_temps[t],
                self.Electron_dens[n],
                self.Size,
                self.Atomic_makeup,
                self.Spect3D_workspace,
                self.Spect3D_program,
                self.PROPACEOS_file,
                self.OUTPUT_loc,
                core_id,
            ))
        core_ids = [task[-1] for task in tasks]
        start_time = time.perf_counter()

        try:
            with Pool(n_cores) as pool:
                results = pool.map(process_cell, tasks)
        except Exception as e:
            print(f'\n[CRITICAL ERROR] Parallel execution failed: {e}')
            raise  # Let it crash

        self.spect3d_outputs = results

    def run_linear_calcs(self):
        results = []
        for te in self.Electron_temps:
            for ne in self.Electron_dens:
                RUN = Run_single_SPECT3D(Atomic_makeup=self.Atomic_makeup,
                                         Electron_temperature=te, Electron_density=ne,
                                         Size=self.Size, Spect3D_workspace=self.Spect3D_workspace,
                                         Spect3D_program_loc=self.Spect3D_program,
                                         Propaceos_file=self.PROPACEOS_file, OUTPUT_loc=self.OUTPUT_loc)
                E, I = RUN.photon_energy, RUN.intensity
                results.append((te, ne, self.Size, E, I))
        self.spect3d_outputs = results
        
    def create_npz_output(self):
        te_list = []
        ne_list = []
        emission_list = []
        for te, ne, size, pe, spectrum in self.spect3d_outputs:
            te_list.append(te), ne_list.append(ne), emission_list.append(spectrum)
        np.savez(os.path.join(self.OUTPUT_loc, 'ne_te_emission.npz'), te=te_list, ne=ne_list,
                 photon_energy=pe, spectra=emission_list)
        return
    
    def create_dat_output(self):
        if not os.path.exists(self.DAT_loc):
            os.makedirs(self.DAT_loc)

        for te, ne, size, pe, spectrum in self.spect3d_outputs:
            Array = np.asarray([pe, spectrum]).T
            np.savetxt(os.path.join(self.DAT_loc, '{}eV_{}_{}mm.dat'.format(te, str(ne).replace('+', ''), int(size*10))), Array, delimiter='\t', header='Dataset = {}eV_{}_{}mm\n'
                                                                                                                                    'Node centered data\n'
                                                                                                                                    'Number of points = {}\n'
                                                                                                                                    'Photon Energy (eV)\t\tIntensity (erg/cm2/s/eV)'.format(te, str(ne).replace('+', ''), int(size*10), len(pe)))

class Create_Ratiocurves:
    def __init__(self, Filters, Filter_loc, Detector_Sensitivity_loc, SCATTERING_CALC, type='dat', Add_errors=False, err=0.2):

        self.Filter_loc = Filter_loc
        self.Detector_sensitivity = Detector_Sensitivity_loc
        self.Filters = Filters
        self.add_errors = Add_errors
        self.error = err

        self.get_detector_responses()

        self.use_ne = SCATTERING_CALC.Electron_dens
        self.use_Te = SCATTERING_CALC.Electron_temps
        self.size = SCATTERING_CALC.Size
        
        if type == 'dat':
            self.DAT_loc = SCATTERING_CALC.DAT_loc
            self.get_dat_spectra()
        elif type == 'npz':
            self.npz_loc = os.path.join(SCATTERING_CALC.OUTPUT_loc, 'ne_te_emission.npz')
            self.get_npz_spectra()

        # self.counts()
        # self.spectrum()
        # self.filter_signals = self.get_intensity_te_comp()


    def get_detector_responses(self):
        DET_RES = Create_Detector_Response(Filters=self.Filters, Filter_loc=self.Filter_loc,
                                    Detector_Sensitivity_loc=self.Detector_sensitivity,
                                    Add_errors=self.add_errors, err=self.error)
        self.Photon_energy = DET_RES.Photon_energy
        self.Detector_responses = DET_RES.Detector_responses
        self.Sensitivity = DET_RES.Sensitivity
        return
    
    def get_dat_spectra(self):
        ########## EXTRACT DAT SPECTRUM ##########
        ## This extracts .dat spectrum in DAT_loc
        OUTPUTS = os.listdir(self.DAT_loc)
        ALL_SPECTRA = []
        ne_all = []
        te_all = []
        for o in OUTPUTS:
            Temp, Ne = float(o.split('_')[0].replace('eV', '')), float(o.split('_')[1])
            if Ne in self.use_ne and Temp in self.use_Te:
                ne_all.append(Ne)
                te_all.append(Temp)
        te_all = np.asarray(te_all)
        ne_all = np.asarray(ne_all)

        te_list = sorted(np.unique(te_all))
        ne_list = sorted(np.unique(ne_all))
        te_size = len(te_list)
        ne_size = len(ne_list)
        te_index = np.digitize(te_all, te_list) - 1  # bin indices
        ne_index = np.digitize(ne_all, ne_list) - 1
       
        ALL_SPECTRA = np.zeros((te_size, ne_size, len(self.Photon_energy)))
        for t in range(te_size):
            for n in range(ne_size):
                o = '{}eV_{}_{}mm.dat'.format(int(te_list[t]), str(ne_list[n]).replace('+', ''), int(self.size*10))
                if o not in OUTPUTS:
                    print('Error in Create_Ratiocurves\n'
                          '\tGet_dat_spectra => Error finding .dat file\n'
                          '\tCheck .dat file {} exists'.format(o))
                    sys.exit()
                Data = np.genfromtxt(os.path.join(self.DAT_loc, o), skip_header=4)
                E, I = Data[:, 0], Data[:, 1]
                I = np.interp(self.Photon_energy, E, I)
                ALL_SPECTRA[t, n, :] = I


        self.All_nes = ne_list
        self.All_Tes = te_list
        self.spect3d_spectra = ALL_SPECTRA
    
    def get_npz_spectra(self):
        data = np.load(self.npz_loc, allow_pickle=True)
        te_all = data['te']
        ne_all = data['ne']
        spectra = data['spectra']
        pe = data['photon_energy']

        interp_func = interp1d(pe, spectra, kind='linear', axis=1, bounds_error=False, fill_value=np.nan)
        int_spectra = interp_func(self.Photon_energy)

        te_list = sorted(np.unique(te_all))
        ne_list = sorted(np.unique(ne_all))
        te_size = len(te_list)
        ne_size = len(ne_list)

        te_index = np.digitize(te_all, te_list) - 1  # bin indices
        ne_index = np.digitize(ne_all, ne_list) - 1

        ALL_SPECTRA = np.zeros((te_size, ne_size, len(self.Photon_energy)))
        for i in range(0, len(int_spectra), 1):
            spectrum = int_spectra[i]
            t, n = te_index[i], ne_index[i]
            ALL_SPECTRA[t, n, :] = spectrum
        
        self.All_nes = ne_list
        self.All_Tes = te_list
        self.spect3d_spectra = ALL_SPECTRA

    def counts(self):
        self.get_detector_responses()
        self.get_spectra()
        for ne in self.All_nes:
            fig, ax = plt.subplots(figsize=(8, 7))
            for i in range(0, len(self.Filters), 1):
                Filter_spectra = []
                Filter_string = self.filter_string(self.Filters[i])
                for s in range(0, len(self.spect3d_spectra), 1):
                    Info = self.spect3d_spectra[s]
                    if Info[1] == ne and Info[0] == 300:
                        I = Info[2]
                        Spectrum_effect = I * self.Detector_responses[i]
                        Filter_spectra.append(Spectrum_effect)
                Spectrum = np.sum(np.asarray(Filter_spectra), axis=0)
                Photon_Spectrum = Spectrum*1e-7*1e-9*(1e-3**2)/(self.Photon_energy*cst.e)
                Total = np.trapezoid(Photon_Spectrum, x=self.Photon_energy)
                plt.plot(self.Photon_energy, Photon_Spectrum, label=r'${}$'.format(Filter_string))
                print(r'Total signal for ${}$ = {:.3g} photons'.format(Filter_string, Total))
            plt.legend()
            plt.xlabel('Photon energy (eV)')
            plt.ylabel(r'Expected Spectrum (photons/eV)')
            plt.minorticks_on()
            plt.suptitle(str(r'$\mathrm{n_e = }$' + str(ne).replace('+', '') + r'$\,\,\mathrm{g/cm^3}$'))
            ax.tick_params(which='major', length=5)
            # ax.grid(color='grey', alpha=0.15)
            ax.tick_params(axis='both', which='major', length=11, direction='in', color='black', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=5, direction='in', color='black', top=True, right=True)
            # plt.yscale('log')
            # plt.ylim(1e5, 5.2e7)
            plt.xlim(0, 5000)
            plt.savefig(self.DAT_loc.replace('DAT_files', 'Freq_signal.png'), dpi=300)
            plt.show()

    def spectrum(self):
        self.get_detector_responses()
        self.get_spectra()
        for ne in self.All_nes:
            fig, ax = plt.subplots(figsize=(8, 7))
            for s in range(0, len(self.spect3d_spectra), 1):
                Info = self.spect3d_spectra[s]
                if Info[1] == ne:
                    # if Info[0] == 100 or Info[0] == 200 or Info[0] == 300 or Info[0] == 50 or Info[0] == 400:
                    if Info[0] == 1000:
                        I = Info[2]
                        # Intensity = (I*1e-7*1e-9*(1e-3**2)/(self.Photon_energy*cst.e)) #* self.Sensitivity # photons/eV
                        # plt.plot(self.Photon_energy, Intensity, label='{} eV'.format(Info[0]))

                        Intensity = (I * 1e-7 * 1e-9 * (1e-3 ** 2) / (cst.e))  #  photons
                        # plt.plot(self.Photon_energy, Intensity, label='{} eV'.format(Info[0]))
                        plt.plot(self.Photon_energy, Intensity, 'k-', label='Plasma spectral emission')

                        for i in range(0, len(self.Filters), 1):
                            Filter_string = self.filter_string(self.Filters[i])
                            Signal = Intensity * self.Detector_responses[i]
                            plt.plot(self.Photon_energy, Signal, '--', label=r'Spectral response ${}$'.format(Filter_string))


            plt.legend(loc='lower right')
            plt.xlabel('Photon energy (eV)')
            # plt.ylabel('Intensity (photons/eV)')
            plt.ylabel('Intensity (photons)')
            plt.minorticks_on()
            plt.suptitle(str(r'$\mathrm{n_e = }$' + str(ne).replace('+', '') + r'$\,\,\mathrm{g/cm^3}$'))
            ax.tick_params(which='major', length=5)
            # ax.grid(color='grey', alpha=0.15)
            ax.tick_params(axis='both', which='major', length=11, direction='in', color='black', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=5, direction='in', color='black', top=True, right=True)
            plt.yscale('log')
            plt.xscale('log')
            # plt.ylim(1, 1e11)
            plt.xlim(0, 10000)
            # plt.savefig(self.DAT_loc.replace('DAT_files', 'Freq_signal.png'), dpi=300)
            plt.show()

    def get_intensity_te_comp(self, plot=False):
        self.get_detector_responses()
        self.get_spectra()

        if plot:
            fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.get_cmap(name='jet', lut=len(self.All_nes))
        for i in range(len(self.All_nes)):
            ne = self.All_nes[i]
            color = cmap(i)
            Filter_signals = []
            for s in range(0, len(self.spect3d_spectra), 1):
                Info = self.spect3d_spectra[s]
                if Info[1] == ne:
                    I = Info[2]
                    Integrated_spectra = []
                    for filter in range(0, len(self.Filters), 1):
                        Spectrum_effect = I * self.Detector_responses[filter]
                        Integrated_spectrum = np.trapezoid(Spectrum_effect, x=self.Photon_energy)
                        Integrated_spectra.append(Integrated_spectrum)
                    Filter_signals.append((Info[0], Integrated_spectra[0], Integrated_spectra[1]))

            Filter_signals = np.asarray(Filter_signals)
            Filter_signals = Filter_signals[np.argsort(Filter_signals[:, 0])] ## Need to sort in temperature order

            norm = Filter_signals[:, 1][np.where(Filter_signals[:, 0] >= 1000)[0][0]]
            if plot:
                ax.plot(Filter_signals[:, 0], Filter_signals[:, 1]/norm, '-', color=color, label='{}'.format(str(self.All_nes[i]).replace('+', '')))
                ax.plot(Filter_signals[:, 0], Filter_signals[:, 2]/norm, '-.', color=color)

        # Array = Filter_signals
        # Text_name = '{}_{}_vs_{}_signal.txt'.format(str(self.All_nes[0]).replace('+', ''),
        #                                      self.filter_string(self.Filters[0], Math=False),
        #                                      self.filter_string(self.Filters[1], Math=False))
        # np.savetxt(self.DAT_loc.replace('DAT_files', Text_name), Array, header='Temperature (eV)\tFilter 1\tFilter 2')
        if plot:
            leg = ax.legend(title=r'$n_e$ (g/cm$^3$)', loc='upper left', fontsize=16, title_fontsize=16)
            ax.add_artist(leg)
            ax.tick_params(axis='both', which='major', length=11, direction='in', color='black', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=5, direction='in', color='black', top=True, right=True)
            ax.set_ylabel('Normalised Signal Intensity (arb. u.)')
            ax.set_xlabel('Temperature (eV)')
            ax.tick_params(which='both', direction='in', top=True, right=True)
            plt.show()

        return Filter_signals

    def get_ratiocurves(self, plot=False):

        def get_intensity_signals(filter_locs=[0, 1]):
            intensity_signals = np.zeros((len(self.Filters), len(self.All_nes), len(self.All_Tes)))
            for filter in range(len(self.Filters)):
                filter_loc = filter_locs[filter]
                for n in range(0, len(self.All_nes), 1):
                    ne = self.spect3d_spectra[0, n, 0]
                    for t in range(0, len(self.All_Tes), 1):
                        te = self.spect3d_spectra[t, 0, 0]
                        intensity = self.spect3d_spectra[t, n, :]*self.Detector_responses[filter_loc]
                        Integrated_spectrum = np.trapezoid(intensity, x=self.Photon_energy)     
                        intensity_signals[filter, n, t] = Integrated_spectrum


            New_temps = np.arange(np.min(self.All_Tes), np.max(self.All_Tes), 2.5)
            ratios = np.zeros((len(self.All_nes), len(self.All_Tes)))
            interp_ratios = np.zeros((len(self.All_nes), len(New_temps)))
            for n in range(0, len(self.All_nes), 1):
                ne = self.All_nes[n]     
                ratio = intensity_signals[0, n, :]/intensity_signals[1, n, :]
                ratios[n, :] = ratio
                Interp_ratio = interp1d(self.All_Tes, ratio, 'cubic', fill_value='extrapolate')
                interp_ratios[n, :] = Interp_ratio(New_temps)
            return ratios, interp_ratios, New_temps

        ratios, interp_ratios, New_temps = get_intensity_signals(filter_locs=[0, 1])
        if self.add_errors:
            ratios_plus, interp_ratios_plus, _ = get_intensity_signals(filter_locs=[2, 4])
            ratios_minus, interp_ratios_minus, _ = get_intensity_signals(filter_locs=[3, 5])

        if plot:
            fig, ax = plt.subplots(figsize=(8, 7))
            for n in range(0, len(self.All_nes), 1):
                ne = self.All_nes[n]
                color = f"C{n}"  # Use matplotlib's default color cycle
                ax.plot(self.All_Tes, ratios[n, :], 'o', color=color)
                ax.plot(New_temps, interp_ratios[n, :], '-', label='{}'.format(str(ne).replace('+', '')), color=color)
                if self.add_errors:
                    ax.fill_between(New_temps, interp_ratios_minus[n, :], interp_ratios_plus[n, :], color=color, alpha=0.3)
            leg = ax.legend(title=r'$n_e$ (cm$-^3$)', loc='upper right', fontsize=16, title_fontsize=16)
            ax.add_artist(leg)
            ax.tick_params(axis='both', which='major', length=11, direction='in', color='black', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=5, direction='in', color='black', top=True, right=True)
            ax.set_ylabel('Ratio (arb. u.)')
            ax.set_xlabel('Temperature (eV)')
            ax.tick_params(which='both', direction='in', top=True, right=True)
            Filter_string = r'$\frac{{{}}}{{{}}}$'.format(filter_string(self.Filters[0]), filter_string(self.Filters[1]))
            plt.suptitle(('Ratiocurves for Filter pack:\n'+ Filter_string), fontsize=18)
            plt.minorticks_on()
            plt.show()

        self.Te, self.Ratio = New_temps, interp_ratios
        return self.Te, self.Ratio

def process_cell(args):
    try:
        te, ne, size, foil_comp, spw, program, prp, output, core = args
        RUN = Run_single_SPECT3D(Atomic_makeup=foil_comp,
                                 Electron_temperature=te, Electron_density=ne,
                                 Size=size, Spect3D_workspace=spw, Spect3D_program_loc=program,
                                 Propaceos_file=prp, OUTPUT_loc=output, core=core)
        E, I = RUN.photon_energy, RUN.intensity
        return (te, ne, size, E, I)
    except Exception as e:
        print(f'[ERROR] process_cell failed at Te={args[0]}, ne={args[1]}: {e}')
        raise  # Re-raise to crash the pool


## This code can be run on its own to perform SPECT3D calculations
## and create ratiocurves for user-specified filters
## The user inputs are defined in the __main__ section
## Or it can be imported as a module to use the classes
if __name__ == "__main__":
    #%%
    ## Define user inputs (file locations, filters, foil composition, parameter space to explore)
    ###############################################################
    ##                                                           ##
    ##                      FILE LOCATIONS                       ##
    ##                                                           ##
    ###############################################################

    PARENT_loc = os.path.join(os.getcwd())

    ## Location of the Spect3D program executable - this is for a Mac
    Spect3D_program_loc = os.path.join('/', 'Applications', 'Spect3D', 'Spect3D.app', 'Contents', 'MacOS', 'Spect3D')
    ## Location of the single-cell baseline Spect3D workspace file
    Spect3D_workspace = os.path.join(PARENT_loc, 'single_cell.spw')

    ## Location of the PROPACEOS files
    PROPACEOS_loc = os.path.join(PARENT_loc, 'EOS')
    
    ## Location of the filter transmission files - calculated from CXRO website
    FILTER_loc = os.path.join(PARENT_loc, 'Filters')
    
    ## Location of the detector sensitivity file - this is for the GXD on OMEGA and NIF
    DETECTOR_SENSITIVITY_loc = os.path.join(PARENT_loc, 'Sensitivities', 'GXD_Sensitivity.csv')
    
    
    ###############################################################
    ##                                                           ##
    ##                      PARAMETERS                           ##
    ##                                                           ##
    ###############################################################

    ## The foil composition should be written as:
    ## ['<CHEMICAL_FORMULA_1> <FRACTION_1>', '<CHEMICAL_FORMULA_2> <FRACTION_2>', etc.]
    Foil_composition = ['C 0.499', 'H 0.438', 'Cl 0.063']
    OUTPUT_file_loc = os.path.join(PARENT_loc, 'Ratiocurves', 'CHCl')
    PROPACEOS_file = os.path.join(PROPACEOS_loc, 'C49_9H43_8Cl6_3.prp')
    if not os.path.exists(OUTPUT_file_loc):
        os.makedirs(OUTPUT_file_loc)
    if not os.path.exists(PROPACEOS_file):
        raise FileNotFoundError(f'PROPACEOS file not found: {PROPACEOS_file}')

    ## Parameter space to explore using Spect3D
    Spect3D_areal_size = 0.1 # cm
    ## OMEGA Te Resolution
    Electron_temps = np.concatenate([np.arange(20, 150, 1), np.arange(150, 1000, 10), np.arange(1000, 3000, 100), np.arange(3000, 5010, 1000)]) # eV
    ## NIF Te Resolution
    # Electron_temps = np.concatenate([np.arange(50, 1000, 10), np.arange(1000, 3000, 25), np.arange(3000, 5010, 50)]) # eV
    Electron_temps = [100, 200, 400, 500, 800, 900, 1200]

    ## Electron density resolution
    # Electron_dens = [8e19, 9e19, 1e20, 2e20, 3e20, 4e20, 5e20, 6e20, 7e20, 8e2x0, 9e20, 1e21] # cm^-3
    Electron_dens = [1e20, 5e20] # cm^-3

    ## This code can only work with two filters.
    ## The filters should be written as:
    ## [[Filter_pack_1], [Filter_pack_2]]
    ## Each filter pack should be writtin as:
    ## ['<FILTER_1> <FILTER_1_THICKNESS_μm> <FILTER_1_THICKNESS_FRACTION_ERROR>', etc.]
    Filters = [['Mylar 1 0', 'V 0.2 0', 'Al 0.8 0'], ['Mylar 2 0', 'V 0.2 0', 'Al 0.8 0']]

    ###############################################################

    #%%
    ## Running SPECT3D calculations
    ## This runs SPECT3D for foil composition for all specified temperature and densities
    ## and saves the run folders in OUTPUT_loc
    ## Perform the initial setup
    CALCS = Scattering_Calculations(Atomic_makeup=Foil_composition, OUTPUT_loc=OUTPUT_file_loc,
                    Electron_temperature=Electron_temps, Electron_density=Electron_dens,
                    Size=Spect3D_areal_size, Spect3D_workspace=Spect3D_workspace, Spect3D_program=Spect3D_program_loc,
                    PROPACEOS_file=PROPACEOS_file)
    ## Run the calculations in parallel
    # CALCS.run_parallel_calcs()
    # ## Run the calculations linear - (use if parallel fails)
    # # CALCS.run_linear_calcs()
    
    # ## Choose how to save the outputs
    # CALCS.create_npz_output()
    # CALCS.create_dat_output()

    #%%
    ## Create Ratiocurves
    ## This creates ratiocurves for user-specified filters
    ## Interpolates spectra either saved in DAT_loc, or in the npz file
    ## The Add_errors bool calculates and plots the ratiocurves for filter packs with +/-20% thickness error (can be changed by pass err=0.xx to Create_Ratiocurves)
    CREATE = Create_Ratiocurves(Filters=Filters, Filter_loc=FILTER_loc, Detector_Sensitivity_loc=DETECTOR_SENSITIVITY_loc,
                    SCATTERING_CALC=CALCS, type='npz', Add_errors=True)
    
    CREATE.get_ratiocurves()
    ## There are other functions in the Create_Ratiocurves class that I need to tidy up...