import pygsp
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from pydoc import locate
import concurrent.futures

class WaveletMachine:
    """
    An implementation of "Learning Structural Node Embeddings Via Diffusion Wavelets".
    """
    def __init__(self, G, settings):
        self.index = list(G.nodes())  
        self.G = pygsp.graphs.Graph(nx.adjacency_matrix(G))
        self.number_of_nodes = len(self.index)
        self.settings = settings
        if self.number_of_nodes > self.settings.switch:
            self.settings.mechanism = "approximate"

        self.steps = [x*self.settings.step_size for x in range(self.settings.sample_number)]

    def single_wavelet_generator(self, node):
        impulse = np.zeros((self.number_of_nodes))
        impulse[node] = 1.0
        diags = np.diag(np.exp(-self.settings.heat_coefficient*self.eigen_values))
        eigen_diag = np.dot(self.eigen_vectors, diags)
        waves = np.dot(eigen_diag, np.transpose(self.eigen_vectors))
        wavelet_coefficients = np.dot(waves, impulse)
        return wavelet_coefficients

    def exact_wavelet_calculator(self, target_nodes):
        """
        Calculates the structural role embedding for target nodes using the exact eigenvalue decomposition.
        """
        self.real_and_imaginary = []
        for node in tqdm(target_nodes):
            wave = self.single_wavelet_generator(self.index.index(node))
            wavelet_coefficients = [np.mean(np.exp(wave*1.0*step*1j)) for step in self.steps]
            self.real_and_imaginary.append(wavelet_coefficients)
        self.real_and_imaginary = np.array(self.real_and_imaginary)

    def exact_structural_wavelet_embedding(self, target_nodes):
        self.G.compute_fourier_basis()
        self.eigen_values = self.G.e / max(self.G.e)
        self.eigen_vectors = self.G.U
        self.exact_wavelet_calculator(target_nodes)

    def approximate_wavelet_calculator(self, target_nodes, max_workers=1):
        self.real_and_imaginary = []
        if not max_workers:
            for node in tqdm(target_nodes):
                self.real_and_imaginary.append(self._load_real_imag(self.index.index(node)))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._load_real_imag, self.index.index(node))
                    for node in target_nodes
                ]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    real_imag = future.result()
                    self.real_and_imaginary.append(real_imag)

        self.real_and_imaginary = np.array(self.real_and_imaginary)

    def _load_real_imag(self, node: int):
        impulse = np.zeros((self.number_of_nodes))
        impulse[node] = 1
        wave_coeffs = pygsp.filters.approximations.cheby_op(self.G, self.chebyshev, impulse)
        real_imag = [np.mean(np.exp(wave_coeffs * 1 * step * 1j)) for step in self.steps]
        return real_imag

    def approximate_structural_wavelet_embedding(self, target_nodes):
        self.G.estimate_lmax()
        self.heat_filter = pygsp.filters.Heat(self.G, tau=[self.settings.heat_coefficient])
        self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter, m=self.settings.approximation)
        self.approximate_wavelet_calculator(target_nodes)

    def create_embedding(self, target_nodes=None):
        """
        Depending on the mechanism setting, creating an exact or approximate embedding for target nodes.
        """
        if self.settings.mechanism == "exact":
            self.exact_structural_wavelet_embedding(target_nodes)
        else:
            self.approximate_structural_wavelet_embedding(target_nodes)

    def transform_and_save_embedding(self, target_nodes):
        """
        Transforming the numpy array with real and imaginary values.
        Creating a pandas dataframe and saving it as a csv.
        """
        print("\nSaving the embedding.")
        features = [self.real_and_imaginary.real, self.real_and_imaginary.imag]
        self.real_and_imaginary = np.concatenate(features, axis=1)
        columns_1 = ["reals_" + str(x) for x in range(self.settings.sample_number)]
        columns_2 = ["imags_" + str(x) for x in range(self.settings.sample_number)]
        columns = columns_1 + columns_2
        self.real_and_imaginary = pd.DataFrame(self.real_and_imaginary, columns=columns)

        
        self.real_and_imaginary.index = target_nodes
        self.real_and_imaginary.index = self.real_and_imaginary.index.astype(str)
        self.real_and_imaginary.to_csv(self.settings.output)

    def get_real_and_imaginary_results(self):
        return np.concatenate([self.real_and_imaginary.real, self.real_and_imaginary.imag], axis=1)
