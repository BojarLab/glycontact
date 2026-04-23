from collections import defaultdict
import os
import copy
import time
from pathlib import Path
from typing import Literal
import numpy as np
import networkx as nx
from glycowork.glycan_data.loader import HashableDict, lib

from glycontact.process import get_all_clusters_frequency, get_structure_graph, get_global_path

# Try to import optional ML dependencies
try:
    import torch
    import torch_geometric
    from torch_geometric.nn import GINConv
except ImportError:
    raise ImportError(
        "Missing required dependencies for machine learning functionality. "
        "Please install glycontact with ML support: pip install glycontact[ml] "
        "or pip install -e git+https://github.com/BojarLab/glycontact.git#egg=glycontact[ml]"
    )


def get_all_structure_graphs(glycan, stereo = None, libr = None):
    """Get all structure graphs for a given glycan.
    Args:
        glycan (str): The glycan name.
        stereo (str, optional): The stereochemistry. If None, both alpha and beta are returned.
        libr (HashableDict, optional): A library of structures. If None, the default library is used.
    Returns:
        list: A list of tuples containing the PDB file name and the corresponding structure graph.
    """
    libr = HashableDict(libr)
    global_path = get_global_path()
    if stereo is None:
        return get_all_structure_graphs(glycan, "alpha", libr) + get_all_structure_graphs(glycan, "beta", libr)
    matching_pdbs = [global_path / glycan / pdb for pdb in sorted(os.listdir(global_path / glycan)) if stereo in pdb]
    return [(pdb, get_structure_graph(glycan, libr = libr, example_path = pdb)) for pdb in matching_pdbs]


def node2y(attr):
    """Extract ML task labels from node attributes.
    Args:
        attr (dict): Node attributes.
    Returns:
        list: A list of labels for the node. If all labels are zero, returns None.
    """
    omega = attr.get("omega_angle", 0)
    if omega is None or omega != omega:
        omega = 0
    output = [
        attr.get("phi_angle", 0),
        attr.get("psi_angle", 0),
        attr.get("SASA", 0),
        attr.get("flexibility", 0),
        omega,
    ]
    if output == [0, 0, 0, 0, 0]:
        return None
    return output


def graph2pyg(g, weight, iupac, conformer):
    """Convert a structure graph to a PyTorch Geometric Data object.
    Args:
        g (networkx.Graph): The structure graph.
        weight (float): The weight of the graph.
        iupac (str): The IUPAC name of the glycan.
        conformer (str): The conformer name.
    Returns:
        torch_geometric.data.Data: The PyTorch Geometric Data object.
    """
    x, y = [], []
    for n in range(len(g.nodes)):
        x.append(lib.get(g.nodes[n]["string_labels"], 0))
        y.append(labels := node2y(g.nodes[n]))
        if labels is None:  # Skip if all labels are zero, i.e., the graph is invalid or broken
            return None
    edge_index = [], []
    for edge in g.edges():
        edge_index[0].append(edge[0])
        edge_index[1].append(edge[1])
        # for bidirectionality
        edge_index[0].append(edge[1])
        edge_index[1].append(edge[0])
    return torch_geometric.data.Data(
        x = torch.tensor(x),
        y = torch.tensor(y),
        edge_index = torch.tensor(edge_index).long(),
        weight = weight,
        iupac = iupac,
        conformer = conformer,
    )


def create_dataset(fresh: bool = True, splits: list[float] = [0.8, 0.2]):
    """Create a dataset of PyTorch Geometric Data objects from the structure graphs of glycans.
    Args:
        fresh (bool): If True, fetches the latest data. If False, uses cached data.
        splits (list): A list of two or three floats representing the train-test split ratios.
    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    if len(splits) not in {2, 3}:
        raise ValueError("splits must be a list of two or three floats. More partitions are not supported yet.")
    # Get all clusters and their frequencies.
    data = {}
    for iupac, freqs in get_all_clusters_frequency(fresh = fresh).items():
        try:
            pygs = []
            broken = False
            graphs = get_all_structure_graphs(iupac, None, lib)
            for pathname, graph in graphs:
                # Get the weight of the graph based on the cluster frequency and convert the graph to a PyG Data object.
                idx = int(pathname.stem.split("_")[0].replace("cluster", ""))
                weight = freqs[idx] if idx < len(freqs) else 0.1
                pyg = graph2pyg(graph, weight, iupac, pathname.stem)
                if pyg is None:
                    continue
                pygs.append((pyg, graph))
            # Normalize the weights of the graphs and assign them to the PyG Data objects.
            weights = np.array([pyg.weight for pyg, _ in pygs])
            weights = weights / np.sum(weights)
            for (pyg, _), weight in zip(pygs, weights):
                pyg.weight = torch.tensor([weight])
            data[iupac] = pygs
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error for {iupac}: {e}")
    # Split the data into training and testing sets using DataSAIL. The glycans are weights based on their number of conformers.
    # Try to import DataSAIL
    try:
        from datasail.sail import datasail
    except ImportError:
        raise ImportError(
            "DataSAIL is required for some functionality but not found. "
            "Please install from: https://github.com/kalininalab/DataSAIL"
        )
    names = ["train", "test"] if len(splits) == 2 else ["train", "val", "test"]
    e_splits, _, _ = datasail(
        techniques = ["I1e"],
        names = names,
        splits = splits,
        e_type = "O",
        e_data = [(d, d) for d in data.keys()],
        e_weights = {d: len(c) for d, c in data.items()},
    )
    # Create the training(, validation) and testing datasets.
    train, val, test = [], [], []
    for name, split in e_splits["I1e"][0].items():
        if split == "train":
            train.extend(data[name])
        elif split == "val":
            val.extend(data[name])
        elif split == "test":
            test.extend(data[name])
    if len(splits) == 3:
        return train, val, test
    return train, test


def mean_conformer(conformers: list[tuple[float, tuple[torch_geometric.data.Data, nx.DiGraph]]]) -> tuple[torch_geometric.data.Data, nx.DiGraph]:
    """Calculate the mean conformer from a list of conformers.
    Args:
        conformers (list): A list of tuples containing the weight and the structure graph.
    Returns:
        tuple: A tuple containing the mean PyTorch Geometric Data object and the mean structure graph.
    """
    G = copy.deepcopy(conformers[0][1][1])
    weights, graphs = zip(*conformers)
    # normalize weights
    weights = [w / sum(weights) for w in weights]
    for i, (weight, (_, nxg)) in enumerate(zip(weights, graphs)):
        for node in nxg.nodes:
            if "phi_angle" in nxg.nodes[node]:
                if i == 0:
                    G.nodes[node]["phi_angle"] = (weight * nxg.nodes[node]["phi_angle"])
                    G.nodes[node]["psi_angle"] = (weight * nxg.nodes[node]["psi_angle"])
                else:
                    G.nodes[node]["phi_angle"] += weight * nxg.nodes[node]["phi_angle"]
                    G.nodes[node]["psi_angle"] += weight * nxg.nodes[node]["psi_angle"]
            elif "SASA" in nxg.nodes[node]:
                if i == 0:
                    G.nodes[node]["SASA"] = (weight * nxg.nodes[node]["SASA"])
                    G.nodes[node]["flexibility"] = (weight * nxg.nodes[node]["flexibility"])
                else:
                    G.nodes[node]["SASA"] += weight * nxg.nodes[node]["SASA"]
                    G.nodes[node]["flexibility"] += weight * nxg.nodes[node]["flexibility"]
    return graph2pyg(G, 1, conformers[0][1][0].iupac, conformers[0][1][0].iupac + "_mean"), G


def clean_split(split: list[tuple[torch_geometric.data.Data, nx.DiGraph]], mode: Literal["mean", "max"] = "max") -> tuple[torch_geometric.data.Data, nx.DiGraph]:
    """Clean the split data by condensing it to one conformer per glycan.
    Args:
        split (list): A list of tuples containing the PyTorch Geometric Data object and the structure graph.
        mode (str): The mode for condensing the data. "mean" for mean conformer, "max" for maximum weight conformer.
    Returns:
        list: A list of tuples containing the condensed PyTorch Geometric Data object and the structure graph.
    """
    data = defaultdict(list)
    for pyg, nxg in split:
        data[pyg.iupac].append((int(pyg.weight.item()), (pyg, nxg)))
    if mode == "max":
        return [max(d, key=lambda x: x[0])[1] for d in data.values()]
    return [mean_conformer(d) for d in data.values()]


class GINSweetNet(torch.nn.Module):
    def __init__(
            self, 
            lib_size: int, # number of unique tokens for graph nodes
            num_classes: int = 1, # number of output classes (>1 for multilabel)
            hidden_dim: int = 128, # dimension of hidden layers
            num_components: int = 5 # number of components in the mixture models
        ) -> None:
        "given glycan graphs as input, predicts properties via a graph neural network"
        super(GINSweetNet, self).__init__()
        # Node embedding
        self.item_embedding = torch.nn.Embedding(num_embeddings = lib_size + 1, embedding_dim = hidden_dim)
        # Output layers for mixture model parameters
        self.num_components = num_components
        self.num_classes = num_classes  # Currently ignored
        # Convolution operations on the graph (Backbone)
        self.body = torch.nn.Sequential(
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p = 0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p = 0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p = 0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.Dropout(p = 0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 4),
            torch.nn.BatchNorm1d(hidden_dim // 4),
            torch.nn.Dropout(p = 0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 4, self.num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        Args:
            x: Input node features [batch_size, num_nodes, hidden_dim]
            edge_index: Edge indices for the graph [2, num_edges]
        Returns:
            Tuple of 
            weights_logits: Logits for mixture weights [batch_size, 2, num_components]
            means: Mean angles in degrees [batch_size, 2, num_components]
            kappas: Concentration parameters [batch_size, 2, num_components]
            sasa_pred: Predicted SASA values [batch_size]
            flex_pred: Predicted flexibility values [batch_size]
        """
        x = self.item_embedding(x)
        for layer in self.body:
            x = layer(x, edge_index)
        x = self.head(x)
        phi_pred = x[:, 0]
        psi_pred = x[:, 1]
        sasa_pred = x[:, 2]
        flex_pred = x[:, 3]
        return (phi_pred, psi_pred), sasa_pred, flex_pred


class VonMisesSweetNet(torch.nn.Module):
    def __init__(
            self, 
            lib_size: int, # number of unique tokens for graph nodes
            num_classes: int = 1, # number of output classes (>1 for multilabel)
            hidden_dim: int = 128, # dimension of hidden layers
            num_components: int = 5 # number of components in the mixture models
        ) -> None:
        "given glycan graphs as input, predicts properties via a graph neural network"
        super(VonMisesSweetNet, self).__init__()
        # Node embedding
        self.item_embedding = torch.nn.Embedding(num_embeddings = lib_size + 1, embedding_dim = hidden_dim)
        # Output layers for mixture model parameters
        self.num_components = num_components
        self.num_classes = num_classes  # Currently ignored
        # Convolution operations on the graph (Backbone)
        self.body = torch.nn.Sequential(
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p = 0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p = 0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
            GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.Dropout(p = 0.3),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
            )),
        )
        # Classification head for von Mises-distributed properties (phi and psi)
        self.head_von_mises = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.Dropout(p = 0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
        )
        # For each torsion angle (phi, psi), predict mixture weights, means, and kappas
        self.fc_weights_von_mises = torch.nn.Linear(hidden_dim // 2, 2 * num_components)  # Logits for mixture weights
        self.fc_means_von_mises = torch.nn.Linear(hidden_dim // 2, 2 * num_components)  # Mean angles
        self.fc_kappas_von_mises = torch.nn.Linear(hidden_dim // 2, 2 * num_components)  # Concentration parameters
        # Classification head for Gaussian-distributed properties (SASA and flexibility)
        self.head_values = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 4),
            torch.nn.BatchNorm1d(hidden_dim // 4),
            torch.nn.Dropout(p = 0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 4, 2),
        )
    
    def predict_von_mises_parameters(
            self, 
            x: torch.Tensor, 
            head: torch.nn.Module, 
            fc_weights: torch.nn.Module, 
            fc_means: torch.nn.Module, 
            fc_kappas: torch.nn.Module
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict mixture parameters for a given input tensor.
        Args:
            x: Input tensor [batch_size, hidden_dim]
            head: Head module for the mixture model
            fc_weights: Fully connected layer for weights
            fc_means: Fully connected layer for means
            fc_kappas: Fully connected layer for kappas
        Returns:
            Tuple of 
            weights_logits: Logits for mixture weights [batch_size, 2, num_components]
            means: Mean angles in degrees [batch_size, 2, num_components]
            kappas: Concentration parameters [batch_size, 2, num_components]
        """
        x = head(x)
        weights_logits = fc_weights(x)  # [batch_size, 2 * num_components]
        means = fc_means(x)  # [batch_size, 2 * num_components]
        kappas_raw = fc_kappas(x)  # [batch_size, 2 * num_components]
        # Reshape parameters to separate phi and psi
        batch_size = x.size(0)
        weights_logits = weights_logits.view(batch_size, 2, self.num_components)
        means = means.view(batch_size, 2, self.num_components)
        # Convert means to proper angle range (-180 to 180)
        means = torch.tanh(means) * 180.0
        # Ensure kappas are positive using softplus
        kappas = torch.nn.functional.softplus(kappas_raw.view(batch_size, 2, self.num_components))
        return weights_logits, means, kappas

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        Args:
            x: Input node features [batch_size, num_nodes, hidden_dim]
            edge_index: Edge indices for the graph [2, num_edges]
        Returns:
            Tuple of 
            weights_logits: Logits for mixture weights [batch_size, 2, num_components]
            means: Mean angles in degrees [batch_size, 2, num_components]
            kappas: Concentration parameters [batch_size, 2, num_components]
            sasa_pred: Predicted SASA values [batch_size]
            flex_pred: Predicted flexibility values [batch_size]
        """
        x = self.item_embedding(x)
        for layer in self.body:
            x = layer(x, edge_index)
        weights_logits_von_mises, means_von_mises, kappas_von_mises = self.predict_von_mises_parameters(
            x, self.head_von_mises, self.fc_weights_von_mises, self.fc_means_von_mises, self.fc_kappas_von_mises
        )
        values = self.head_values(x)
        sasa_pred = values[:, 0]
        flex_pred = values[:, 1]
        return (weights_logits_von_mises, means_von_mises, kappas_von_mises), sasa_pred, flex_pred

    
def mixture_von_mises_nll(angles: torch.Tensor, weights_logits: torch.Tensor, mus: torch.Tensor, kappas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative log-likelihood for mixture of von Mises distributions
    Args:
        angles: True angles in degrees [batch_size, 2] (phi, psi)
        weights_logits: Raw logits for mixture weights [batch_size, 2, n_components]
        mus: Mean angles in degrees [batch_size, 2, n_components]
        kappas: Concentration parameters [batch_size, 2, n_components]
    Returns:
        Negative log-likelihood
    """
    # Convert angles to radians
    angles_rad = angles * (np.pi / 180.0)
    mus_rad = mus * (np.pi / 180.0)
    # Normalize weights along component dimension
    weights = torch.nn.functional.softmax(weights_logits, dim = 2)
    total_log_probs = []
    # Compute for phi and psi separately
    for angle_idx in range(angles.size(1)):
        # Extract values for this angle (phi or psi)
        angle_rad = angles_rad[:, angle_idx].unsqueeze(1)  # [batch_size, 1]
        angle_mu = mus_rad[:, angle_idx, :]  # [batch_size, n_components]
        angle_kappas = kappas[:, angle_idx, :]  # [batch_size, n_components]
        angle_weights = weights[:, angle_idx, :]  # [batch_size, n_components]
        # Compute von Mises PDF for each component
        # Using the formula: exp(kappa * cos(x - mu)) / (2*pi*I0(kappa))
        # For numerical stability, we approximate log(I0(kappa))
        cos_term = torch.cos(angle_rad - angle_mu)  # [batch_size, n_components]
        log_bessel = torch.log(torch.exp(angle_kappas) / torch.sqrt(2 * np.pi * angle_kappas + 1e-10))
        log_von_mises = angle_kappas * cos_term - np.log(2 * np.pi) - log_bessel
        # Apply weights and compute mixture log probability using logsumexp for numerical stability
        weighted_log_probs = torch.log(angle_weights + 1e-10) + log_von_mises  # [batch_size, n_components]
        angle_log_prob = torch.logsumexp(weighted_log_probs, dim = 1)  # [batch_size]
        total_log_probs.append(angle_log_prob)  # Sum log probabilities across angles
    # Return negative mean log-likelihood
    return -torch.mean(total_log_probs[0]), -torch.mean(total_log_probs[1])


def build_baselines(data: list[nx.DiGraph], fn: callable = np.mean) -> tuple[callable, callable, callable, callable]:
    """Build baseline functions to predict SASA, flexibility, phi, and psi angles based on monosaccharides.
    Args:
        data: List of structure graphs.
        fn: Function to aggregate values (e.g., np.mean, np.median).
    Returns:
        Tuple of functions for phi, psi, SASA, and flexibility.
    """
    sasa, flex, phi, psi = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for g in data:
        for n in g.nodes:
            if "SASA" in g.nodes[n]:
                name = g.nodes[n]["string_labels"]
                sasa[name].append(g.nodes[n]["SASA"])
                flex[name].append(g.nodes[n]["flexibility"])
            elif "phi_angle" in g.nodes[n]:
                name = (
                    g.nodes[list(g.predecessors(n))[0]]["string_labels"], 
                    g.nodes[list(g.successors(n))[0]]["string_labels"]
                )
                phi[name].append(g.nodes[n]["phi_angle"])
                psi[name].append(g.nodes[n]["psi_angle"])
    for pred in [sasa, flex, phi, psi]:
        default = []
        for k, v in pred.items():
            default += v
            pred[k] = fn(v)
        pred["default"] = fn(default)
    return lambda x: phi.get(x, phi["default"]), \
        lambda x: psi.get(x, psi["default"]), \
        lambda x: sasa.get(x, sasa["default"]), \
        lambda x: flex.get(x, flex["default"])


def sample_angle(weights: torch.Tensor, mus: torch.Tensor, kappas: torch.Tensor) -> torch.Tensor:
    """Sample an angle from a mixture of von Mises distributions.
    Args:
        weights: Mixture weights [n_components]
        mus: Mean angles in degrees [n_components]
        kappas: Concentration parameters [n_components]
    Returns:
        Sampled angle in degrees
    """
    idx = np.random.choice(len(weights), p = weights)
    mu = (mus[idx] * np.pi / 180.0) % (2 * np.pi)
    if mu > np.pi:
        mu -= 2 * np.pi
    angle_sample = torch.distributions.von_mises.VonMises(mu, kappas[idx] + 1e-10).sample()
    angle_sample = angle_sample * 180.0 / np.pi
    return angle_sample


def sample_from_model(model: torch.nn.Module, structures: list[torch_geometric.data.Data, nx.DiGraph], count: int = 10):
    """Sample from the model using the provided structures
    Args:
        model: The trained model
        structures: List of structure graphs
    Returns:
        List of sampled angles
    """
    sampled_structures = []
    if isinstance(model, VonMisesSweetNet):
        model.eval()
    with torch.no_grad():
        for i, (data, graph) in enumerate(structures):
            print(f"\r{i + 1} / {len(structures)}", end = "")
            for _ in range(count):
                angular_pred, sasa_pred, flex_pred = model(data.x.to("cuda"), data.edge_index.to("cuda"))
                G = copy.deepcopy(graph)
                for n, node in enumerate(G.nodes):
                    if "phi_angle" in G.nodes[node]:
                        if isinstance(model, GINSweetNet):
                            phi_pred, psi_pred = angular_pred
                            G.nodes[node]["phi_angle"] = phi_pred[n].item()
                            G.nodes[node]["psi_angle"] = psi_pred[n].item()
                        else:
                            weights_logits_von_mises, mus_von_mises, kappas_von_mises = angular_pred
                            weights_von_mises = torch.nn.functional.softmax(weights_logits_von_mises, dim = 2).cpu().numpy()
                            G.nodes[node]["phi_angle"] = sample_angle(weights_von_mises[n, 0], mus_von_mises[n, 0], kappas_von_mises[n, 0]).item()
                            G.nodes[node]["psi_angle"] = sample_angle(weights_von_mises[n, 1], mus_von_mises[n, 1], kappas_von_mises[n, 1]).item()
                    elif "SASA" in G.nodes[node]:
                        G.nodes[node]["SASA"] = sasa_pred[n].item()
                        G.nodes[node]["flexibility"] = flex_pred[n].item()
                sampled_structures.append(G)
    print()
    return sampled_structures


def eval_baseline(nxgraphs: list[nx.DiGraph], phi_pred: callable, psi_pred: callable, sasa_pred: callable, flex_pred: callable) -> list[nx.DiGraph]:
    """Evaluate the baseline model by predicting angles and properties for each graph.
    Args:
        nxgraphs: List of structure graphs
        phi_pred: Function to predict phi angles
        psi_pred: Function to predict psi angles
        sasa_pred: Function to predict SASA
        flex_pred: Function to predict flexibility
    Returns:
        List of predicted structure graphs
    """
    predicted_structures = []
    for graph in nxgraphs:
        pred = copy.deepcopy(graph)
        for node in graph.nodes:
            if "phi_angle" in graph.nodes[node]:
                name = (
                    graph.nodes[list(graph.predecessors(node))[0]]["string_labels"], 
                    graph.nodes[list(graph.successors(node))[0]]["string_labels"]
                )
                pred.nodes[node]["phi_angle"] = phi_pred(name)
                pred.nodes[node]["psi_angle"] = psi_pred(name)
            if "SASA" in graph.nodes[node]:
                name = graph.nodes[node]["string_labels"]
                pred.nodes[node]["SASA"] = sasa_pred(name)
                pred.nodes[node]["flexibility"] = flex_pred(name)
        predicted_structures.append(pred)
    return predicted_structures


def angular_rmse(predicted_graphs: list[nx.DiGraph], true_graphs: list[nx.DiGraph]) -> tuple[float, float]:
    """Calculate the root mean square error (RMSE) for phi and psi angles.
    Args:
        predicted_graphs: List of predicted structure graphs
        true_graphs: List of true structure graphs
    Returns:
        Tuple of RMSE for phi and psi angles
    """
    count = len(predicted_graphs) / len(true_graphs)
    phi_preds, psi_preds, phi_labels, psi_labels = [], [], [], []
    for i, true_g in enumerate(true_graphs):
        for pred_g in predicted_graphs[int(i * count) : int((i + 1) * count)]:
            for node in pred_g.nodes:
                if "phi_angle" in pred_g.nodes[node]:
                    phi_preds.append(pred_g.nodes[node]["phi_angle"])
                    phi_labels.append(true_g.nodes[node]["phi_angle"])
                    psi_preds.append(pred_g.nodes[node]["psi_angle"])
                    psi_labels.append(true_g.nodes[node]["psi_angle"])
    phi_rmse, psi_rmse = periodic_rmse(torch.stack([torch.tensor(phi_preds), torch.tensor(psi_preds)], dim = 1), torch.stack([torch.tensor(phi_labels), torch.tensor(psi_labels)], dim = 1))
    return torch.mean(phi_rmse).item() * 180 / np.pi, torch.mean(psi_rmse).item() * 180 / np.pi


def value_rmse(predicted_graphs: list[nx.DiGraph], true_graphs: list[nx.DiGraph], name: Literal["SASA", "flexibility"]) -> float:
    """Calculate the root mean square error (RMSE) for a specific property (SASA or flexibility).
    Args:
        predicted_graphs: List of predicted structure graphs
        true_graphs: List of true structure graphs
        name: The property to calculate RMSE for (e.g., "SASA" or "flexibility")
    Returns:
        RMSE value
    """
    count = len(predicted_graphs) / len(true_graphs)
    preds, labels = [], []
    for i, true_g in enumerate(true_graphs):
        for pred_g in predicted_graphs[int(i * count) : int((i + 1) * count)]:
            for node in pred_g.nodes:
                if name in pred_g.nodes[node]:
                    preds.append(pred_g.nodes[node][name])
                    labels.append(true_g.nodes[node][name])
    return np.sqrt(np.mean((np.array(preds) - np.array(labels)) ** 2))


def evaluate_model(model: torch.nn.Module | tuple[callable, callable, callable, callable], structures, count: int = 10):
    """Evaluate the model by sampling angles and properties from the structure graphs.
    Args:
        model: The trained model. This can be a trained SweetNet or a tuple of baseline predictors for phi, psi, SASA, and flexibility.
        structures: List of structure graphs
        count: Number of samples to generate for each graph
    Returns:
        Tuple of RMSE values for phi, psi, SASA, and flexibility
    """
    nx_structures = [s[1] for s in structures]
    if isinstance(model, torch.nn.Module):
        predictions = sample_from_model(model, structures, count)
    else:
        predictions = eval_baseline(nx_structures, model[0], model[1], model[2], model[3])
    phi_rmse, psi_rmse = angular_rmse(predictions, nx_structures)
    sasa_rmse = value_rmse(predictions, nx_structures, "SASA")
    flex_rmse = value_rmse(predictions, nx_structures, "flexibility")
    return phi_rmse, psi_rmse, sasa_rmse, flex_rmse, predictions


def periodic_mse(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the periodic mean squared error (MSE) for angles.
    Args:
        pred: Predicted angles in degrees [batch_size, 2]
        target: True angles in degrees [batch_size, 2
    Returns:
        Tuple of MSE for phi and psi angles
    """
    # Convert angles to radians for easier calculation
    pred_rad = pred * (np.pi / 180.0)
    target_rad = target * (np.pi / 180.0)
    # Calculate the difference using sine and cosine to handle periodicity
    diff_sin = torch.sin(pred_rad) - torch.sin(target_rad)
    diff_cos = torch.cos(pred_rad) - torch.cos(target_rad)
    # Calculate the squared differences
    squared_diff = diff_sin**2 + diff_cos**2
    # Calculate separate losses for phi and psi angles
    phi_loss = squared_diff[:, 0].mean()
    psi_loss = squared_diff[:, 1].mean()
    return phi_loss, psi_loss


def periodic_rmse(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the periodic root mean square error (RMSE) for angles.
    Args:
        pred: Predicted angles in degrees [batch_size, 2]
        target: True angles in degrees [batch_size, 2]
    Returns:
        Tuple of RMSE for phi and psi angles
    """
    phi_mse, psi_mse = periodic_mse(pred, target)
    phi_rmse = torch.sqrt(phi_mse)
    psi_rmse = torch.sqrt(psi_mse)
    return phi_rmse, psi_rmse


def train_model(
    model: torch.nn.Module, # graph neural network for analyzing glycans
    dataloaders: dict[str, torch.utils.data.DataLoader], # dict with 'train' and 'val' loaders
    optimizer: torch.optim.Optimizer, # PyTorch optimizer, has to be SAM if mode != "regression"
    scheduler: torch.optim.lr_scheduler._LRScheduler | None, # PyTorch learning rate decay
    num_epochs: int = 25, # number of epochs for training
):
    blank_metrics = {k: [] for k in {"loss", "phi_loss", "psi_loss", "sasa_loss", "flex_loss"}}
    metrics = {"train": copy.deepcopy(blank_metrics), "val": copy.deepcopy(blank_metrics)}
    best_loss = float("inf")
    best_model = None
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print(model.body[0].nn[0].weight[0, 0].isnan() == True)
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_metrics = copy.deepcopy(blank_metrics)
            running_metrics["weights"] = []
            for data in dataloaders[phase]:
                # Get all relevant node attributes
                x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # First forward pass
                    angular_pred, sasa_pred, flex_pred = model(x.to("cuda"), edge_index.to("cuda"))
                    y = y.to("cuda")
                    mono_mask = y[:, 2] != 0  # Do based on SASA
                    if isinstance(model, GINSweetNet):
                        phi_pred, psi_pred = angular_pred
                        phi_loss, psi_loss = periodic_mse(torch.stack([phi_pred[~mono_mask], psi_pred[~mono_mask]], dim = 1), y[~mono_mask, :2])
                    else:
                        weights_logits_von_mises, mus_von_mises, kappas_von_mises = angular_pred
                        phi_loss, psi_loss = mixture_von_mises_nll(y[~mono_mask, :2], weights_logits_von_mises[~mono_mask], mus_von_mises[~mono_mask], kappas_von_mises[~mono_mask])
                    sasa_loss = torch.sqrt(torch.nn.functional.mse_loss(sasa_pred[mono_mask], y[mono_mask, 2]))
                    flex_loss = torch.sqrt(torch.nn.functional.mse_loss(flex_pred[mono_mask], y[mono_mask, 3]))
                    loss = phi_loss + psi_loss + sasa_loss / 60 + flex_loss
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # Collecting relevant metrics
                running_metrics["loss"].append(loss.item())
                running_metrics["phi_loss"].append(phi_loss.item())
                running_metrics["psi_loss"].append(psi_loss.item())
                running_metrics["sasa_loss"].append(sasa_loss.item())
                running_metrics["flex_loss"].append(flex_loss.item())
                running_metrics["weights"].append(batch.max().cpu() + 1)
            # Averaging metrics at end of epoch
            for key in running_metrics:
                if key == "weights":
                    continue
                metrics[phase][key].append(np.average(running_metrics[key], weights = running_metrics["weights"]))
            print('{} Loss: {:.4f} Phi: {:.4f} Psi: {:.4f} SASA: {:.4f} Flex: {:.4f} LR: {:.4f}'.format(
                phase, 
                metrics[phase]["loss"][-1], 
                metrics[phase]["phi_loss"][-1], 
                metrics[phase]["psi_loss"][-1], 
                metrics[phase]["sasa_loss"][-1], 
                metrics[phase]["flex_loss"][-1],
                float(scheduler.get_last_lr()[0]) if scheduler else 0.001,
            ))
            # Keep best model state_dict
            if phase == "val":
                if metrics[phase]["loss"][-1] <= best_loss:
                    best_loss = metrics[phase]["loss"][-1]
                    best_model = copy.deepcopy(model.state_dict())
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(metrics[phase]["loss"][-1])
                    else:
                        scheduler.step()
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model)
    return metrics, model
