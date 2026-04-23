import pytest
import pandas as pd
import numpy as np
import networkx as nx
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
os.environ['PYTEST_RUNNING'] = '1'

# Import the functions to test
from glycontact.process import *

TEST_GLYCAN = "Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc"
TEST_PATH = this_dir = Path(__file__).parent / TEST_GLYCAN
TEST_EXAMPLE = TEST_PATH / "cluster0_alpha.pdb"


def test_make_atom_contact_table():
    result = get_contact_tables(TEST_GLYCAN, level='atom', my_path=TEST_PATH)[0]
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # Should be square matrix


def test_make_monosaccharide_contact_table():
    result = get_contact_tables(TEST_GLYCAN, level='monosaccharide', my_path=TEST_PATH)[0]
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # Should be square matrix


def test_focus_table_on_residue():
    # Create a test table with multiple residue types
    test_table = pd.DataFrame(
        [[0, 1, 2], [3, 0, 4], [5, 6, 0]],
        index=['1_MAN', '2_GLC', '3_GAL'],
        columns=['1_MAN', '2_GLC', '3_GAL']
    )
    result = focus_table_on_residue(test_table, 'MAN')
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    assert list(result.index) == ['1_MAN']


def test_get_glycoshape_IUPAC():
  result = get_glycoshape_IUPAC(fresh=False)
  assert isinstance(result, set)
  assert len(result) > 0  # Should have at least some glycans
  # Check if some common glycans are in the result
  common_glycans = ["Man(a1-3)Man", "GlcNAc(b1-4)GlcNAc"]
  assert any(glycan in result for glycan in common_glycans)


def test_calculate_torsion_angle():
    # Create a set of 4 coordinates that form a known torsion angle
    coords = [
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1]
    ]
    result = calculate_torsion_angle(coords)
    assert isinstance(result, float)
    assert result == pytest.approx(-90.0, abs=1e-5)


def test_convert_glycan_to_class():
    test_glycan = "Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)]Man(a1-6)Man"
    result = convert_glycan_to_class(test_glycan)
    assert isinstance(result, str)
    assert "X" in result  # Should contain X for hexoses
    assert "XNAc" in result  # Should contain XNAc for GlcNAc


def test_group_by_silhouette():
    test_glycans = [
        "Man(a1-3)[Gal(b1-4)]Man",
        "Gal(b1-3)[Fuc(a1-4)]GlcNAc",
        "Man(a1-2)Man"
    ]
    result = group_by_silhouette(test_glycans, mode='X')
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    # The first two glycans should have the same silhouette
    assert result.iloc[0]['silhouette'] == result.iloc[1]['silhouette']
    # The third glycan should have a different silhouette
    assert result.iloc[0]['silhouette'] != result.iloc[2]['silhouette']


@pytest.fixture(scope="module")
def real_data():
  # Single check for glycan database
  if not TEST_PATH.exists():
    pytest.skip(f"Test glycan {TEST_GLYCAN} not available in database")
  example_pdb = get_example_pdb(TEST_GLYCAN, my_path=TEST_PATH)
  df, interaction_dict = get_annotation(TEST_GLYCAN, example_pdb, threshold=3.5)
  contacts = get_contact_tables(TEST_GLYCAN, my_path=TEST_PATH)
  return {
    'pdb': example_pdb,
    'df': df,
    'interaction_dict': interaction_dict,
    'contact_tables': contacts
  }


def test_monosaccharide_preference_structure(real_data):
  # Create contact table and then run the function
  contact_table = make_monosaccharide_contact_table(real_data['df'], threshold=20, mode='distance')
  result = monosaccharide_preference_structure(contact_table, 'GlcNAc', threshold=2)
  assert isinstance(result, dict)


def test_map_data_to_graph(real_data):
  # Run all the preprocessing functions
  df = real_data['df']
  interaction_dict = real_data['interaction_dict']
  # Run the ring conformation function
  ring_conf = get_ring_conformations(df)
  # Run the torsion angles function
  torsion_angles = get_glycosidic_torsions(df, interaction_dict)
  # Create a computed DataFrame
  residue_ids = df['residue_number'].unique()
  computed_df = pd.DataFrame({
    'Monosaccharide_id': residue_ids,
    'Monosaccharide': [df[df['residue_number'] == r]['monosaccharide'].iloc[0] for r in residue_ids],
    'SASA': [100.0] * len(residue_ids),
    'flexibility': [0.5] * len(residue_ids)
  })
  # Run the function that's being tested
  result = map_data_to_graph(computed_df, interaction_dict, ring_conf, torsion_angles)
  assert isinstance(result, nx.Graph)
  assert len(result.nodes) > 0
  assert len(result.edges) > 0


def test_inter_structure_variability_table(real_data):
    result = inter_structure_variability_table(real_data['contact_tables'], mode='standard')
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # Should be square
    # Values should be non-negative
    assert (result.values >= 0).all()


def test_make_correlation_matrix(real_data):
    result = make_correlation_matrix(real_data['contact_tables'])
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # Should be square
    # Diagonal elements should be close to 1 (self-correlation)
    assert np.allclose(np.diag(result), 1.0, atol=0.1)


def test_inter_structure_frequency_table(real_data):
    result = inter_structure_frequency_table(real_data['contact_tables'], threshold=10)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # Should be square
    # Values should be integers (counts)
    assert np.all(result.values.astype(int) == result.values)


def test_get_sasa_table():
    result = get_sasa_table(TEST_GLYCAN, my_path=TEST_PATH)
    assert isinstance(result, pd.DataFrame)
    assert 'SASA' in result.columns
    assert 'Monosaccharide' in result.columns


def test_get_annotation():
    df, interactions = get_annotation(TEST_GLYCAN, get_example_pdb(TEST_GLYCAN, my_path=TEST_PATH), threshold=3.5)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(interactions, dict)


def test_annotation_pipeline():
    dfs, int_dicts = annotation_pipeline(TEST_GLYCAN, my_path=TEST_PATH)
    assert isinstance(dfs, tuple)
    assert isinstance(int_dicts, tuple)


def test_get_all_clusters_frequency():
    result = get_all_clusters_frequency(fresh=False)
    assert isinstance(result, dict)


def test_glycan_cluster_pattern():
    major, minor = glycan_cluster_pattern(threshold=70, mute=True)
    assert isinstance(major, list)
    assert isinstance(minor, list)


def test_get_structure_graph():
    result = get_structure_graph(TEST_GLYCAN, example_path = TEST_EXAMPLE, sasa_flex_path=TEST_EXAMPLE)
    assert isinstance(result, nx.Graph)
    assert len(result.nodes) > 0
    assert len(result.edges) > 0


def test_get_ring_conformations(real_data):
    result = get_ring_conformations(real_data['df'])
    assert isinstance(result, pd.DataFrame)
    assert 'residue' in result.columns
    assert 'monosaccharide' in result.columns
    assert 'conformation' in result.columns


def test_get_glycosidic_torsions(real_data):
    result = get_glycosidic_torsions(real_data['df'], real_data['interaction_dict'])
    assert isinstance(result, pd.DataFrame)
    assert 'linkage' in result.columns
    assert 'phi' in result.columns
    assert 'psi' in result.columns
    assert 'anomeric_form' in result.columns
    assert 'position' in result.columns


def test_get_similar_glycans():
    result = get_similar_glycans(TEST_GLYCAN, rmsd_cutoff=3.0, glycan_database=unilectin_data,
                                 pdb_path=TEST_EXAMPLE)
    assert isinstance(result, list)

def test_gsid_conversion():
  result = gsid_conversion("Man(a1-3)Man")
  assert isinstance(result, str)

def test_convert_ID_not_found():
  result = convert_ID("not-a-real-id")
  assert result == "Not Found"

def test_make_monosaccharide_contact_table_both_mode(real_data):
  result = make_monosaccharide_contact_table(real_data['df'], mode='both')
  assert isinstance(result, list)
  assert len(result) == 2
  assert isinstance(result[0], pd.DataFrame)
  assert isinstance(result[1], pd.DataFrame)

def test_calculate_hsic():
  rng = np.random.default_rng(42)
  X = rng.standard_normal(30)
  Y = X + 0.1 * rng.standard_normal(30)
  hsic, p_value = calculate_hsic(X, Y)
  assert isinstance(float(hsic), float)
  assert 0.0 <= float(p_value) <= 1.0

def test_df_to_pdb_content(real_data):
  result = df_to_pdb_content(real_data['df'])
  assert isinstance(result, str)
  assert 'ATOM' in result
  assert 'END' in result

def test_extract_functional_groups(real_data):
  result = extract_functional_groups(real_data['df'])
  assert isinstance(result, dict)
  assert 'oh_groups' in result
  assert 'ch_groups' in result

def test_calculate_ring_normals(real_data):
  functional_groups = extract_functional_groups(real_data['df'])
  result = calculate_ring_normals(real_data['df'], functional_groups)
  assert isinstance(result, dict)
  assert 'oh_groups' in result

def test_get_functional_group_analysis():
  result = get_functional_group_analysis(TEST_GLYCAN, my_path=TEST_PATH)
  assert isinstance(result, dict)
  assert 'functional_groups' in result or 'error' in result

def test_group_by_silhouette_class_mode():
  test_glycans = ["Man(a1-3)Man", "Gal(b1-3)GlcNAc", "Fuc(a1-2)Gal"]
  result = group_by_silhouette(test_glycans, mode='class')
  assert isinstance(result, pd.DataFrame)
  assert len(result) == 3

def test_align_point_sets_fast():
  rng = np.random.default_rng(0)
  coords1 = rng.standard_normal((10, 3))
  coords2 = coords1 + 0.1
  transformed, rmsd = align_point_sets(coords1, coords2, fast=True)
  assert transformed.shape == coords1.shape
  assert rmsd >= 0.0

def test_inter_structure_variability_table_amplify(real_data):
  result = inter_structure_variability_table(real_data['contact_tables'], mode='amplify')
  assert isinstance(result, pd.DataFrame)
  assert result.shape[0] == result.shape[1]

def test_inter_structure_torsion_variability():
  result = inter_structure_torsion_variability(TEST_GLYCAN, my_path=TEST_PATH)
  assert isinstance(result, pd.DataFrame)

def test_calculate_torsion_flexibility_per_residue():
  result = calculate_torsion_flexibility_per_residue(TEST_GLYCAN, my_path=TEST_PATH)
  assert isinstance(result, dict)

def test_get_ring_conformations_empty_df():
  result = get_ring_conformations(pd.DataFrame())
  assert isinstance(result, pd.DataFrame)
  assert list(result.columns) == ['residue', 'monosaccharide', 'Q', 'theta', 'phi', 'conformation']

def test_correct_dataframe(real_data):
  df_copy = real_data['df'].copy()
  result = correct_dataframe(df_copy)
  assert isinstance(result, pd.DataFrame)
  assert 'monosaccharide' in result.columns

# ── lwca.py ──────────────────────────────────────────────────────────────────

pytest_torch = pytest.importorskip("torch", reason="torch not installed")

import torch
from glycontact.lwca import LinearWarmupCosineAnnealingLR, linear_warmup_decay


def _make_optimizer(lr=0.1):
  model = torch.nn.Linear(2, 2)
  return torch.optim.SGD(model.parameters(), lr=lr)


def test_linear_warmup_decay_warmup_phase():
  fn = linear_warmup_decay(warmup_steps=10, total_steps=100, cosine=True)
  assert fn(0) == 0.0
  assert fn(5) == pytest.approx(0.5, abs=1e-6)
  assert fn(10) == pytest.approx(1.0, abs=1e-6)


def test_linear_warmup_decay_cosine_phase():
  fn = linear_warmup_decay(warmup_steps=0, total_steps=100, cosine=True)
  assert fn(0) == pytest.approx(1.0, abs=1e-6)
  assert fn(100) == pytest.approx(0.0, abs=1e-5)


def test_linear_warmup_decay_linear_phase():
  fn = linear_warmup_decay(warmup_steps=0, total_steps=100, cosine=False, linear=True)
  assert fn(50) == pytest.approx(0.5, abs=1e-6)
  assert fn(100) == pytest.approx(0.0, abs=1e-6)


def test_linear_warmup_decay_no_decay():
  fn = linear_warmup_decay(warmup_steps=0, total_steps=100, cosine=False, linear=False)
  assert fn(50) == pytest.approx(1.0, abs=1e-6)


def test_linear_warmup_decay_assertion():
  with pytest.raises(AssertionError):
    linear_warmup_decay(warmup_steps=5, total_steps=100, cosine=True, linear=True)


def test_lwca_scheduler_init():
  opt = _make_optimizer()
  sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=5, max_epochs=20)
  assert sched.warmup_epochs == 5
  assert sched.max_epochs == 20


def test_lwca_scheduler_warmup_increases_lr():
  opt = _make_optimizer(lr=0.1)
  sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=5, max_epochs=20, warmup_start_lr=0.0)
  lrs = []
  for _ in range(5):
    opt.step()
    sched.step()
    lrs.append(opt.param_groups[0]['lr'])
  assert lrs[-1] >= lrs[0]


def test_lwca_scheduler_state_dict_round_trip():
  opt = _make_optimizer()
  sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=3, max_epochs=10)
  sd = sched.state_dict()
  assert 'last_epoch' in sd
  assert 'warmup_epochs' in sd
  sched.load_state_dict(sd)
  assert sched.warmup_epochs == 3


def test_lwca_get_last_lr():
  opt = _make_optimizer()
  sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=2, max_epochs=10)
  lrs = sched.get_last_lr()
  assert isinstance(lrs, list)
  assert len(lrs) == len(opt.param_groups)


# ── learning.py ──────────────────────────────────────────────────────────────

pytest_torch = pytest.importorskip("torch", reason="torch not installed")

from glycontact.learning import node2y, periodic_mse, periodic_rmse, build_baselines, sample_angle


def test_node2y_all_zero_returns_none():
  assert node2y({}) is None
  assert node2y({"phi_angle": 0, "psi_angle": 0, "SASA": 0, "flexibility": 0}) is None


def test_node2y_partial_values():
  result = node2y({"phi_angle": 45.0, "SASA": 0, "psi_angle": 0, "flexibility": 0})
  assert result == [45.0, 0, 0, 0, 0]


def test_node2y_all_values():
  result = node2y({"phi_angle": 10.0, "psi_angle": -20.0, "SASA": 50.0, "flexibility": 0.3})
  assert result == [10.0, -20.0, 50.0, 0.3, 0]


def test_periodic_mse_identical():
  pred = torch.zeros(10, 2)
  phi_loss, psi_loss = periodic_mse(pred, pred)
  assert phi_loss.item() == pytest.approx(0.0, abs=1e-6)
  assert psi_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_periodic_mse_180_offset():
  pred = torch.full((5, 2), 180.0)
  target = torch.full((5, 2), 0.0)
  phi_loss, psi_loss = periodic_mse(pred, target)
  assert phi_loss.item() > 0
  assert psi_loss.item() > 0


def test_periodic_rmse_is_sqrt_of_mse():
  rng = torch.manual_seed(0)
  pred = torch.randn(8, 2) * 90
  target = torch.randn(8, 2) * 90
  phi_mse, psi_mse = periodic_mse(pred, target)
  phi_rmse, psi_rmse = periodic_rmse(pred, target)
  assert phi_rmse.item() == pytest.approx(phi_mse.sqrt().item(), abs=1e-5)
  assert psi_rmse.item() == pytest.approx(psi_mse.sqrt().item(), abs=1e-5)


def _make_nx_graph_with_attrs():
  G = nx.DiGraph()
  G.add_node(0, string_labels="GlcNAc", SASA=80.0, flexibility=0.4)
  G.add_node(1, string_labels="a1-4", phi_angle=45.0, psi_angle=-60.0)
  G.add_node(2, string_labels="Man", SASA=60.0, flexibility=0.2)
  G.add_edge(0, 1)
  G.add_edge(1, 2)
  return G


def test_build_baselines_returns_callables():
  graphs = [_make_nx_graph_with_attrs()]
  phi_fn, psi_fn, sasa_fn, flex_fn = build_baselines(graphs)
  assert callable(phi_fn)
  assert callable(psi_fn)
  assert callable(sasa_fn)
  assert callable(flex_fn)


def test_build_baselines_sasa_lookup():
  graphs = [_make_nx_graph_with_attrs(), _make_nx_graph_with_attrs()]
  _, _, sasa_fn, flex_fn = build_baselines(graphs)
  assert isinstance(sasa_fn("GlcNAc"), float)
  assert isinstance(flex_fn("Man"), float)


def test_build_baselines_default_fallback():
  graphs = [_make_nx_graph_with_attrs()]
  phi_fn, psi_fn, _, _ = build_baselines(graphs)
  result = phi_fn(("GlcNAc", "Man"))
  assert isinstance(result, (int, float, np.floating))


def test_sample_angle_returns_scalar():
  weights = np.array([0.5, 0.5])
  mus = torch.tensor([30.0, -30.0])
  kappas = torch.tensor([2.0, 2.0])
  result = sample_angle(weights, mus, kappas)
  assert isinstance(result.item(), float)
  assert -180.0 <= result.item() <= 180.0


# ── visualize.py ─────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from glycontact.visualize import (
  draw_contact_map, make_gif, show_correlations, show_correlation_dendrogram,
  plot_monosaccharide_instability, plot_glycan_score, show_monosaccharide_preference_structure,
  add_snfg_symbol, _do_3d_plotting, plot_glycan_3D, plot_superimposed_glycans,
  calculate_average_metric, find_difference, extract_torsion_angles, ramachandran_plot
)

_MINIMAL_PDB = "\n".join([
  "HETATM    1  C1  MAN X   1       1.000   2.000   3.000  1.00  0.00           C",
  "HETATM    2  C2  MAN X   1       2.000   3.000   4.000  1.00  0.00           C",
  "HETATM    3  C3  MAN X   1       3.000   4.000   5.000  1.00  0.00           C",
  "HETATM    4  C4  MAN X   1       4.000   5.000   6.000  1.00  0.00           C",
  "HETATM    5  C5  MAN X   1       5.000   6.000   7.000  1.00  0.00           C",
  "HETATM    6  O5  MAN X   1       6.000   7.000   8.000  1.00  0.00           O",
  "END",
])
_MAN_COORDS = np.array([[1.,2.,3.],[2.,3.,4.],[3.,4.,5.],[4.,5.,6.],[5.,6.,7.],[6.,7.,8.]])
_MAN_LABELS = ["1_MAN_C1","1_MAN_C2","1_MAN_C3","1_MAN_C4","1_MAN_C5","1_MAN_O5"]


@pytest.fixture
def pdb_tmp(tmp_path):
  p = tmp_path / "test.pdb"
  p.write_text(_MINIMAL_PDB)
  return p


def _twin_graphs():
  def _g(*specs):
    G = nx.DiGraph()
    for i, (lbl, sasa) in enumerate(specs):
      G.add_node(i, string_labels=lbl, Monosaccharide=lbl, SASA=sasa)
    return G
  return {
    "Fuc(a1-6)GlcNAc": _g(("Fuc", 40.), ("GlcNAc", 80.)),
    "GlcNAc":           _g(("GlcNAc", 80.), ("GlcNAc", 60.)),
    "Fuc(a1-6)Man":     _g(("Fuc", 35.), ("Man", 75.)),
    "Man":              _g(("Man", 70.), ("Man", 55.)),
  }


def test_draw_contact_map_return_plot():
  df = pd.DataFrame([[0, 1], [1, 0]], index=['1_MAN', '2_GLC'], columns=['1_MAN', '2_GLC'])
  ax = draw_contact_map(df, return_plot=True)
  assert ax is not None
  plt.close('all')


def test_draw_contact_map_no_return():
  df = pd.DataFrame([[0, 2], [2, 0]], index=['1_MAN', '2_GLC'], columns=['1_MAN', '2_GLC'])
  with patch('glycontact.visualize.plt.show'):
    result = draw_contact_map(df, return_plot=False)
  assert result is None
  plt.close('all')


def test_draw_contact_map_with_filepath(tmp_path):
  df = pd.DataFrame([[0,1],[1,0]], index=['1_MAN','2_GLC'], columns=['1_MAN','2_GLC'])
  with patch('glycontact.visualize.plt.savefig') as mock_save:
    with patch('glycontact.visualize.plt.show'):
      draw_contact_map(df, filepath=str(tmp_path/'map.png'), return_plot=False)
  mock_save.assert_called_once()
  plt.close('all')


def _make_structure_graph_with_sasa():
  G = nx.DiGraph()
  G.add_node(0, string_labels="Fuc", SASA=40.0, flexibility=0.1)
  G.add_node(1, string_labels="a1-2", phi_angle=30.0, psi_angle=-50.0)
  G.add_node(2, string_labels="Gal", SASA=70.0, flexibility=0.3)
  G.add_edge(0, 1)
  G.add_edge(1, 2)
  return G


def test_calculate_average_metric_excludes_pattern():
  G = _make_structure_graph_with_sasa()
  result = calculate_average_metric(G, "Fuc", "SASA")
  assert isinstance(result, float)
  assert result == pytest.approx(70.0, abs=1e-5)


def test_calculate_average_metric_missing_metric():
  G = _make_structure_graph_with_sasa()
  result = calculate_average_metric(G, "Fuc", "nonexistent_metric")
  assert result == 0.0


def test_extract_torsion_angles_returns_lists():
  phi, psi = extract_torsion_angles("Fuc(a1-2)Gal")
  assert isinstance(phi, list)
  assert isinstance(psi, list)
  assert len(phi) == len(psi)


def test_extract_torsion_angles_known_linkage_nonempty():
  phi, psi = extract_torsion_angles("GlcNAc(b1-4)GlcNAc")
  assert len(phi) > 0


def test_ramachandran_plot_returns_figure():
  phi, psi = extract_torsion_angles("GlcNAc(b1-4)GlcNAc")
  if len(phi) < 4:
    pytest.skip("Not enough angle data for ramachandran plot")
  fig = ramachandran_plot("GlcNAc(b1-4)GlcNAc", density=False)
  assert isinstance(fig, plt.Figure)
  plt.close('all')


def test_ramachandran_plot_raises_for_unknown_linkage():
  with pytest.raises(ValueError):
    ramachandran_plot("Zzz(x1-9)Yyy")


def test_ramachandran_plot_with_density():
  phi, psi = extract_torsion_angles("GlcNAc(b1-4)GlcNAc")
  if len(phi) < 4:
    pytest.skip("Insufficient angle data for density plot")
  fig = ramachandran_plot("GlcNAc(b1-4)GlcNAc", density=True)
  assert isinstance(fig, plt.Figure)
  plt.close('all')


def test_ramachandran_plot_with_filepath(tmp_path):
  phi, psi = extract_torsion_angles("GlcNAc(b1-4)GlcNAc")
  if len(phi) < 4:
    pytest.skip("Insufficient angle data")
  with patch('glycontact.visualize.plt.savefig') as mock_save:
    ramachandran_plot("GlcNAc(b1-4)GlcNAc", density=False, filepath=str(tmp_path/'rama.png'))
  mock_save.assert_called_once()
  plt.close('all')


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_make_gif():
  df = pd.DataFrame([[0,1],[1,0]], index=['1_MAN','2_GLC'], columns=['1_MAN','2_GLC'])
  with patch('glycontact.visualize.imageio.mimsave') as mock_save:
    with patch('glycontact.visualize.display'):
      with patch('glycontact.visualize.Image'):
        make_gif('pfx', [df])
  mock_save.assert_called_once()
  plt.close('all')


def test_show_correlations():
  df = pd.DataFrame([[1.,.5],[.5,1.]], index=['MAN','GLC'], columns=['MAN','GLC'])
  with patch('glycontact.visualize.plt.show'):
    show_correlations(df)
  plt.close('all')


@pytest.mark.filterwarnings("ignore::scipy.cluster.hierarchy.ClusterWarning")
def test_show_correlation_dendrogram():
  df = pd.DataFrame(
    [[0.,.5,.9],[.5,0.,.8],[.9,.8,0.]],
    index=['MAN','GLC','GAL'], columns=['MAN','GLC','GAL']
  )
  with patch('glycontact.visualize.plt.show'):
    result = show_correlation_dendrogram(df)
  assert isinstance(result, dict)
  plt.close('all')


def test_plot_monosaccharide_instability_sum():
  mock_df = pd.DataFrame({'1_MAN':[1.,2.],'2_GLC':[.5,1.5]}, index=['c0','c1'])
  with patch('glycontact.visualize.inter_structure_variability_table', return_value=mock_df):
    with patch('glycontact.visualize.plt.show'):
      plot_monosaccharide_instability('Man(a1-2)Man', mode='sum')
  plt.close('all')


def test_plot_monosaccharide_instability_mean_filepath(tmp_path):
  mock_df = pd.DataFrame({'1_MAN':[1.,2.],'2_GLC':[.5,1.5]}, index=['c0','c1'])
  with patch('glycontact.visualize.inter_structure_variability_table', return_value=mock_df):
    with patch('glycontact.visualize.plt.savefig') as mock_save:
      with patch('glycontact.visualize.plt.show'):
        plot_monosaccharide_instability('Man(a1-2)Man', mode='mean', filepath=str(tmp_path)+'/')
  mock_save.assert_called_once()
  plt.close('all')


def test_plot_glycan_score_with_score_list():
  mock_draw = MagicMock()
  with patch('glycontact.visualize.canonicalize_iupac', return_value='Man(a1-2)Man'):
    with patch('glycontact.visualize.GlycoDraw', return_value=mock_draw):
      result = plot_glycan_score('Man(a1-2)Man', score_list=[1.,2.,3.])
  assert result == mock_draw


def test_plot_glycan_score_zero_range():
  mock_draw = MagicMock()
  with patch('glycontact.visualize.canonicalize_iupac', return_value='Man(a1-2)Man'):
    with patch('glycontact.visualize.GlycoDraw', return_value=mock_draw):
      result = plot_glycan_score('Man(a1-2)Man', score_list=[5.,5.,5.])
  assert result == mock_draw


def test_plot_glycan_score_from_graph():
  mock_draw = MagicMock()
  with patch('glycontact.visualize.canonicalize_iupac', return_value='Man(a1-2)Man'):
    with patch('glycontact.visualize.get_structure_graph', return_value=MagicMock()):
      with patch('glycontact.visualize.nx.get_node_attributes', return_value={'0':1.,'1':2.}):
        with patch('glycontact.visualize.GlycoDraw', return_value=mock_draw):
          result = plot_glycan_score('Man(a1-2)Man')
  assert result == mock_draw


def test_plot_glycan_score_with_filepath():
  mock_draw = MagicMock()
  with patch('glycontact.visualize.canonicalize_iupac', return_value='Man(a1-2)Man'):
    with patch('glycontact.visualize.GlycoDraw', return_value=mock_draw):
      result = plot_glycan_score('Man(a1-2)Man', score_list=[1.,2.,3.], filepath='/tmp/')
  assert result == mock_draw


def test_show_monosaccharide_preference_structure():
  with patch('glycontact.visualize.monosaccharide_preference_structure',
             return_value={'r1':'A','r2':'B','r3':'A'}):
    with patch('glycontact.visualize.plt.show'):
      show_monosaccharide_preference_structure(MagicMock(), 'GlcNAc', 3.5)
  plt.close('all')


def test_add_snfg_sphere_gal():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Gal')
  v.addSphere.assert_called_once()


def test_add_snfg_sphere_glc():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Glc')
  v.addSphere.assert_called_once()


def test_add_snfg_sphere_man():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Man')
  v.addSphere.assert_called_once()


def test_add_snfg_cube_glcnac():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'GlcNAc')
  v.addBox.assert_called_once()


def test_add_snfg_cube_galnac():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'GalNAc')
  v.addBox.assert_called_once()


def test_add_snfg_diamond_neu5ac():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Neu5Ac')
  assert v.addCylinder.call_count == 12


def test_add_snfg_diamond_neu5gc():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Neu5Gc')
  assert v.addCylinder.call_count == 12


def test_add_snfg_cone_fuc():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Fuc')
  v.addArrow.assert_called_once()


def test_add_snfg_cone_rha():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Rha')
  v.addArrow.assert_called_once()


def test_add_snfg_unknown_no_call():
  v = MagicMock()
  add_snfg_symbol(v, np.zeros(3), 'Unknown')
  v.addSphere.assert_not_called()
  v.addBox.assert_not_called()
  v.addArrow.assert_not_called()


def test_do_3d_plotting_basic(pdb_tmp):
  v = MagicMock()
  with patch.dict('glycontact.visualize.map_dict', {'MAN': 'Manb1'}):
    _do_3d_plotting(str(pdb_tmp), _MAN_COORDS, _MAN_LABELS, view=v)
  v.addModel.assert_called_once()
  v.setStyle.assert_called()


def test_do_3d_plotting_snfg_and_labels(pdb_tmp):
  v = MagicMock()
  with patch.dict('glycontact.visualize.map_dict', {'MAN': 'Manb1'}):
    _do_3d_plotting(str(pdb_tmp), _MAN_COORDS, _MAN_LABELS, view=v, show_snfg=True, show_labels=True)
  v.addLabel.assert_called()


def test_do_3d_plotting_show_volume(pdb_tmp):
  v = MagicMock()
  with patch.dict('glycontact.visualize.map_dict', {'MAN': 'Manb1'}):
    _do_3d_plotting(str(pdb_tmp), _MAN_COORDS, _MAN_LABELS, view=v, show_volume=True)
  v.addSurface.assert_called_once()


def test_do_3d_plotting_color_and_mobile(pdb_tmp):
  v = MagicMock()
  with patch.dict('glycontact.visualize.map_dict', {'MAN': 'Manb1'}):
    _do_3d_plotting(str(pdb_tmp), _MAN_COORDS, _MAN_LABELS,
                    view=v, color='skyblueCarbon', bond_color='blue', pos='mobile')
  assert v.setStyle.call_count >= 2


def test_do_3d_plotting_creates_view_when_none(pdb_tmp):
  mock_view = MagicMock()
  with patch('glycontact.visualize.py3Dmol.view', return_value=mock_view):
    with patch.dict('glycontact.visualize.map_dict', {'MAN': 'Manb1'}):
      _do_3d_plotting(str(pdb_tmp), _MAN_COORDS, _MAN_LABELS, view=None)
  mock_view.addModel.assert_called_once()


def test_do_3d_plotting_df_path(pdb_tmp):
  v = MagicMock()
  with patch('glycontact.visualize.df_to_pdb_content', return_value=_MINIMAL_PDB):
    with patch.dict('glycontact.visualize.map_dict', {'MAN': 'Manb1'}):
      _do_3d_plotting((MagicMock(),), _MAN_COORDS, _MAN_LABELS, view=v)
  v.addModel.assert_called_once()


def test_plot_glycan_3D_basic():
  mock_view = MagicMock()
  mock_df = pd.DataFrame({
    'atom_name':['C1','C2'], 'x':[1.,2.], 'y':[1.,2.], 'z':[1.,2.],
    'residue_number':[1,1], 'monosaccharide':['MAN','MAN']
  })
  with patch('glycontact.visualize.py3Dmol.view', return_value=mock_view):
    with patch('glycontact.visualize.get_example_pdb', return_value='fake.pdb'):
      with patch('glycontact.visualize.extract_3D_coordinates', return_value=mock_df):
        with patch('glycontact.visualize._do_3d_plotting'):
          result = plot_glycan_3D('Man(a1-2)Man')
  assert result == mock_view


def test_plot_glycan_3D_with_filepath(tmp_path):
  mock_view = MagicMock()
  pdb = str(tmp_path / 'g.pdb')
  mock_df = pd.DataFrame({
    'atom_name':['C1'], 'x':[1.], 'y':[1.], 'z':[1.],
    'residue_number':[1], 'monosaccharide':['MAN']
  })
  with patch('glycontact.visualize.py3Dmol.view', return_value=mock_view):
    with patch('glycontact.visualize.extract_3D_coordinates', return_value=mock_df):
      with patch('glycontact.visualize._do_3d_plotting'):
        result = plot_glycan_3D('Man(a1-2)Man', filepath=pdb)
  assert result == mock_view


def test_plot_superimposed_glycans_no_animate(pdb_tmp):
  superpos = {
    'ref_conformer': str(pdb_tmp), 'ref_coords': _MAN_COORDS, 'ref_labels': _MAN_LABELS,
    'mobile_conformer': str(pdb_tmp), 'transformed_coords': _MAN_COORDS, 'mobile_labels': _MAN_LABELS,
    'rmsd': 1.23
  }
  mock_view = MagicMock()
  with patch('glycontact.visualize.py3Dmol.view', return_value=mock_view):
    with patch('glycontact.visualize._do_3d_plotting'):
      result = plot_superimposed_glycans(superpos, animate=False)
  mock_view.addLabel.assert_called_once()
  assert result == mock_view


def test_plot_superimposed_glycans_animate(pdb_tmp):
  superpos = {
    'ref_conformer': str(pdb_tmp), 'ref_coords': _MAN_COORDS, 'ref_labels': _MAN_LABELS,
    'mobile_conformer': str(pdb_tmp), 'transformed_coords': _MAN_COORDS, 'mobile_labels': _MAN_LABELS,
    'rmsd': 0.5
  }
  mock_view = MagicMock()
  with patch('glycontact.visualize.py3Dmol.view', return_value=mock_view):
    with patch('glycontact.visualize._do_3d_plotting'):
      plot_superimposed_glycans(superpos, animate=True)
  mock_view.spin.assert_called_once_with(True)


def test_plot_superimposed_glycans_with_filepath(pdb_tmp):
  superpos = {
    'ref_conformer': str(pdb_tmp), 'ref_coords': _MAN_COORDS, 'ref_labels': _MAN_LABELS,
    'mobile_conformer': str(pdb_tmp), 'transformed_coords': _MAN_COORDS, 'mobile_labels': _MAN_LABELS,
    'rmsd': 0.5
  }
  mock_view = MagicMock()
  with patch('glycontact.visualize.py3Dmol.view', return_value=mock_view):
    with patch('glycontact.visualize._do_3d_plotting'):
      with patch('glycontact.visualize.display'):
        with patch('glycontact.visualize.HTML'):
          result = plot_superimposed_glycans(superpos, filepath='out.png', animate=False)
  assert result == mock_view


def test_calculate_average_metric_predecessor_chain():
  G = nx.DiGraph()
  G.add_node(0, string_labels="Root", SASA=100.)
  G.add_node(1, string_labels="Mid", SASA=90.)
  G.add_node(2, string_labels="Fuc", SASA=40., Monosaccharide="Fuc")
  G.add_node(3, string_labels="Gal", SASA=70.)
  G.add_edge(0, 1)
  G.add_edge(1, 2)
  G.add_edge(2, 3)
  # "Fuc" node has predecessor Mid, which has predecessor Root → Root goes into predecessor_nodes
  result = calculate_average_metric(G, "Fuc", "SASA")
  assert isinstance(result, float)
  # Root (100) and Gal (70) could be included depending on exclude logic
  assert result > 0.


def test_calculate_average_metric_all_excluded_returns_zero():
  G = nx.DiGraph()
  G.add_node(0, string_labels="Fuc", SASA=40., Monosaccharide="Fuc")
  result = calculate_average_metric(G, "Fuc", "SASA")
  assert result == 0.0

def test_find_difference_raises_none_struc_dict():
  with pytest.raises(ValueError):
    find_difference([], pattern="X", struc_dict=None)


def test_find_difference_raises_no_pattern():
  with pytest.raises(ValueError):
    find_difference([], struc_dict={})


def test_find_difference_raises_no_pattern_with_alternative():
  with pytest.raises(ValueError):
    find_difference([], alternative="a2-6", struc_dict=None)


def test_find_difference_presence_absence():
  graphs = _twin_graphs()
  with patch('glycontact.visualize.compare_glycans', side_effect=lambda a, b: a == b):
    result = find_difference(list(graphs.keys()), pattern="Fuc(a1-6)", struc_dict=graphs)
  assert result['pattern'] == "Fuc(a1-6)"
  assert result['alternative'] is None
  assert result['n_pairs'] == 2


def test_find_difference_substitution():
  def _g(sasa_a, sasa_b):
    G = nx.DiGraph()
    G.add_node(0, string_labels="Neu5Ac", Monosaccharide="Neu5Ac", SASA=sasa_a)
    G.add_node(1, string_labels="Gal", Monosaccharide="Gal", SASA=sasa_b)
    return G
  graphs = {
    "Neu5Ac(a2-3)Gal":     _g(50., 70.),
    "Neu5Ac(a2-6)Gal":     _g(55., 65.),
    "Neu5Ac(a2-3)GalNAc":  _g(48., 68.),
    "Neu5Ac(a2-6)GalNAc":  _g(53., 63.),
  }
  with patch('glycontact.visualize.compare_glycans', side_effect=lambda a, b: a == b):
    result = find_difference(list(graphs.keys()), pattern="a2-3", alternative="a2-6",
                             struc_dict=graphs)
  assert result['alternative'] == "a2-6"
  assert result['n_pairs'] == 2


def test_find_difference_with_plot():
  graphs = _twin_graphs()
  with patch('glycontact.visualize.compare_glycans', side_effect=lambda a, b: a == b):
    with patch('glycontact.visualize.plt.show'):
      result = find_difference(list(graphs.keys()), pattern="Fuc(a1-6)",
                               struc_dict=graphs, plot=True)
  plt.close('all')
  assert 'plot' in result