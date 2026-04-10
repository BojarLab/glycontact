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


def test_get_ring_conformations(real_data):
  # Actually run the function with the real data
  result = get_ring_conformations(real_data['df'])
  assert isinstance(result, pd.DataFrame)
  assert 'residue' in result.columns
  assert 'conformation' in result.columns


def test_get_glycosidic_torsions(real_data):
  # Actually run the function with the real data
  result = get_glycosidic_torsions(real_data['df'], real_data['interaction_dict'])
  assert isinstance(result, pd.DataFrame)
  assert 'linkage' in result.columns
  assert 'phi' in result.columns
  assert 'psi' in result.columns


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