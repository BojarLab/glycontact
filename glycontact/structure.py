import re
import numpy as np
import pandas as pd
import mdtraj as md
from multiprocessing import Pool
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from glycowork.motif.processing import canonicalize_iupac
from glycontact.process import (get_annotation, map_dict, C2_PATTERN, extract_3D_coordinates,
                                extract_glycan_coords, get_example_pdb, get_glycoshape_IUPAC,
                                get_global_path, global_path, unilectin_data)


def get_binding_pocket(glycan, pdb_path, binding_monosaccharide = None, cutoff = 4.0, all_atoms = True, filepath = ''):
  """Extract amino acid residues within a cutoff distance from a specific monosaccharide in a glycan.
  Args:
    glycan (str): IUPAC glycan sequence
    pdb_path (str): Path to PDB file containing the glycan structure
    binding_monosaccharide (str): Monosaccharide identifier within the glycan (e.g., 'NAG', 'MAN', 'BMA'); if None, uses entire glycan
    cutoff (float): Distance cutoff in Angstroms (default 4.0)
    all_atoms (bool): If True, return all atoms within cutoff; if False, return only closest atom per residue
    filepath (str): filepath to save extracted binding pocket as PDB file, if desired; Optional
  Returns:
    pd.DataFrame: DataFrame with columns for residue info (chain, resSeq, resName, atom_name, distance_min)
  """
  glycan_df, interaction_dict = get_annotation(glycan, pdb_path, threshold = 3.5)
  if len(glycan_df) == 0:
    return pd.DataFrame()
  if binding_monosaccharide is None:
    target_residues = glycan_df
  else:
    target_residues = glycan_df[glycan_df['monosaccharide'] == binding_monosaccharide]
    if len(target_residues) == 0:
      mapped_name = None
      for pdb_code, iupac in map_dict.items():
        if binding_monosaccharide in iupac or iupac.startswith(binding_monosaccharide):
          potential_residues = glycan_df[glycan_df['monosaccharide'] == pdb_code]
          if len(potential_residues) > 0:
            target_residues = potential_residues
            break
    if len(target_residues) == 0:
      return pd.DataFrame()
  traj = md.load(pdb_path)
  topology = traj.topology
  target_atom_indices = []
  target_atoms = []
  for _, row in target_residues.iterrows():
    target_chain = row['chain_id']
    target_residue_number = row['residue_number']
    for res in topology.residues:
      if res.resSeq == target_residue_number and res.chain.chain_id == target_chain:
        for atom in res.atoms:
          target_atom_indices.append(atom.index)
          target_atoms.append(atom)
        break
  if len(target_atom_indices) == 0:
    return pd.DataFrame()
  target_coords = traj.xyz[0, target_atom_indices, :] * 10
  protein_residues = [res for res in topology.residues if res.is_protein]
  binding_pocket_data = []
  for residue in protein_residues:
    residue_atoms = [atom for atom in residue.atoms]
    residue_atom_indices = [atom.index for atom in residue_atoms]
    residue_coords = traj.xyz[0, residue_atom_indices, :] * 10
    distances = cdist(target_coords, residue_coords)
    if all_atoms:
      for atom_idx, atom in enumerate(residue_atoms):
        min_distance_to_atom = np.min(distances[:, atom_idx])
        if min_distance_to_atom <= cutoff:
          target_atom_idx = np.argmin(distances[:, atom_idx])
          target_atom = target_atoms[target_atom_idx]
          binding_pocket_data.append({
            'chain': residue.chain.chain_id,
            'resSeq': residue.resSeq,
            'resName': residue.name,
            'atom_name': atom.name,
            'target_atom': f"{target_atom.residue.name}{target_atom.residue.resSeq}_{target_atom.name}",
            'distance_min': min_distance_to_atom
          })
    else:
      min_distance = np.min(distances)
      if min_distance <= cutoff:
        min_dist_idx = np.unravel_index(np.argmin(distances), distances.shape)
        target_atom = target_atoms[min_dist_idx[0]]
        closest_residue_atom = residue_atoms[min_dist_idx[1]]
        binding_pocket_data.append({
          'chain': residue.chain.chain_id,
          'resSeq': residue.resSeq,
          'resName': residue.name,
          'atom_name': closest_residue_atom.name,
          'target_atom': f"{target_atom.residue.name}{target_atom.residue.resSeq}_{target_atom.name}",
          'distance_min': min_distance
        })
  result_df = pd.DataFrame(binding_pocket_data)
  if len(result_df) > 0:
    result_df = result_df.sort_values('distance_min').reset_index(drop = True)
  if filepath:
    save_binding_pocket_pdb(result_df, pdb_path, glycan, filepath)
  return result_df


def save_binding_pocket_pdb(result_df, pdb_path, glycan, output_path):
  """Create a new PDB file containing only the binding pocket residues and the specified glycan.
  Args:
    result_df (pd.DataFrame): DataFrame from get_binding_pocket containing binding pocket residues
    pdb_path (str): Path to original PDB file
    glycan (str): IUPAC glycan sequence
    output_path (str): Path where the new PDB file should be saved
  Returns:
    str: Path to the saved PDB file
  """
  glycan_df, interaction_dict = get_annotation(glycan, pdb_path, threshold = 3.5)
  if len(glycan_df) == 0:
    raise ValueError(f"Could not find glycan {glycan} in PDB file")
  traj = md.load(pdb_path)
  topology = traj.topology
  atom_indices_to_keep = []
  glycan_residues = set((row['chain_id'], row['residue_number']) for _, row in glycan_df.iterrows())
  for residue in topology.residues:
    if (residue.chain.chain_id, residue.resSeq) in glycan_residues:
      for atom in residue.atoms:
        atom_indices_to_keep.append(atom.index)
  pocket_residues = set()
  for _, row in result_df.iterrows():
    pocket_residues.add((row['chain'], row['resSeq'], row['resName']))
  for residue in topology.residues:
    if residue.is_protein:
      res_tuple = (residue.chain.chain_id, residue.resSeq, residue.name)
      if res_tuple in pocket_residues:
        for atom in residue.atoms:
          atom_indices_to_keep.append(atom.index)
  atom_indices_to_keep = sorted(set(atom_indices_to_keep))
  subset_traj = traj.atom_slice(atom_indices_to_keep)
  subset_traj.save_pdb(output_path)
  return output_path


def get_glycan_shielding(glycan, pdb_path, cutoff = 15.0, threshold = 1.0, same_chain_only = True):
  """Calculate the change in solvent accessible surface area (delta-SASA) of protein residues due to glycan attachment.
  Args:
    glycan (str): IUPAC glycan sequence
    pdb_path (str): Path to PDB file containing the glycan-protein complex
    cutoff (float): Distance cutoff in Angstroms to identify potentially affected residues (default 15.0)
    threshold (float): Minimum delta-SASA in A^2 to include in results (default 1.0)
    same_chain_only (bool): If True, only return residues from the same protein chain as glycan attachment (default True)
  Returns:
    pd.DataFrame: DataFrame with columns chain, resSeq, resName, SASA_protein, SASA_complex, delta_SASA, percent_shielded for residues showing appreciable shielding
  """
  glycan_df, interaction_dict = get_annotation(glycan, pdb_path, threshold = 3.5)
  if len(glycan_df) == 0:
    return pd.DataFrame()
  traj = md.load(pdb_path)
  topology = traj.topology
  specified_glycan_residues = set((row['chain_id'], row['residue_number']) for _, row in glycan_df.iterrows())
  specified_glycan_atom_indices, atoms_without_specified_glycan = [], []
  original_to_no_glycan_res_idx = {}
  no_glycan_res_counter = 0
  for orig_idx, res in enumerate(topology.residues):
    res_key = (res.chain.chain_id, res.resSeq)
    if res_key in specified_glycan_residues:
      for atom in res.atoms:
        specified_glycan_atom_indices.append(atom.index)
    else:
      original_to_no_glycan_res_idx[orig_idx] = no_glycan_res_counter
      no_glycan_res_counter += 1
      for atom in res.atoms:
        atoms_without_specified_glycan.append(atom.index)
  if len(specified_glycan_atom_indices) == 0 or len(atoms_without_specified_glycan) == 0:
    return pd.DataFrame()
  glycan_coords = traj.xyz[0, specified_glycan_atom_indices, :] * 10
  attachment_chain = None
  if same_chain_only:
    min_dist = float('inf')
    for res in topology.residues:
      if res.is_protein and (res.chain.chain_id, res.resSeq) not in specified_glycan_residues:
        residue_atom_indices = [atom.index for atom in res.atoms]
        residue_coords = traj.xyz[0, residue_atom_indices, :] * 10
        distances = cdist(glycan_coords, residue_coords)
        res_min_dist = np.min(distances)
        if res_min_dist < min_dist:
          min_dist = res_min_dist
          attachment_chain = res.chain.chain_id
  nearby_residue_orig_indices = []
  for orig_idx, res in enumerate(topology.residues):
    if res.is_protein and (res.chain.chain_id, res.resSeq) not in specified_glycan_residues:
      if same_chain_only and res.chain.chain_id != attachment_chain:
        continue
      residue_atom_indices = [atom.index for atom in res.atoms]
      residue_coords = traj.xyz[0, residue_atom_indices, :] * 10
      distances = cdist(glycan_coords, residue_coords)
      if np.min(distances) <= cutoff:
        nearby_residue_orig_indices.append(orig_idx)
  if len(nearby_residue_orig_indices) == 0:
    return pd.DataFrame()
  traj_without_glycan = traj.atom_slice(atoms_without_specified_glycan)
  sasa_without_glycan = md.shrake_rupley(traj_without_glycan, mode = 'residue') * 100
  sasa_complex = md.shrake_rupley(traj, mode = 'residue') * 100
  results = []
  for orig_idx in nearby_residue_orig_indices:
    res = list(topology.residues)[orig_idx]
    if orig_idx in original_to_no_glycan_res_idx:
      no_glycan_idx = original_to_no_glycan_res_idx[orig_idx]
      sasa_without_val = sasa_without_glycan[0, no_glycan_idx]
      sasa_with_val = sasa_complex[0, orig_idx]
      delta = sasa_without_val - sasa_with_val
      if abs(delta) >= threshold:
        percent_shielded = (delta / sasa_without_val * 100) if sasa_without_val > 0 else 0
        results.append({
          'chain': res.chain.chain_id,
          'resSeq': res.resSeq,
          'resName': res.name,
          'SASA_without_glycan': sasa_without_val,
          'SASA_with_glycan': sasa_with_val,
          'delta_SASA': delta,
          'percent_shielded': percent_shielded
        })
  result_df = pd.DataFrame(results)
  if len(result_df) > 0:
    result_df = result_df.sort_values('delta_SASA', ascending = False).reset_index(drop = True)
  return result_df


def get_pdb_atom_monosaccharides(mol):
  """Maps atom indices in a PDB-loaded RDKit mol to their IUPAC monosaccharide names"""
  result = {}
  for atom in mol.GetAtoms():
    info = atom.GetPDBResidueInfo()
    if info is None:
      continue
    res_name = info.GetResidueName().strip()
    if res_name in map_dict:
      result[atom.GetIdx()] = map_dict[res_name].split('(')[0].strip()
  return result

def get_glycan_sequences_from_pdb(pdb_file):
  """Extracts glycan sequences from a PDB file containing protein and glycan.
  Args:
    pdb_file (str): Path to the PDB file
  Returns:
    list: List of IUPAC glycan sequences found in the PDB
  """
  df = extract_3D_coordinates(pdb_file)
  if len(df) == 0:
    return []
  glycan_residues = df[df['monosaccharide'].isin(map_dict.keys())].copy()
  if len(glycan_residues) == 0:
    return []
  residue_info = {}
  for _, row in glycan_residues.iterrows():
    res_key = (row['chain_id'], row['residue_number'])
    if res_key not in residue_info:
      residue_info[res_key] = {'mono': row['monosaccharide'], 'atoms': {}}
    residue_info[res_key]['atoms'][row['atom_name']] = np.array([row['x'], row['y'], row['z']])
  connections = []
  for res1_key, res1_data in residue_info.items():
    mono_code = res1_data['mono']
    mono_name = map_dict.get(mono_code, '').split('(')[0]
    is_c2_linked = bool(re.search(C2_PATTERN, mono_code))
    link_carbon = 'C2' if is_c2_linked else 'C1'
    if link_carbon not in res1_data['atoms']:
      continue
    c_coord = res1_data['atoms'][link_carbon]
    is_l_sugar = mono_name.startswith('L') or 'Fuc' in mono_name or 'Rha' in mono_name
    for res2_key, res2_data in residue_info.items():
      if res1_key == res2_key:
        continue
      for atom_name, coord in res2_data['atoms'].items():
        if atom_name.startswith('O') and atom_name[1:].isdigit():
          dist = np.linalg.norm(c_coord - coord)
          if dist < 1.6:
            linkage_pos = atom_name[1:]
            if is_c2_linked:
              o_ref = res1_data['atoms'].get('O6')
              c_ref = res1_data['atoms'].get('C3')
            else:
              o_ref = res1_data['atoms'].get('O5')
              c_ref = res1_data['atoms'].get('C2')
            if o_ref is not None and c_ref is not None:
              v1 = o_ref - c_coord
              v2 = c_ref - c_coord
              v3 = coord - c_coord
              cross = np.cross(v1, v2)
              is_alpha = np.dot(cross, v3) < 0
              if is_l_sugar:
                is_alpha = not is_alpha
              anomeric = 'a' if is_alpha else 'b'
            else:
              anomeric = 'a'
            connections.append((res1_key, res2_key, linkage_pos, anomeric))
            break
  graph = {res: [] for res in residue_info.keys()}
  for donor, acceptor, link_pos, anomer in connections:
    graph[acceptor].append((donor, link_pos, anomer))
  sequences, visited = [], set()

  def build_sequence(res_key):
    if res_key in visited:
      return None
    visited.add(res_key)
    mono_code = residue_info[res_key]['mono']
    mono = map_dict.get(mono_code, '').split('(')[0]
    children = graph[res_key]
    if not children:
      return mono
    child_parts = []
    for child_key, link_pos, anomer in children:
      child_seq = build_sequence(child_key)
      if child_seq:
        anomeric_carbon = 2 if child_seq.endswith(("Neu5Ac", "Neu5Gc", "Kdn", "Fru", "Fruf")) else 1
        child_parts.append(f"{child_seq}({anomer}{anomeric_carbon}-{link_pos})")
    if len(child_parts) == 1:
      return f"{child_parts[0]}{mono}"
    elif len(child_parts) > 1:
      return f"{child_parts[0]}{''.join(f'[{cp}]' for cp in child_parts[1:])}{mono}"
    return mono

  reducing_ends = [res for res in residue_info.keys() if not any(conn[0] == res for conn in connections)]
  for root in reducing_ends:
    seq = build_sequence(root)
    if seq:
      sequences.append(seq)
  return sorted(list(set(sequences)), key = len, reverse = True)


def align_point_sets(mobile_coords, ref_coords, fast = False):
  """Find optimal rigid transformation to align two point sets using SVD-based Kabsch algorithm or Nelder-Mead optimization.
  Args:
    mobile_coords (np.ndarray): Nx3 array of coordinates to transform
    ref_coords (np.ndarray): Mx3 array of reference coordinates
    fast (bool): Whether to use SVD-based Kabsch algorithm with k-d trees or Nelder-Mead optimization. Defaults to the latter
  Returns:
    Tuple of (transformed coordinates, RMSD)
  """
  if fast:  # SVD-based Kabsch algorithm with k-d trees
    # Center the coordinates
    mobile_centroid = np.mean(mobile_coords, axis = 0)
    ref_centroid = np.mean(ref_coords, axis = 0)
    mobile_centered = mobile_coords - mobile_centroid
    ref_centered = ref_coords - ref_centroid
    # Find closest atoms (correspondence) between sets
    tree = cKDTree(ref_centered)
    _, indices = tree.query(mobile_centered)
    matched_ref = ref_centered[indices]
    # Compute covariance matrix
    covariance = mobile_centered.T @ matched_ref
    # Compute optimal rotation using SVD
    u, _, vt = np.linalg.svd(covariance)
    # Handle reflection case
    d = np.linalg.det(vt.T @ u.T)
    correction = np.eye(3)
    correction[2, 2] = d
    rotation = vt.T @ correction @ u.T
    # Apply rotation and translation
    transformed_coords = (mobile_coords - mobile_centroid) @ rotation + ref_centroid
    # Calculate final RMSD
    squared_diffs = np.sum((transformed_coords - ref_coords[indices]) ** 2, axis = 1)
    rmsd = np.sqrt(np.mean(squared_diffs))
  else:  # Nelder-Mead simplex optimization

    def get_rotation_matrix(angles):
      """Create 3D rotation matrix from angles."""
      cx, cy, cz = np.cos(angles)
      sx, sy, sz = np.sin(angles)
      Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
      Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
      Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
      return Rx @ Ry @ Rz

    def objective(params):
      """Objective function to minimize."""
      angles = params[:3]
      translation = params[3:]
      # Apply rotation and translation
      R = get_rotation_matrix(angles)
      transformed = (mobile_coords @ R) + translation
      # Calculate distances between all points
      distances = cdist(transformed, ref_coords)
      # Use sum of minimum distances as score
      return np.min(distances, axis = 1).sum()

    # Initial guess
    initial_guess = np.zeros(6)  # 3 rotation angles + 3 translation components
    # Optimize alignment
    result = minimize(objective, initial_guess, method = 'Nelder-Mead')
    # Get final transformation
    final_angles = result.x[:3]
    final_translation = result.x[3:]
    R = get_rotation_matrix(final_angles)
    transformed_coords = (mobile_coords @ R) + final_translation
    # Calculate final RMSD
    distances = cdist(transformed_coords, ref_coords)
    min_distances = np.min(distances, axis = 1)
    rmsd = np.sqrt(np.mean(min_distances ** 2))
  return transformed_coords, rmsd


def superimpose_glycans(ref_glycan, mobile_glycan, ref_residues = None, mobile_residues = None, main_chain_only = False,
                        fast = False):
  """Superimpose two glycan structures and calculate RMSD.
  Args:
    ref_glycan (str): Reference glycan or PDB path.
    mobile_glycan (str): Mobile glycan or PDB path to superimpose.
    ref_residues (list, optional): Residue numbers for reference glycan.
    mobile_residues (list, optional): Residue numbers for mobile glycan.
    main_chain_only (bool): If True, uses only main chain atoms.
    fast (bool): Whether to use SVD-based Kabsch algorithm with k-d trees or Nelder-Mead optimization. Defaults to the latter
  Returns:
    Dict containing:
        - ref_coords: Original coordinates of reference
        - transformed_coords: Aligned mobile coordinates
        - rmsd: Root mean square deviation
        - ref_labels: Atom labels from reference structure
        - mobile_labels: Atom labels from mobile structure
        - ref_conformer: PDB path of reference conformer
        - mobile_conformer: PDB path of mobile conformer
  """
  if isinstance(ref_glycan, str) and '.' not in ref_glycan:
    ref_conformers = list(((get_global_path() if global_path is None else global_path) / canonicalize_iupac(ref_glycan)).glob('*.pdb'))
  else:
    ref_conformers = [ref_glycan]
  if isinstance(mobile_glycan, str) and '.' not in mobile_glycan:
    mobile_conformers = list(((get_global_path() if global_path is None else global_path) / canonicalize_iupac(mobile_glycan)).glob('*.pdb'))
  else:
    mobile_conformers = [mobile_glycan]
  best_rmsd = float('inf')
  best_result = {'rmsd': best_rmsd}
  mobile_coord_cache = {mobile_pdb: extract_glycan_coords(mobile_pdb, mobile_residues, main_chain_only) for mobile_pdb in mobile_conformers}
  # Iterate over all possible pairs of conformers
  for ref_pdb in ref_conformers:  # Extract coordinates for reference conformer
    ref_coords, ref_labels = extract_glycan_coords(ref_pdb, ref_residues, main_chain_only)
    for mobile_pdb in mobile_conformers:  # Extract coordinates for mobile conformer
      mobile_coords, mobile_labels = mobile_coord_cache[mobile_pdb]
      transformed_coords, rmsd = align_point_sets(mobile_coords, ref_coords, fast = fast)
      if rmsd < best_rmsd:
        best_rmsd = rmsd
        best_result = {
            'ref_coords': ref_coords,
            'transformed_coords': transformed_coords,
            'rmsd': rmsd,
            'ref_labels': ref_labels,
            'mobile_labels': mobile_labels,
            'ref_conformer': ref_pdb,
            'mobile_conformer': mobile_pdb
            }
  return best_result


def _process_single_glycan(args):
  glycan, query_coords, rmsd_cutoff, fast = args
  best_rmsd = float('inf')
  best_structure = None
  pdb_files = list(((get_global_path() if global_path is None else global_path) / glycan).glob('*.pdb'))
  for pdb_file in pdb_files:
    try:
      coords, _ = extract_glycan_coords(pdb_file)
      if abs(len(coords) - len(query_coords)) <= 50:
        transformed, rmsd = align_point_sets(coords, query_coords, fast = fast)
        if rmsd < best_rmsd:
          best_rmsd = rmsd
          best_structure = pdb_file
    except Exception:
      continue
  return glycan, best_rmsd, best_structure


def get_similar_glycans(query_glycan, pdb_path = None, glycan_database = None, rmsd_cutoff = 2.0,
                        fast = False, unilectin_id = 0):
  """Search for structurally similar glycans by comparing against all available
  conformers/structures and keeping the best match for each glycan.
  Args:
    query_glycan (str): PDB file or coordinates of query structure
    pdb_path (str, optional): Optional specific path to query PDB file
    glycan_database (list, optional): List of candidate glycan structures
    rmsd_cutoff (float): Maximum RMSD to consider as similar
    fast (bool): Whether to use SVD-based Kabsch algorithm with k-d trees or Nelder-Mead optimization. Defaults to the latter
    unilectin_id (int): if pdb_path=='unilectin', will retrieve that structure ID from unilectin; Defaults to the first
  Returns:
    List of (glycan_id, rmsd, best_structure) tuples sorted by similarity
  """
  query_glycan = canonicalize_iupac(query_glycan)
  glycans = get_glycoshape_IUPAC() if glycan_database is None else glycan_database
  glycans = [g for g in glycans if ((get_global_path() if global_path is None else global_path) / g).exists() and any(((get_global_path() if global_path is None else global_path) / g).iterdir()) and g != query_glycan]
  # Get query coordinates once
  query_glycan_path = get_example_pdb(query_glycan) if pdb_path is None else pdb_path
  query_coords, _ = extract_glycan_coords(query_glycan_path) if pdb_path != 'unilectin' else extract_glycan_coords(unilectin_data[query_glycan][unilectin_id][0])
  # Prepare args for parallel processing
  process_args = [(g, query_coords, rmsd_cutoff, fast) for g in glycans]
  results = []
  with Pool() as pool:
    for glycan, rmsd, best_structure in tqdm(pool.imap_unordered(_process_single_glycan, process_args),
                                             total = len(glycans), desc = "Searching for similar glycans"):
      if rmsd <= rmsd_cutoff and best_structure is not None:
        conformer = '_'.join(best_structure.stem.split('_')[-2:])
        results.append({
                'glycan': glycan,
                'rmsd': round(rmsd, 3),
                'conformer': conformer
                })
  return sorted(results, key = lambda x: x['rmsd'])