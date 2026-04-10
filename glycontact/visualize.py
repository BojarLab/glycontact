import seaborn as sns
import networkx as nx
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import py3Dmol
from io import BytesIO
from pathlib import Path
from scipy import stats
from scipy.cluster import hierarchy
from collections import Counter
from typing import Dict, Tuple, List, Optional, Union
from IPython.display import Image, display, HTML
from glycontact.process import (inter_structure_variability_table, get_structure_graph, structure_graphs, df_to_pdb_content,
                                monosaccharide_preference_structure, map_dict, get_example_pdb, extract_3D_coordinates, unilectin_data)
from glycowork.glycan_data.stats import cohen_d
from glycowork.motif.draw import GlycoDraw
from glycowork.motif.processing import canonicalize_iupac, rescue_glycans
from glycowork.motif.graph import compare_glycans


def draw_contact_map(act, filepath='', size = 0.5, return_plot=False) :
  """Visualizes an atom contact map as a heatmap.
  Args:
      act (pd.DataFrame): The atom contact table from make_atom_contact_table()
      filepath (str, optional): Path to save the figure. If empty, no file is saved.
      size (float, optional): Text size for the plot. Defaults to 0.5.
      return_plot (bool, optional): If True, returns the plot object. Defaults to False.
  Returns:
      matplotlib.axes.Axes or None: Heatmap object if return_plot is True, None otherwise.
  """
  sns.set(font_scale=size)
  heatmap = sns.heatmap(act,xticklabels=True,yticklabels=True, cmap="magma")
  heatmap.set_yticklabels(heatmap.get_xticklabels(), rotation=0)
  if filepath:
    plt.savefig(filepath, dpi = 300, bbox_inches='tight')
  if return_plot:
    return heatmap
  else:
    plt.show()


def make_gif(prefix, tables):
  """Creates an animated GIF from a series of contact map visualizations.
  Args:
      prefix (str): Prefix for the output filename.
      tables (list): List of contact tables to animate, one per frame.
  Returns:
      None: Displays the resulting GIF animation.
  """
  output_path = f'{prefix}_animation.gif'
  images = []
  for table in tables:
    ax = draw_contact_map(table, return_plot=True)
    fig = ax.figure
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    images.append(imageio.imread(buf))
    plt.close(fig)  # Clean up memory
  if images:
    imageio.mimsave(output_path, images, duration=0.2, loop=0)
    display(Image(filename=output_path))


def show_correlations(corr_df,font_size=1):
  """Visualizes a correlation matrix as a heatmap.
  Args:
      corr_df (pd.DataFrame): Correlation matrix as a DataFrame.
      font_size (float, optional): Font size for the plot. Defaults to 1.
  Returns:
      None: Displays the heatmap.
  """
  sns.set(font_scale=font_size)
  # Visualize the correlation matrix as a heatmap
  plt.figure(figsize=(10, 8))
  sns.heatmap(corr_df, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
  plt.title('Correlation Matrix of Distances')
  plt.tight_layout()
  plt.show()


def show_correlation_dendrogram(corr_df, font_size = 1):
  """Creates a hierarchical clustering dendrogram from a correlation matrix.
  Args:
      corr_df (pd.DataFrame): Correlation matrix as a DataFrame.
      font_size (float, optional): Font size for the plot. Defaults to 1.
  Returns:
      dict: Dictionary mapping cluster colors to lists of monosaccharides in each cluster.
  """
  plt.figure(figsize=(10, 8))
  linkage = hierarchy.linkage(corr_df.values, method='ward')
  dendrogram = hierarchy.dendrogram(linkage, labels=corr_df.columns, leaf_rotation=90, leaf_font_size=8)
  # Group results by cluster
  monolist = dendrogram['ivl']
  clustlist = dendrogram['leaves_color_list']
  res_dict = {color: [] for color in set(clustlist)}
  for mono, clust in zip(monolist, clustlist):
    res_dict[clust].append(mono)
  plt.tick_params(axis='x', which='major', labelsize=font_size)
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('Residue')
  plt.ylabel('Distance')
  plt.tight_layout()
  plt.show()
  return res_dict


def plot_monosaccharide_instability(glycan, filepath='', mode='sum'):
  """Plots monosaccharide variability across different clusters of a glycan.
  Args:
      glycan (str): IUPAC glycan sequence.
      filepath (str, optional):  Path prefix for output file. If empty, no file is saved. 
      mode (str, optional): Method for aggregating variability ('sum' or 'mean'). Defaults to 'sum'.
  Returns:
      None: Displays the plot and optionally saves to file.
  """
  variability_table = inter_structure_variability_table(glycan)
  stability_scores = (variability_table.sum() if mode == 'sum' else variability_table.mean())
  sorted_scores = sorted(stability_scores.items(), key=lambda x: x[1])
  plt.figure(figsize=(12, 6))
  bars, labels = zip(*sorted_scores)
  plt.bar(bars, labels)
  plt.xlabel('Monosaccharides')
  plt.ylabel('Variability score')
  plt.xticks(rotation=90)
  plt.tight_layout()
  if filepath:
    plt.savefig(f'{filepath}{glycan}_instabilities.pdf', dpi = 300, bbox_inches='tight')
  plt.show()


def plot_glycan_score(glycan, score_list=[], attribute="SASA", filepath='') :
  """Displays a glycan structure with monosaccharides highlighted according to a score.
  Args:
      glycan (str): IUPAC glycan sequence.
      score_list (list, optional): List of values for highlighting monosaccharides. 
                                 If empty, uses values from the structure graph.
      attribute (str, optional): Attribute to use from structure graph if score_list is empty. 
                               Defaults to "SASA".
      filepath (str, optional): Path prefix for output file. If empty, no file is saved. 
  Returns:
      GlycoDraw: The rendered glycan drawing object.
  """
  glycan = canonicalize_iupac(glycan)
  if not score_list:
    ggraph = get_structure_graph(glycan)
    scores = np.array(list(nx.get_node_attributes(ggraph, attribute).values()))
  else:
    scores = np.array(score_list[:-1])  # Remove -R value
  # Normalize scores
  score_range = scores.max() - scores.min()
  normalized_scores = (scores - scores.min()) / score_range if score_range > 0 else np.zeros_like(scores)
  filepath = f"{filepath}{glycan}_highlighted.pdf" if filepath else ''
  return GlycoDraw(glycan, per_residue=normalized_scores.tolist(), filepath=filepath)


def show_monosaccharide_preference_structure(df, monosaccharide, threshold, mode='default'):
  """Visualizes preference statistics for a specific monosaccharide type.
  Args:
      df (pd.DataFrame): Monosaccharide distance table.
      monosaccharide (str): Target monosaccharide type.
      threshold (float): Distance threshold for interactions.
      mode (str, optional): Analysis mode ('default', 'monolink', or 'monosaccharide'). 
                          Defaults to 'default'.
  Returns:
      None: Displays a histogram of monosaccharide preferences.
  """
  res_dict = monosaccharide_preference_structure(df, monosaccharide, threshold, mode)
  value_counts = Counter(res_dict.values())
  # Plotting the histogram
  plt.bar(value_counts.keys(), value_counts.values())
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.title(f'Frequency of Encountered Values for {monosaccharide} above {threshold}')
  plt.tight_layout()
  plt.show()


def add_snfg_symbol(view, center, mono_name, alpha=1.0):
  """Adds a 3D Symbol Nomenclature for Glycans (SNFG) representation to a py3Dmol view.
  Args:
      view (py3Dmol.view): The py3Dmol view object.
      center (numpy.ndarray): 3D coordinates for the center of the symbol.
      mono_name (str): Name of the monosaccharide (e.g., 'Neu5Ac', 'Gal', 'GlcNAc').
      alpha (float, optional): Transparency level (0.0-1.0). Defaults to 1.0.
  Returns:
      None: Modifies the view object in-place.
  """
  # Define SNFG mapping (monosaccharide to shape and color)
  snfg_map = {
          'Neu5Ac': {'shape': 'diamond', 'color': '#A15989'},  # Purple diamond for sialic acid
          'Neu5Gc': {'shape': 'diamond', 'color': '#91D3E3'},  # Turqoise diamond for sialic acid
          'GlcNAc': {'shape': 'cube', 'color': '#0385AE'},     # Blue cube for N-acetylglucosamine
          'GalNAc': {'shape': 'cube', 'color': '#FCC326'},     # Yellow cube for N-acetylgalactosamine
          'Gal': {'shape': 'sphere', 'color': '#FCC326'},      # Yellow sphere for galactose
          'Glc': {'shape': 'sphere', 'color': '#0385AE'},      # Blue sphere for glucose
          'Man': {'shape': 'sphere', 'color': '#058F60'},      # Green sphere for mannose
          'Fuc': {'shape': 'cone', 'color': '#C23537'},     # Red triangle for fucose
          'Rha': {'shape': 'cone', 'color': '#058F60'}     # Green triangle for rhamnose
          }
  if mono_name not in snfg_map:
    return  # Skip if monosaccharide not in mapping
  symbol_spec = snfg_map[mono_name]
  color = symbol_spec['color']
  # Make reference structure slightly transparent to distinguish
  # Add the appropriate shape based on SNFG specification
  if symbol_spec['shape'] == 'sphere':
    view.addSphere({
            'center': {'x': center[0], 'y': center[1], 'z': center[2]},
            'radius': 0.7,  # Larger than atom spheres
            'color': color,
            'alpha': alpha
            })
  elif symbol_spec['shape'] == 'cube':
    # Create cube using eight vertices and faces
    size = 1.0  # Size of cube
    view.addBox({
          'center': {'x': center[0], 'y': center[1], 'z': center[2]},
          'dimensions': {'w': size, 'h': size, 'd': size},
          'color': color,
          'alpha': alpha
          })
  elif symbol_spec['shape'] == 'diamond':
    # Create an octahedron (diamond) using cylinders for edges
    size = 0.8  # Adjust size
    # Define the six vertices of an octahedron relative to the center
    vertices = np.array([
            [center[0] + size, center[1], center[2]],       # +X
            [center[0] - size, center[1], center[2]],       # -X
            [center[0], center[1] + size, center[2]],       # +Y
            [center[0], center[1] - size, center[2]],       # -Y
            [center[0], center[1], center[2] + size],       # +Z
            [center[0], center[1], center[2] - size]        # -Z
            ])
    # Define the 12 edges by specifying pairs of vertices
    edges = [
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 4), (2, 5),
            (3, 4), (3, 5)
            ]
    # First add surface for filled faces
    vertices_list = vertices.tolist()
    faces = [
            [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],  # Right half
            [1, 2, 4], [1, 4, 3], [1, 3, 5], [1, 5, 2]   # Left half
            ]
    # Then add thinner cylinders for edges
    for edge in edges:
      start = vertices[edge[0]]
      end = vertices[edge[1]]
      view.addCylinder({
              'start': {'x': start[0], 'y': start[1], 'z': start[2]},
              'end': {'x': end[0], 'y': end[1], 'z': end[2]},
              'radius': 0.1,
              'color': color,
              'opacity': alpha
              })
  elif symbol_spec['shape'] == 'cone':
    # Create a cone using addArrow
    size = 0.8  # Adjust size
    height = size * 1.5
    radius = size / 2
    # Define the direction of the cone (e.g., along the +Z axis)
    direction = np.array([0, 1, 0])
    # Define start and end points
    start = center.tolist()
    end = (center + direction * height).tolist()
    view.addArrow({
            'start': {'x': start[0], 'y': start[1], 'z': start[2]},
            'end': {'x': end[0], 'y': end[1], 'z': end[2]},
            'radius': radius,  # Controls the base radius of the cone
            'mid': 0.01,      # Make arrow almost entirely cone
            'color': color, 'alpha': alpha, 'resolution': 32  # Higher resolution for smoother cone
            })


def _do_3d_plotting(pdb_file, coords, labels, view=None, color='', bond_color=None, alpha=1.0, show_snfg=True,
                    show_labels=False, show_volume=False, pos='ref'):
  """Internal function for 3D plotting of a glycan structure.
  Args:
      pdb_file (str): Path to PDB file.
      coords (numpy.ndarray): Nx3 array of atomic coordinates.
      labels (list): List of N atom labels (format: "residue_mono_atom").
      view (py3Dmol.view, optional): Existing py3Dmol view object. If None, creates new.
      color (str, optional): Color scheme for the structure.
      bond_color (str, optional): Color for bonds.
      alpha (float, optional): Transparency value (0.0-1.0). Defaults to 1.0.
      show_snfg (bool, optional): Whether to show SNFG symbols. Defaults to True.
      show_labels (bool, optional): Whether to show monosaccharide labels. Defaults to False.
      show_volume (bool, optional): Whether to show volume surface. Defaults to False.
      pos (str, optional): Position identifier ('ref' or other). Defaults to 'ref'.
  Returns:
      None: Modifies the view object in-place.
  """
  if view is None:
    view = py3Dmol.view(width=800, height=800)
  # Read PDB file for connectivity information
  pdb_content = open(pdb_file, 'r').read() if isinstance(pdb_file, (Path, str)) else df_to_pdb_content(pdb_file[0])
  if pos != 'ref':
    pdb_content = pdb_content.replace(" X ", " B ")
  # Create a new PDB content with updated coordinates
  new_pdb_lines, coord_idx = [], 0
  pdb_lines = pdb_content.split('\n')
  for line in pdb_lines:
    if line.startswith('ATOM') or line.startswith('HETATM'):
      # Skip hydrogen atoms as they're filtered in coords
      if line[12:16].strip()[0] == 'H':
        continue
      # Update coordinates in the PDB line
      new_line = (line[:30] + f"{coords[coord_idx][0]:8.3f}{coords[coord_idx][1]:8.3f}{coords[coord_idx][2]:8.3f}" + line[54:])
      new_pdb_lines.append(new_line)
      coord_idx += 1
    else:
      new_pdb_lines.append(line)
  # Add the model with updated coordinates
  view.addModel('\n'.join(new_pdb_lines), "pdb")
  if show_volume:
    view.addSurface(py3Dmol.VDW, {'opacity': 0.5})
        
  def get_mono_info(label):
    parts = label.split('_')
    return parts[0], parts[1]
        
  def get_atom_type(label):
    return label.split('_')[-1][0]
        
  # Group atoms by monosaccharide
  mono_groups = {}
  for i, (coord, label) in enumerate(zip(coords, labels)):
    if label.split('_')[-1].startswith('H'):
      continue  # Skip hydrogen atoms
    mono_id, mono_name = get_mono_info(label)
    if mono_id not in mono_groups:
      mono_groups[mono_id] = {'atoms': [], 'center': [], 'name': mono_name}
    atom_name = label.split('_')[-1]
    mono_groups[mono_id]['atoms'].append({
            'coord': coord,
            'name': atom_name,
            'type': get_atom_type(label),
            'idx': i,
            'full_label': label
            })
  # Add atoms and create bonds for each monosaccharide
  for mono_id, group in mono_groups.items():
    atoms = group['atoms']
    mono_name = group['name']
    is_sialic = mono_name in ['SIA', 'NGC']
    # Create lookup for atoms by name
    atom_lookup = {atom['name']: atom for atom in atoms}
    # Handle ring bonds
    ring_atoms = {'C2', 'C3', 'C4', 'C5', 'C6', 'O6'} if is_sialic else {'C1', 'C2', 'C3', 'C4', 'C5', 'O5'}
    # Add SNFG symbols and labels if requested
    if all(a in atom_lookup for a in ring_atoms):
      center = np.mean([atom_lookup[a]['coord'] for a in ring_atoms], axis=0)
      mono_groups[mono_id]['center'] = center
      mono_name = map_dict[mono_name][:-2]  # Remove linkage info
      if show_snfg:
        add_snfg_symbol(view, center, mono_name, alpha=alpha)
      if show_labels:
        offset = 1.5 if show_snfg else 1.0
        label_pos = center + np.array([0, 0, offset])
        view.addLabel(mono_name, {
                'position': {'x': label_pos[0], 'y': label_pos[1], 'z': label_pos[2]},
                'backgroundColor': bond_color,
                'fontColor': 'white',
                'fontSize': 12,
                'alpha': 0.8
                })
  view.setStyle({'stick': {}})
  if color:
    view.setStyle({'chain': 'B'}, {'stick':{'colorscheme': color}})


@rescue_glycans
def plot_glycan_3D(glycan, filepath=None, stereo=None, view=None, show_volume=False, volume_params={}, **plot_kwargs):
  """Creates a 3D visualization of a glycan structure from its IUPAC sequence.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): Stereochemistry specification ('alpha' or 'beta'). 
                            If None, inferred from sequence.
      filepath (str, optional): Path to PDB file to use.
      view (py3Dmol.view, optional): Existing py3Dmol view object. If None, creates new.
      show_volume (bool, optional): Whether to show volume surface. Defaults to False.
      volume_params (dict, optional): Parameters for volume rendering.
      **plot_kwargs: Additional arguments passed to _do_3d_plotting.
  Returns:
      py3Dmol.view: The configured view object with rendered glycan.
  """
  # Create view if not provided
  if view is None:
    view = py3Dmol.view(width=800, height=800)
  # Get structure data
  pdb_file = get_example_pdb(glycan, stereo=stereo) if not filepath else filepath
  coords_df = pdb_file[0] if isinstance(pdb_file, tuple) else extract_3D_coordinates(pdb_file)
  coords_df = coords_df[~coords_df['atom_name'].str.startswith('H')]
  coords = coords_df[['x', 'y', 'z']].values
  labels = [f"{row['residue_number']}_{row['monosaccharide']}_{row['atom_name']}" for _, row in coords_df.iterrows()]
  # Plot structure
  _do_3d_plotting(pdb_file, coords, labels, view=view, show_volume=show_volume, **plot_kwargs)
  # Set view options
  view.zoomTo()
  view.render()
  return view


def plot_superimposed_glycans(superposition_result, filepath='', animate=True, rotation_speed=1,
                              show_labels=False, show_snfg=True):
  """Creates a 3D visualization of superimposed glycan structures.
  Args:
      superposition_result (dict): Output from superimpose_glycans() function.
      filepath (str, optional): Path to save the visualization image. If empty, no file is saved.
      animate (bool, optional): Whether to animate the visualization. Defaults to True.
      rotation_speed (int, optional): Speed of rotation if animated. Defaults to 1.
      show_labels (bool, optional): Whether to show monosaccharide labels. Defaults to False.
      show_snfg (bool, optional): Whether to show SNFG symbols. Defaults to True.
  Returns:
      py3Dmol.view: The configured view object with rendered superimposed glycans.
  """
  view = py3Dmol.view(width=800, height=800)
  # Plot both structures
  _do_3d_plotting(superposition_result['ref_conformer'], superposition_result['ref_coords'], superposition_result['ref_labels'], view=view,
                   alpha=0.85, show_snfg=show_snfg, show_labels=show_labels)
  _do_3d_plotting(superposition_result['mobile_conformer'], superposition_result['transformed_coords'], superposition_result['mobile_labels'], view=view,
                   color='skyblueCarbon', alpha=1.0, show_snfg=show_snfg, show_labels=show_labels, pos="mobile")
  # Add RMSD information
  rmsd = superposition_result['rmsd']
  view.addLabel(f'RMSD: {rmsd:.2f} Å', {
          'position': {'x': superposition_result['ref_coords'][0][0],
                    'y': superposition_result['ref_coords'][0][1],
                    'z': superposition_result['ref_coords'][0][2] + 5},
          'backgroundColor': 'black',
          'fontColor': 'white',
          'fontSize': 14
          })
  # Set view options
  view.zoomTo()
  view.render()
  if filepath:
    capture_html = f"""
        <button onclick="saveImage()" style="padding: 8px 16px; margin: 10px 0;">Save Image</button>
        <div id="debug_output" style="color: #666;"></div>
        <script>
        function log(msg) {{
            console.log(msg);
            document.getElementById('debug_output').textContent = msg;
        }}
        
        function saveImage() {{
            log('Finding canvas...');
            
            // Try different selectors to find the canvas
            let canvas = document.querySelector('.viewer_3Dmoljs canvas') || 
                        document.querySelector('.mol-container canvas') ||
                        document.querySelector('canvas');
                        
            if (!canvas) {{
                log('Error: No canvas found!');
                return;
            }}
            
            log('Canvas found, getting data URL...');
            
            try {{
                let dataURL = canvas.toDataURL('image/png');
                log('Got data URL, creating download...');
                
                let link = document.createElement('a');
                link.download = '{filepath}';
                link.href = dataURL;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                log('Download triggered!');
            }} catch (error) {{
                log('Error: ' + error.message);
                console.error(error);
            }}
        }}
        </script>
        """
    display(HTML(capture_html))
    print(f"Click the 'Save Image' button above to save the visualization to {filepath}")
  if animate:
    view.spin(True)
  return view


def calculate_average_metric(graph, pattern_in, metric):
  """Calculate average of the specified metric across all nodes in the graph.
  Excludes nodes related to the pattern to ensure fair comparison.
  Args:
    graph: NetworkX DiGraph of glycan structure
    pattern: Pattern to exclude from calculation
    metric: Metric to average
  Returns:
    Average value of metric across non-pattern nodes
  """
  pattern_nodes = []
  pattern = pattern_in.replace('[', '').replace(']', '')
  for node, attrs in graph.nodes(data=True):
    if pattern in attrs.get('string_labels', '') or pattern in attrs.get('Monosaccharide', ''):
      pattern_nodes.append(node)
  predecessor_nodes = []
  for pattern_node in pattern_nodes:
    pred = list(graph.predecessors(pattern_node))
    if pred:  # Each node has exactly one predecessor
      pred = list(graph.predecessors(pred[0]))
      if pred:
        predecessor_nodes.append(pred[0])
  exclude_nodes = set(pattern_nodes + predecessor_nodes)
  metric_values = []
  for node, attrs in graph.nodes(data=True):
    if node in exclude_nodes:
      continue
    if metric not in attrs or not isinstance(attrs[metric], (int, float)):
      continue
    metric_values.append(attrs[metric])
  if not metric_values:
    return 0.0
  return sum(metric_values) / len(metric_values)


def find_difference(glycans, pattern=None, alternative=None, metric="SASA", struc_dict=structure_graphs, plot=False):
  """Analyze differences in glycan properties between twin pairs based on pattern presence/absence or pattern substitution.
  Args:
    glycans: List of glycan structures to analyze
    pattern: String pattern to analyze (e.g., "[Fuc(a1-6)]") for presence/absence comparison
    alternative: String pattern to compare against the first pattern (e.g., "a2-6" vs "a2-3")
                When provided, function compares substitution rather than presence/absence
    metric: Property to compare (default: "SASA")
    struc_dict: Dictionary of glycan structure graphs
    plot: Whether to generate and return a visualization (default: False)
  Returns:
    Dictionary with statistical analysis results
  """
  if struc_dict is None:
    raise ValueError("structure_graphs dictionary must be provided")
  twins = []
  if alternative is None:
    # Case 1: Presence/absence comparison (original functionality)
    if pattern is None:
      raise ValueError("pattern must be provided for presence/absence comparison")
    for g in glycans:
      if pattern in g:
        twin = [t for t in glycans if compare_glycans(t, g.replace(pattern,""))]
        if twin:
          twins.append((g, twin[0]))
  else:
    # Case 2: Substitution comparison
    if pattern is None:
      raise ValueError("pattern must be provided along with alternative")
    for g in glycans:
      if pattern in g:
        # Create the alternative version by substituting the pattern
        alt_version = g.replace(pattern, alternative)
        # Check if the alternative version exists in the dataset
        twin = [t for t in glycans if compare_glycans(t, alt_version)]
        if twin:
          twins.append((g, twin[0]))
  # Continue with the original analysis
  with_pattern_values, without_pattern_values = [], []
  for with_pat, without_pat in twins:
    graph_with = struc_dict.get(with_pat, '')
    graph_without = struc_dict.get(without_pat, '')
    if not (graph_with and graph_without):
      continue
    with_pat_value = calculate_average_metric(graph_with, pattern if alternative is None else pattern, metric)
    without_pat_value = calculate_average_metric(graph_without, pattern if alternative is None else alternative, metric)
    with_pattern_values.append(with_pat_value)
    without_pattern_values.append(without_pat_value)
  with_pattern_array = np.array(with_pattern_values)
  without_pattern_array = np.array(without_pattern_values)
  differences = with_pattern_array - without_pattern_array
  ttest_result = stats.ttest_rel(with_pattern_array, without_pattern_array)
  effect_size, _ = cohen_d(with_pattern_array, without_pattern_array, paired=True)
  # Create descriptive labels based on the comparison type
  if alternative is None:
    label_with = f'With {pattern}'
    label_without = f'Without {pattern}'
  else:
    label_with = f'With {pattern}'
    label_without = f'With {alternative}'
  results = {
    "pattern": pattern,
    "alternative": alternative,
    "metric": metric,
    "n_pairs": len(with_pattern_values),
    "with_pattern_mean": np.mean(with_pattern_array),
    "without_pattern_mean": np.mean(without_pattern_array),
    "mean_difference": np.mean(differences),
    "median_difference": np.median(differences),
    "std_difference": np.std(differences),
    "ttest_statistic": ttest_result.statistic,
    "ttest_pvalue": ttest_result.pvalue,
    "significant": ttest_result.pvalue < 0.05,
    "effect_size": effect_size,
    "raw_with_pattern": with_pattern_array.tolist(),
    "raw_without_pattern": without_pattern_array.tolist(),
    "twins": twins  # Include the identified twin pairs in the results
  }
  if plot:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = {
      label_without: without_pattern_array,
      label_with: with_pattern_array
    }
    sns.boxplot(data=data, ax=ax, width=0.5, palette="Set2")
    positions = [0, 1]
    # For connecting lines, use fixed positions without jitter
    for i in range(len(with_pattern_array)):
      ax.plot([positions[0], positions[1]], 
              [without_pattern_array[i], with_pattern_array[i]], 
              'k-', alpha=0.2, linewidth=0.5)
    # Add jittered points on top of lines
    for i, (label, values) in enumerate(data.items()):
      # Add jittered points
      np.random.seed(42)  # For consistent jitter
      noise = np.random.normal(0, 0.04, size=len(values))
      ax.scatter(noise + positions[i], values, alpha=0.7, s=40, edgecolor='white', linewidth=0.5)
    ax.set_ylabel(f'{metric} Value')
    # Update title based on comparison type
    if alternative is None:
      ax.set_title(f'Comparison of {metric} with and without {pattern}')
    else:
      ax.set_title(f'Comparison of {metric} between {pattern} and {alternative}')
    if results["significant"]:
      p_text = f'p-value: {results["ttest_pvalue"]:.2e}'
      d_text = f", Cohen's d: {results['effect_size']:.2f}"
      stat_text = p_text + d_text
      ax.text(0.5, 0.01, stat_text, 
              horizontalalignment='center', verticalalignment='bottom', 
              transform=ax.transAxes, fontsize=12, 
              bbox=dict(facecolor='lightgray', alpha=0.5))
    ax.text(0, max(without_pattern_array) * 1.05, 
            f'Mean: {results["without_pattern_mean"]:.3f}', 
            horizontalalignment='center')
    ax.text(1, max(with_pattern_array) * 1.05, 
            f'Mean: {results["with_pattern_mean"]:.3f}', 
            horizontalalignment='center')
    plt.tight_layout()
    results["plot"] = fig
  return results


def extract_torsion_angles(disaccharide: str, 
                          structure_graphs: Dict[str, nx.DiGraph] = structure_graphs
                          ) -> Tuple[List[float], List[float]]:
  """Extract phi and psi torsion angles for a specific disaccharide linkage across all glycan structures.
  Args:
    disaccharide: String representing the disaccharide (e.g., "Fuc(a1-2)Gal")
    structure_graphs: Dictionary of glycan structure graphs
  Returns:
    Tuple of (phi_angles, psi_angles) lists
  """
  # Parse the disaccharide pattern
  parts = disaccharide.split("(")
  successor_sugar = parts[0]  # e.g., "Fuc"
  linkage_parts = parts[1].split(")")
  linkage = linkage_parts[0]  # e.g., "a1-2"
  predecessor_sugar = linkage_parts[1]  # e.g., "Gal"
  # Collect torsion angles
  phi_angles, psi_angles = [], []
  # Search through all glycan structures
  for glycan_name, graph in structure_graphs.items():
    # Find all nodes that match the linkage
    for node, attrs in graph.nodes(data=True):
      if attrs.get('string_labels', '') != linkage:
        continue
      # Check if this linkage connects the correct sugars
      successors = list(graph.successors(node))
      predecessors = list(graph.predecessors(node))
      if not successors or not predecessors:
        continue
      successor_node = successors[0]
      predecessor_node = predecessors[0]
      successor_attrs = graph.nodes[successor_node]
      predecessor_attrs = graph.nodes[predecessor_node]
      successor_label = successor_attrs.get('string_labels', '')
      predecessor_label = predecessor_attrs.get('string_labels', '')
      if successor_sugar in successor_label and predecessor_sugar in predecessor_label:
        # Found a matching disaccharide linkage, extract angles
        phi = attrs.get('phi_angle')
        psi = attrs.get('psi_angle')
        if phi is not None and psi is not None:
          phi_angles.append(phi)
          psi_angles.append(psi)
  return phi_angles, psi_angles


def ramachandran_plot(disaccharide: str, 
                      structure_graphs: Dict[str, nx.DiGraph] = structure_graphs,
                      density: bool = True,
                      filepath: Optional[str] = None) -> plt.Figure:
  """Generate a Ramachandran plot for a specific disaccharide linkage.
  Args:
    disaccharide: String representing the disaccharide (e.g., "Fuc(a1-2)Gal")
    structure_graphs: Dictionary of glycan structure graphs
    density: Whether to show density contours (default: True)
    filepath: Path to save the figure (optional)
  Returns:
    Matplotlib figure object
  """
  phi_angles, psi_angles = extract_torsion_angles(disaccharide, structure_graphs=structure_graphs)
  if not phi_angles or not psi_angles:
    raise ValueError(f"No torsion angles found for disaccharide: {disaccharide}")
  fig, ax = plt.subplots(figsize=(8, 8))
  # Create the scatter plot
  scatter = ax.scatter(phi_angles, psi_angles, alpha=0.7, s=30, c='blue', edgecolor='white')
  # Add density contours if requested
  if density and len(phi_angles) > 3:
    # Calculate the point density
    xy = np.vstack([phi_angles, psi_angles])
    z = stats.gaussian_kde(xy)(xy)
    # Sort the points by density for better visualization
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = np.array(phi_angles)[idx], np.array(psi_angles)[idx], z[idx]
    # Clear the previous plot and redraw with density coloring
    ax.clear()
    scatter = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=30, cmap='viridis', edgecolor='white')
    contour = ax.tricontour(phi_angles, psi_angles, z, levels=5, linewidths=0.5, colors='k', alpha=0.4)
    plt.colorbar(scatter, ax=ax, label='Density')
  # Set plot limits and labels
  ax.set_xlim(-180, 180)
  ax.set_ylim(-180, 180)
  ax.set_xlabel('Phi (°)')
  ax.set_ylabel('Psi (°)')
  ax.set_title(f'Ramachandran Plot for {disaccharide}')
  # Add grid lines
  ax.grid(alpha=0.3)
  # Add the origin lines
  ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
  ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
  # Add statistics
  stats_text = (
    f"n = {len(phi_angles)}\n"
    f"Mean φ = {np.mean(phi_angles):.1f}° ± {np.std(phi_angles):.1f}°\n"
    f"Mean ψ = {np.mean(psi_angles):.1f}° ± {np.std(psi_angles):.1f}°"
  )
  ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
  plt.tight_layout()
  if filepath:
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
  return fig
