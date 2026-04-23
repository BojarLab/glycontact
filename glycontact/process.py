import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
import glob
import copy
import datetime
import tempfile
import tarfile
from collections import Counter, defaultdict
from itertools import combinations
import subprocess
import json
import requests
import shutil
import pickle
import zipfile
from random import Random
from io import StringIO
from tqdm import tqdm
from pathlib import Path
from urllib.parse import quote
from typing import Tuple, Dict, List
from scipy.stats import chi2
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
from multiprocessing import Pool
from glycowork.glycan_data.loader import DataFrameSerializer
from glycowork.motif.graph import glycan_to_nxGraph, glycan_to_graph
from glycowork.motif.annotate import get_k_saccharides
from glycowork.motif.processing import canonicalize_iupac, rescue_glycans, min_process_glycans
from glycowork.motif.tokenization import stemify_glycan
import mdtraj as md

# MAN indicates either alpha and beta bonds, instead of just alpha.. this is a problem
# GalNAc is recorded as "GLC" which is wrong: need for a checker function that counts the number of atoms - Glc = 21 (<25), GalNAc = 28 (>25)
map_dict = {'NDG':'GlcNAc(a','NAG':'GlcNAc(b','MAN':'Man(a', 'BMA':'Man(b', 'AFL':'Fuc(a', 'MBG':'Gal1Me(b', 'AMG':'Gal1Me(a', 'MMA':'Man1Me(a', '2F8':'GlcNAc1Me(a',
              'FUC':'Fuc(a', 'FUL':'Fuc(b', 'FCA':'dFuc(a', 'FCB':'dFuc(b', '0FA':'D-Fuc(a', 'GYE':'dFucf(b', 'M6P':'Man6P(a', 'MAG':'GlcNAc1Me(b', 'SGA': 'Gal3S(b', 'SEJ':'D-Ara(b', '64K':'D-Ara(a',
              'GAL':'Gal(b', 'GLA':'Gal(a', 'GIV':'lGal(b', 'GXL':'lGal(a', 'GZL':'Galf(b', '2kA': 'L-Gul(a', '0mA': 'L-Man(a', 'GYP':'Glc1Me(a', 'MFU':'Fuc1Me', 'MFB':'Fuc1Me(b', 'MNA':'Neu2Me5Ac(a',
              'GLC':'Glc(a', '0WB':'ManNAc(b', 'ZAD':'Ara(b', '0aU':'Ara(b', '2aU':'Ara(b', '3aU':'Ara(b', '0aD':'Ara(a', '2aD':'Ara(a', '3aD':'Ara(a', 'YIO':'Gal1S(b', 'BDF':'Fru(b', 'RIP':'Rib(b',
              'IDR':'IdoA(a', 'RAM':'Rha(a', 'RHM':'Rha(b', 'RM4':'Rha(b', 'XXR':'D-Rha(a', '0aU': 'Araf(b', '2aU': 'Araf(b', '3aU': 'Araf(b', 'ZaU': 'Araf(a', 'TYV':'Tyv(a', 'ABE':'Abe(a', 'ARW':'D-Ara1Me(b',
              '0AU':'Ara(b', '2AU':'Ara(b', '3AU':'Ara(b', '0AD':'Ara(a', '2AD':'Ara(a', '3AD':'Ara(a', '3HA': 'D-Rha(a', 'ARB': 'D-Ara(b', 'MUR': 'MurNAc(b', 'MGC':'GalNAc1Me(a', 'GXL':'L-Gal(a',
              'A2G':'GalNAc(a', 'NGA': 'GalNAc(b', 'YYQ':'lGlcNAc(a', 'XYP':'Xyl(b', 'XYS':'Xyl(a', 'WOA': 'GalA(b', '3OA': 'GalA(a', 'TOA': 'GalA(a', 'GCV':'GlcA4Me(a', 'VDV':'Allf(a', 'VDS':'Allf(b',
              'XYZ':'Xylf(b', '1CU': 'Fru(b',  '0CU': 'Fru(b', '4CD': 'Fru(a', '1CD': 'Fru(a', 'LXC':'lXyl(b', 'HSY':'lXyl(a', 'SIA':'Neu5Ac(a', 'SLB':'Neu5Ac(b', 'FRU': 'Fru(b', 'AHR':'Araf(a', '2GS':'Gal2Me(a',
              'NGC':'Neu5Gc(a', 'NGE':'Neu5Gc(b', 'BDP':'GlcA(b', 'GCU':'GlcA(a','VYS':'GlcNS(a', '0YS':'GlcNS(a', '4YS':'GlcNS(a', '6YS':'GlcNS(a', 'UYS':'GlcNS(a', 'QYS':'GlcNS(a', 'GCS':'GlcN(b',
              'PA1':'GlcN(a', 'ROH':' ', 'BGC':'Glc(b', '0OA':'GalA(a', '4OA':'GalA(a', 'BCA':'2-4-diacetimido-2-4-6-trideoxyhexose(a', 'ASO':'1,5-Anhydro-Glc(?', 'L6N':'Glc6Me(?', 'BM3':'ManNAc(a',
              "NAG6SO3":"GlcNAc6S(b", "NDG6SO3":"GlcNAc6S(a", "GLC4SO3":"GalNAc4S(b", "NGA4SO3":"GalNAc4S(b", 'A2G4SO3':'GalNAc4S(a', "IDR2SO3":"IdoA2S(a", '289':'DDManHep(a',
              "BDP3SO3":"GlcA3S(b", "BDP2SO3":"GlcA2S(b", "GCU2SO3":"GlcA2S(a", "SIA9ACX":"Neu5Ac9Ac(a", "MAN3MEX":"Man3Me(a", '5N6':'Neu5Ac9Ac(a', 'PKM':'Neu4Ac5Ac(a', 'GMH':'LDManHep(a',
              "SIA9MEX":"Neu5Ac9Me(a", "NGC9MEX":"Neu5Gc9Me(a", "BDP4MEX":"GlcA4Me(b", "GAL6SO3":"Gal6S(b", "NDG3SO3":"GlcNAc3S6S(a", "TOA2SO3": "GalA2S(a", 'GN1':'GlcNAc1P(a',
              "NAG6PCX":"GlcNAc6PCho(b", "UYS6SO3":"GlcNS6S(a", 'VYS3SO3':'GlcNS3S6S(a',  'VYS6SO3':'GlcNS3S6S(a', "QYS3SO3":"GlcNS3S6S(a", "QYS6SO3":"GlcNS3S6S(a", "4YS6SO3":"GlcNS6S(a", "6YS6SO3":"GlcNS6S(a",
              "FUC2MEX3MEX4MEX": "Fuc2Me3Me4Me(a", "QYS3SO36SO3": "GlcNAc3S6S(a", "VYS3SO36SO3": "GlcNS3S6S(a", "NDG3SO36SO3": "GlcNS3S6S(a", "RAM2MEX3MEX": "Rha2Me3Me(a",
            "SIO":"Neu4Ac5Ac9Ac(a", "1GN":"GalN(b", "KD5":"4,7-Anhydro-Kdof(a", "BDR":"Ribf(b", "G1P":"Glc1P(a", "3LJ":"GlcN6S(a", "SGN":"GlcNS6S(a", "95Z":"ManN(a", "GCS":"GlcN(b", "ADA":"GalA(a",
            "GTR":"GalA(b", "3MG":"Glc3Me(b", "ZB1":"Glc3Me(a", "NGS":"GlcNAc6S(b", "ANA":"Neu2Me4Ac5Ac(a", "M6D":"Man6P(b", "G6S":"Gal6S(b", "GL0":"Gul(b", "ZEL":"D-Alt1Me(b", "EGA":"Gal1Et(b",
            "ARA":"Ara(a", "2FG":"Gal2F(b", "MN0":"Neu2Me5Gc(a", "PZU":"Par(a", "A1Q":"LDManHepOMe(a", "GQ1":"Glc4S(a", "G4S":"Gal4S(b", "6S2":"GlcNAc1Me6S(b", "6C2":"GlcNAcA1Me(b",
            "X6X":"GalN(a", "TVD":"GlcNAc1NAc(b", "MJJ":"Neu2Me5Ac9Ac(a", "K5B":"4,7-Anhydro-Kdof(b", "GAL3SO3": "Gal3S(b", "GAL3SO36SO3": "Gal3S6S(b", "GAL4SO36SO3": "Gal4S6S(b", "GAL4SO3": "Gal4S(b",
            "A2G6SO3": "GalNAc6S(a", "GLC6SO3": "Glc6S(a", "0KN": "Kdn(a", "0eB": "Alt(a", "0bA": "Sor(a", "0JA": "Tag(a", "0NB": "D-All(a", 'FUC2MEX': 'Fuc2Me(a', 'FUC3MEX': 'Fuc3Me(a', 'FUC4MEX': 'Fuc4Me(a',
            'RAM2MEX': 'Rha2Me(a', 'RAM3MEX': 'Rha3Me(a', 'FUC2MEX3MEX': 'Fuc2Me3Me(a', 'FUC2MEX4MEX': 'Fuc2Me4Me(a', 'FUC3MEX4MEX': 'Fuc3Me4Me(a', 'RAM2MEX3MEX': 'Rha2Me3Me(a', '0WA': 'ManNAc(a', '0RU': 'Ribf(a',
            '0QB': 'Qui(a', '0PD': 'Psif(a', '0DA': 'Lyx(a', '0kB': 'L-Gul(a', '0tA': 'L-Tal(a', 'NBG': 'GlcNAc(b', 'KDO': 'Kdo(a',}
NON_MONO = {'SO3', 'ACX', 'MEX', 'PCX'}
BETA = {'GlcNAc', 'Glc', 'Xyl'}
C2_PATTERN = 'NGC|SIA|NGE|4CD|0CU|1CU|1CD|FRU|5N6|PKM|0KN|0bA|0JA'

this_dir = Path(__file__).parent

original_path = Path(os.getenv('GLYCONTACT_DATA_PATH')) if os.getenv('GLYCONTACT_DATA_PATH') else this_dir / 'glycans_pdb'
fallback_path = this_dir / 'GlycoShape.zip'
json_path = this_dir / "20260417_GLYCOSHAPE.json"
with open(json_path) as f:
    glycoshape_mirror = json.load(f)


def gsid_conversion(glycan) :
    """ Convert an input glycan from glytoucan ID, glycoshape ID or IUPAC format into the iupac format.
    Args:
    glycan (str): glycan sequence in iupac format, glytoucan ID or glycoshape ID
    Returns:
    String of the converted input glycan in the iupac format
    """
    for key, entry in glycoshape_mirror.items():
        if key == glycan :
            return entry.get('iupac', None)
        if entry.get('ID') == glycan :
            return entry.get('iupac', None)
    return glycan


def fetch_and_convert_pdbs(base_output = "glycontact/glycans_pdb"):
    """Fetches all existing GlyToucan IDs from glycoshape.org and downloads their PDB files.
    Each file is renamed and stored in a folder corresponding to its IUPAC name:
        glycontact/glycans_pdb/{IUPAC}/{IUPAC}.pdb
    Args:
    base_output (str): path where PDB files are stored.
    """
    # Step 1: Get all GlyToucan IDs
    available_url = "https://glycoshape.org/api/available"
    print(f"Fetching available GlyToucan IDs from {available_url}...")
    response = requests.get(available_url)
    response.raise_for_status()
    ids = response.json()
    print(f"Found {len(ids)} available IDs.")
    base_pdb_url = "https://glycoshape.org/api/pdb/"
    # Step 2: Download each PDB
    for gly_id in ids:
        try:
            pdb_url = f"{base_pdb_url}{gly_id}"
            pdb_response = requests.get(pdb_url)
            pdb_response.raise_for_status()
            # Step 3: Convert ID → IUPAC
            iupac_name = gsid_conversion(gly_id)
            if not iupac_name:
                print(f"Could not convert {gly_id}, skipping.")
                continue
            # Step 4: Sanitize folder/file name
            safe_name = "".join(c if c.isalnum() or c in "-_()" else "_" for c in iupac_name)
            # Step 5: Create output path
            output_dir = Path(base_output) / safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{safe_name}.pdb"
            # Step 6: Save file
            with open(output_path, "wb") as f:
                f.write(pdb_response.content)
        except Exception as e:
            print(f"Failed to process {gly_id}: {e}")


def process_glycoshape(fallback_path):
  # Get the directory where the zip file is located
  base_dir = Path(fallback_path).parent
  # Create GlycoShape folder first
  glycoshape_dir = base_dir / "GlycoShape"
  glycoshape_dir.mkdir(exist_ok=True)
  # Extract main GlycoShape.zip into the GlycoShape folder
  with zipfile.ZipFile(fallback_path, 'r') as zip_ref:
    zip_ref.extractall(glycoshape_dir)
  # Remove original GlycoShape.zip
  Path(fallback_path).unlink()
  # Change working directory to GlycoShape folder
  original_cwd = os.getcwd()
  os.chdir(glycoshape_dir)
  try:
    # Get list of zip files first for tqdm
    zip_files = list(Path(".").glob("*.zip"))
    # Process all zip files within the GlycoShape folder
    for zip_file in tqdm(zip_files, desc = "Processing glycan structures"):
      # Get canonical glycan name for folder
      glycan_name = gsid_conversion(zip_file.stem)
      try:
        canonical_name = canonicalize_iupac(glycan_name)
      except ValueError:
        canonical_name = glycan_name
      glycan_folder = Path(canonical_name)
      glycan_folder.mkdir(exist_ok = True)
      # Extract individual zip file into its own folder
      with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(glycan_folder)
      # Remove the zip file after extraction
      zip_file.unlink()
    # Process all extracted glycan folders
    for item in Path(".").iterdir():
      if not item.is_dir():
        continue
      glycan_folder = item
      # Find PDB_format_ATOM subfolder
      pdb_atom_path = None
      for root, dirs, files in os.walk(glycan_folder):
        if "PDB_format_ATOM" in dirs:
          pdb_atom_path = Path(root) / "PDB_format_ATOM"
          break
      if pdb_atom_path and pdb_atom_path.exists():
        # Move all files from PDB_format_ATOM to glycan folder
        for file in pdb_atom_path.iterdir():
          if file.is_file():
            shutil.move(str(file), str(glycan_folder / file.name))
        # Remove everything except moved PDB files
        for item in glycan_folder.iterdir():
          if item.is_dir():
            shutil.rmtree(item)
          elif not item.suffix.lower() in ['.pdb']:
            item.unlink()
        # Rename files to remove .PDB part
        for file in glycan_folder.iterdir():
          if ".PDB." in file.name:
            new_name = file.name.replace(".PDB.", ".")
            file.rename(glycan_folder / new_name)
  finally:
    # Change back to original directory
    os.chdir(original_cwd)
  # Rename GlycoShape folder to glycans_pdb
  glycoshape_dir.rename(base_dir / "glycans_pdb")


def get_global_path():
  if original_path.exists():
    return original_path
  elif fallback_path.exists():
    print("Identified zipped GlycoShape structures. Starting extraction.")
    process_glycoshape(fallback_path)
    if original_path.exists() and any(original_path.iterdir()):
      print("Extraction succeeded. You should be good to go.")
      return original_path
    else:
      raise FileNotFoundError("Extraction of GlycoShape structures failed. If you followed all the steps described on https://github.com/BojarLab/glycontact, feel free to open an issue.")
  else:
    parent_zip_path = this_dir.parent / 'GlycoShape.zip'
    if parent_zip_path.exists():
      shutil.move(str(parent_zip_path), str(fallback_path))
      print("Found GlycoShape.zip one folder above. Moved to expected location.")
      print("Identified zipped GlycoShape structures. Starting extraction.")
      process_glycoshape(fallback_path)
      if original_path.exists() and any(original_path.iterdir()):
        print("Extraction succeeded. You should be good to go.")
        return original_path
      else:
        raise FileNotFoundError("Extraction of GlycoShape structures failed. If you followed all the steps described on https://github.com/BojarLab/glycontact, feel free to open an issue.")
    else:
      original_path.mkdir(parents = True, exist_ok = True)
      return original_path


if os.getenv('PYTEST_RUNNING') or os.getenv('SKIP_GLYCOSHAPE_CHECK'):
  global_path = Path('dummy_glycoshape_path')
else:
  global_path = None

with open(this_dir / "glycan_graphs.pkl", "rb") as file:
    structure_graphs = pickle.load(file)


class ComplexDictSerializer(DataFrameSerializer):
  """Extends DataFrameSerializer with methods to handle complex defaultdict structures."""

  @classmethod
  def serialize_complex_dict(cls, data_dict: defaultdict, path: str) -> None:
    """Serialize a defaultdict of (DataFrame, dict) tuples to a single JSON file"""
    # Convert defaultdict to a serializable structure
    serialized_dict = {}
    for key, tuple_list in data_dict.items():
      serialized_dict[str(key)] = []
      for df, d in tuple_list:
        # Serialize DataFrame to a dict
        serialized_df = {
          'columns': list(df.columns),
          'index': list(df.index),
          'data': [[cls._serialize_cell(val) for val in row] for _, row in df.iterrows()],
          'dtypes': {col: str(df[col].dtype) for col in df.columns}
        }
        serialized_dict[str(key)].append((serialized_df, d))
    # Write to file
    with open(path, 'w') as f:
      json.dump(serialized_dict, f)

  @classmethod
  def deserialize_complex_dict(cls, path: str) -> defaultdict:
    """Deserialize a defaultdict of (DataFrame, dict) tuples from a single JSON file"""
    with open(path, 'r') as f:
      serialized_dict = json.load(f)
    result = defaultdict(list)
    for key, pairs in serialized_dict.items():
      for serialized_df, d in pairs:
        # Deserialize DataFrame
        deserialized_data = []
        for row in serialized_df['data']:
          deserialized_row = [cls._deserialize_cell(cell) for cell in row]
          deserialized_data.append(deserialized_row)
        df = pd.DataFrame(
          data = deserialized_data,
          columns = serialized_df['columns'],
          index = serialized_df['index']
        )
        if 'dtypes' in serialized_df:
          for col, dtype_str in serialized_df['dtypes'].items():
            if col in df.columns:
              try:
                if 'int' in dtype_str:
                  df[col] = pd.to_numeric(df[col], errors = 'coerce').fillna(0).astype(int)
                elif 'float' in dtype_str:
                  df[col] = pd.to_numeric(df[col], errors = 'coerce')
              except:
                pass
        result[key].append((df, d))
    return result


unilectin_data = ComplexDictSerializer.deserialize_complex_dict(this_dir / "unilectin_data.json")

def convert_ID(input_ID, output_format = 'iupac'):
    """ Convert an input glycan from any format into any specified format.
    Args:
    input_ID (str): glycan sequence or ID
    output_format (str): authorized formats are: 'glytoucan', 'ID', 'iupac', 'glycam', 'wurcs', 'glycoct', 'smiles', 'oxford'
    Returns:
    String of the converted input glycan in the desired format
    """
    for key, entry in glycoshape_mirror.items():
        if key == input_ID :
            return entry.get(output_format, None)
        for format in ['ID', 'iupac', 'glycam', 'wurcs', 'glycoct', 'smiles', 'oxford'] :
            if entry.get(format) == input_ID:
                if output_format == 'glytoucan' :
                    return key
                else :
                    return entry.get(output_format, None)
    return "Not Found"


def fetch_pdbs(glycan, stereo = None, my_path = None):
  """Given a glycan sequence, will query first GlycoShape and then UniLectin for appropriate PDB files.
  Args:
  glycan (str): glycan sequence, preferably in IUPAC-condensed
  stereo (str, optional): specification of whether reducing end alpha or beta is desired
  my_path (Path, optional): custom path to PDB folder
  Returns:
  List of Paths for GlycoShape and list of get_annotation output tuples for UniLectin
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  glycan_path = (get_global_path() if global_path is None else global_path) / glycan if my_path is None else my_path
  if not os.path.exists(glycan_path):
    print(f"Glycan {glycan} not found locally. Downloading from GlycoShape...")
    try:
      download_from_glycoshape(glycan)
      if not os.path.exists(glycan_path):
        raise FileNotFoundError(f"Download failed or no structures available for glycan: {glycan}")
    except Exception as e:
      print(f"GlycoShape download failed: {e}. Trying UniLectin as fallback...")
      if glycan in unilectin_data:
        matching_pdbs = [(res, inty) for res, inty in unilectin_data[glycan] if res.IUPAC.tolist()[0].endswith(stereo[0]) or f"({stereo[0]}1-1)" in res.IUPAC.tolist()[0]]
        if not matching_pdbs:
          raise FileNotFoundError(f"No PDB files with '{stereo}' stereochemistry found for glycan: {glycan}")
        return matching_pdbs
      else:
        raise FileNotFoundError(f"Could not find glycan {glycan} in GlycoShape or UniLectin: {e}")
  matching_pdbs = [glycan_path / pdb for pdb in os.listdir(glycan_path) if stereo in pdb]
  if not matching_pdbs:
      fallback = 'alpha' if stereo == 'beta' else 'beta'
      matching_pdbs = [glycan_path / pdb for pdb in os.listdir(glycan_path) if fallback in pdb]
      if not matching_pdbs:
          raise FileNotFoundError(
              f"No PDB files with '{stereo}' or '{fallback}' stereochemistry found for glycan: {glycan}")
  return matching_pdbs


def get_glycoshape_IUPAC(fresh = False):
  """Retrieves a list of available glycans from GlycoShape database.
  Args:
  fresh (bool): If True, fetches data directly from GlycoShape API.
                   If False, uses cached data from the local mirror.
  Returns:
      set: Set of IUPAC-formatted glycan sequences available in the database.
  """
  if fresh:
    return json.loads(subprocess.run('curl -X GET https://glycoshape.org/api/available_glycans', shell = True, capture_output = True, text = True).stdout)['glycan_list']
  else:
    return set(entry["iupac"] for entry in glycoshape_mirror.values())


def download_from_glycoshape(IUPAC):
  """Downloads PDB files for a given IUPAC sequence from the GlycoShape database.
  Args:
      IUPAC (str): IUPAC-formatted glycan sequence to download.
  Returns:
      bool: False if IUPAC is improperly formatted, None otherwise.
  """
  if IUPAC[-1]==']':
    print('This IUPAC is not formatted properly: ignored')
    return False
  IUPAC_clean = canonicalize_iupac(IUPAC)
  base_path = get_global_path() if global_path is None else global_path
  outpath = base_path / IUPAC_clean
  os.makedirs(outpath, exist_ok = True)
  gly_id = None
  for key, entry in glycoshape_mirror.items():
    if entry.get('iupac') == IUPAC or entry.get('iupac') == IUPAC_clean:
      gly_id = key
      break
  if not gly_id:
    raise ValueError(f"Could not find GlyToucan ID for IUPAC: {IUPAC}")
  download_url = f"https://glycoshape.org/api/download/{gly_id}"
  response = requests.get(download_url)
  response.raise_for_status()
  with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = Path(tmpdir) / "glycan.zip"
    with open(zip_path, "wb") as f:
      f.write(response.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(tmpdir)
    new_format_files = list(Path(tmpdir).rglob("PDB/*.pdb"))
    if new_format_files:
      for pdb_file in new_format_files:
        stereo = pdb_file.stem
        if stereo not in ['alpha', 'beta']:
          continue
        with open(pdb_file, 'r') as f:
          lines = f.readlines()
        model_num, current_lines = 0, []
        for line in lines:
          if line.startswith('MODEL'):
            model_num = int(line.split()[1])
            current_lines = []
          elif line.startswith('ENDMDL'):
            if current_lines:
              cluster_name = f"cluster{model_num-1}_{stereo}.pdb"
              with open(outpath / cluster_name, 'w') as out:
                out.writelines(current_lines)
          elif not line.startswith('MODEL') and not line.startswith('ENDMDL'):
            current_lines.append(line)
    else:
      for item in Path(tmpdir).rglob("*.PDB.pdb"):
        if "cluster" in item.name and ("_alpha" in item.name or "_beta" in item.name):
          new_name = item.name.replace(".PDB.", ".")
          shutil.copy(str(item), str(outpath / new_name))


def extract_3D_coordinates(pdb_file):
  """Extracts 3D coordinates from a PDB file and returns them as a DataFrame.
  Args:
      pdb_file (str): Path to the PDB file.
  Returns:
      pd.DataFrame: DataFrame containing extracted atom coordinates with columns for
                   atom information, coordinates, and properties.
  """
  permitted = set(map_dict.keys()) | NON_MONO
  # Define column names for the DataFrame
  columns = ['record_name', 'atom_number', 'atom_name', 'monosaccharide', 'chain_id', 'residue_number',
           'x', 'y', 'z', 'occupancy', 'temperature_factor', 'element']
  has_protein, has_hetatm = False, False
  with open(pdb_file, 'r') as pdb_f:
    lines = pdb_f.readlines()
  for line in lines:
    if line.startswith(('SEQRES', 'DBREF')):  # Protein sequence indicators
      has_protein = True
    if line.startswith('HETATM'):
      has_hetatm = True
    if has_protein and has_hetatm:
      break
  # Open the PDB file for reading
  relevant_lines = [line for line in lines if line.startswith('HETATM')] if has_protein else [line for line in lines if line.startswith('ATOM')]
  # Handle NMR structures by relabeling chain IDs
  has_model = any(line.startswith('MODEL') for line in lines)
  if has_model and ((has_protein and has_hetatm) or not has_protein):
    modified_lines, current_model = [], 0
    record_type = 'HETATM' if has_protein else 'ATOM'
    for line in lines:
      if line.startswith('MODEL'):
        current_model = int(line.split()[1])
      elif line.startswith(record_type):
        # Relabel chain ID (at position 21)
        modified_line = f"{line[:21]}{chr(ord('A') + current_model - 1)}{line[22:]}"
        modified_lines.append(modified_line)
    relevant_lines = modified_lines
  # Read the relevant lines into a DataFrame using fixed-width format
  out = pd.read_fwf(StringIO(''.join(relevant_lines)), names = columns,
                    colspecs = [(0, 6), (6, 11), (12, 16), (17, 20), (20, 22), (22, 26),
                                (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78)])
  for col in ['x', 'y', 'z', 'occupancy', 'temperature_factor']:
      out[col] = pd.to_numeric(out[col], errors = 'coerce')
  out = out.dropna(subset = ['x', 'y', 'z'])
  return out[out.monosaccharide.isin(permitted)].reset_index(drop = True)


def make_atom_contact_table(coord_df, threshold = 10, mode = 'exclusive'):
  """Creates a contact table showing distances between atoms in a PDB structure.
  Args:
      coord_df (pd.DataFrame): Dataframe of coordinates from extract_3D_coordinates.
      threshold (float): Maximum distance to consider, longer distances set to threshold+1.
      mode (str): 'exclusive' to exclude intra-residue distances, 'inclusive' to include them.
  Returns:
      pd.DataFrame: Matrix of distances between atoms.
  """
  mono_nomenclature = 'IUPAC' if 'IUPAC' in coord_df else 'monosaccharide'
  coords = coord_df[['x', 'y', 'z']].values
  diff = coords[:, np.newaxis, :] - coords
  distances = np.abs(diff).sum(axis = 2)
  labels = [f"{num}_{mono}_{atom}_{anum}" for num, mono, atom, anum in
         zip(coord_df['residue_number'], coord_df[mono_nomenclature], coord_df['atom_name'], coord_df['atom_number'])]
  if mode == 'exclusive':
    # Mask intra-residue distances
    mask = coord_df['residue_number'].values[:, np.newaxis] != coord_df['residue_number'].values
    distances = np.where(mask, np.where(distances <= threshold, distances, threshold + 1), 0)
  else:
    distances = np.where(distances <= threshold, distances, threshold + 1)
  return pd.DataFrame(distances, index = labels, columns = labels)


def make_monosaccharide_contact_table(coord_df, threshold = 10, mode = 'binary'):
  """Creates a contact table at the monosaccharide level rather than atom level.
  Args:
      coord_df (pd.DataFrame): Dataframe of coordinates from extract_3D_coordinates.
      threshold (float): Maximum distance to consider.
      mode (str): 'binary' for binary contact matrix, 'distance' for distance values,
                 'both' to return both matrices.
  Returns:
      pd.DataFrame or list: Contact table(s) between monosaccharides.
  """
  mono_nomenclature = 'IUPAC' if 'IUPAC' in coord_df.columns else 'monosaccharide'
  residues = sorted(coord_df['residue_number'].unique())
  n_residues = len(residues)
  binary_matrix = np.ones((n_residues, n_residues))
  dist_matrix = np.full((n_residues, n_residues), threshold + 1)
  labels = [f"{i}_{coord_df[coord_df['residue_number'] == i][mono_nomenclature].iloc[0]}" for i in residues]
  coords_by_residue = {res: coord_df[coord_df['residue_number'] == res][['x','y','z']].values for res in residues}
  for i, res1 in enumerate(residues):
    coords1 = coords_by_residue[res1]
    for j, res2 in enumerate(residues[i:], i):
      coords2 = coords_by_residue[res2]
      # Compute all pairwise distances
      diffs = coords1[:, np.newaxis, :] - coords2
      distances = np.abs(diffs).sum(axis = 2)
      min_dist = np.min(distances)
      if min_dist <= threshold:
        binary_matrix[i, j] = binary_matrix[j, i] = 0
        dist_matrix[i, j] = dist_matrix[j, i] = min_dist
  if mode == 'binary':
    return pd.DataFrame(binary_matrix, index = labels, columns = labels)
  if mode == 'distance':
    return pd.DataFrame(dist_matrix, index = labels, columns = labels)
  return [pd.DataFrame(binary_matrix, index = labels, columns = labels),
        pd.DataFrame(dist_matrix, index = labels, columns = labels)]


def focus_table_on_residue(table, residue) :
  """Filters a monosaccharide contact table to keep only one residue type.
  Args:
      table (pd.DataFrame): Monosaccharide contact table.
      residue (str): Residue type to focus on (e.g., 'MAN').
  Returns:
      pd.DataFrame: Filtered contact table.
  """
  mask = table.columns.str.contains(residue, regex = False)
  return table.loc[mask, mask]


def get_contact_tables(glycan, stereo = None, level = "monosaccharide", my_path = None):
  """Gets contact tables for a given glycan across all its PDB structures.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      level (str): 'monosaccharide' or 'atom' to determine detail level.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      list: List of contact tables for each PDB structure.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  dfs, _ = annotation_pipeline(glycan, my_path = my_path, threshold = 3.5, stereo = stereo)
  if level == "monosaccharide":
    return [make_monosaccharide_contact_table(df, mode = 'distance', threshold = 200) for df in dfs if len(df) > 0]
  else:
    return [make_atom_contact_table(df, mode = 'distance', threshold = 200) for df in dfs if len(df) > 0]


@rescue_glycans
def inter_structure_variability_table(glycan, stereo = None, mode = 'standard', my_path = None, fresh = False):
  """Creates a table showing stability of atom/monosaccharide positions across different PDB structures of the same glycan.
  Args:
      glycan (str or list): Glycan in IUPAC sequence or list of contact tables.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      mode (str): 'standard', 'amplify', or 'weighted' for different calculation methods.
      my_path (str, optional): Custom path to PDB folders.
      fresh (bool): If True, fetches fresh cluster frequencies.
  Returns:
      pd.DataFrame: Variability table showing how much positions vary across structures.
  """
  if isinstance(glycan, str):
    dfs = get_contact_tables(glycan, stereo, my_path = my_path)
  elif isinstance(glycan, list):
    dfs = glycan
  if len(dfs) < 1:
    return pd.DataFrame()
  if stereo is None and isinstance(glycan, str):
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  columns = dfs[0].columns
  values_array = np.array([df.values for df in dfs])
  mean_values = np.mean(values_array, axis = 0)
  deviations = np.abs(values_array - mean_values)
  if mode == 'weighted':
    weights = np.array(get_all_clusters_frequency(fresh = fresh).get(glycan, [100.0])) / 100
    weights = [1.0] * len(dfs) if len(weights) != len(dfs) else weights
    result = np.average(deviations, weights = weights, axis = 0)
  elif mode == 'amplify':
    result = np.sum(deviations, axis = 0) ** 2
  else:  # standard mode
    result = np.sum(deviations, axis = 0)
  return pd.DataFrame(result, columns = columns, index = columns)


@rescue_glycans
def inter_structure_torsion_variability(glycan, stereo = None, mode = 'standard', my_path = None, fresh = False):
  """Creates a table showing variability of torsion angles across different PDB structures of the same glycan.
  Args:
      glycan (str): Glycan in IUPAC sequence.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      mode (str): 'standard', 'amplify', or 'weighted' for different calculation methods.
      my_path (str, optional): Custom path to PDB folders.
      fresh (bool): If True, fetches fresh cluster frequencies.
  Returns:
      pd.DataFrame: Variability table showing how much torsion angles vary across structures.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  dfs, int_dicts = annotation_pipeline(glycan, threshold = 3.5, stereo = stereo, my_path = my_path)
  if len(dfs) < 1:
    return pd.DataFrame()
  torsion_tables = []
  for df, int_dict in zip(dfs, int_dicts):
    if len(df) > 0:
      torsion_df = get_glycosidic_torsions(df, int_dict)
      if len(torsion_df) > 0:
        torsion_tables.append(torsion_df)
  if len(torsion_tables) < 2:
    return pd.DataFrame()
  linkages = torsion_tables[0]['linkage'].tolist()
  phi_values = np.array([[table[table['linkage'] == link]['phi'].iloc[0] if len(table[table['linkage'] == link]) > 0 else np.nan for link in linkages] for table in torsion_tables])
  psi_values = np.array([[table[table['linkage'] == link]['psi'].iloc[0] if len(table[table['linkage'] == link]) > 0 else np.nan for link in linkages] for table in torsion_tables])
  omega_raw = [[table[table['linkage'] == link]['omega'].iloc[0] if len(table[table['linkage'] == link]) > 0 else np.nan for link in linkages] for table in torsion_tables]
  omega_values = np.array([[val if val is not None else np.nan for val in row] for row in omega_raw])
  def circular_std(angles):
    valid_mask = pd.notna(angles) & (angles != None)
    valid_angles = angles[valid_mask]
    if len(valid_angles) == 0:
      return np.nan
    angles_rad = np.radians(valid_angles)
    mean_angle = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
    circular_var = 1 - np.sqrt(np.mean(np.cos(angles_rad))**2 + np.mean(np.sin(angles_rad))**2)
    return np.degrees(np.sqrt(circular_var))
  if mode == 'weighted':
    weights = np.array(get_all_clusters_frequency(fresh=fresh).get(glycan, [100.0])) / 100
    weights = [1.0] * len(torsion_tables) if len(weights) != len(torsion_tables) else weights
    phi_variability = [circular_std(phi_values[:, i]) * np.mean(weights) for i in range(len(linkages))]
    psi_variability = [circular_std(psi_values[:, i]) * np.mean(weights) for i in range(len(linkages))]
    omega_variability = [circular_std(omega_values[:, i]) * np.mean(weights) for i in range(len(linkages))]
  else:
    phi_variability = [circular_std(phi_values[:, i]) for i in range(len(linkages))]
    psi_variability = [circular_std(psi_values[:, i]) for i in range(len(linkages))]
    omega_variability = [circular_std(omega_values[:, i]) for i in range(len(linkages))]
  if mode == 'amplify':
    phi_variability = [v**2 if not np.isnan(v) else np.nan for v in phi_variability]
    psi_variability = [v**2 if not np.isnan(v) else np.nan for v in psi_variability]
    omega_variability = [v**2 if not np.isnan(v) else np.nan for v in omega_variability]
  return pd.DataFrame({
    'linkage': linkages,
    'phi_variability': phi_variability,
    'psi_variability': psi_variability,
    'omega_variability': omega_variability
  })


@rescue_glycans
def calculate_torsion_flexibility_per_residue(glycan, mode = 'standard', stereo = None, my_path = None):
  """Calculates torsion angle flexibility for each monosaccharide based on its participating linkages.
  Args:
      glycan (str): IUPAC glycan sequence.
      mode (str): 'standard', 'amplify', or 'weighted' for calculation method.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      dict: Mapping of residue_number to torsion flexibility value.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  torsion_var = inter_structure_torsion_variability(glycan, stereo = stereo, mode = mode, my_path = my_path)
  if len(torsion_var) == 0:
    return {}
  residue_torsion_flex = {}
  for _, row in torsion_var.iterrows():
    linkage = row['linkage']
    residue_nums = [int(match.group()) for match in re.finditer(r'(\d+)(?=_)', linkage)]
    if len(residue_nums) >= 2:
      donor_res, acceptor_res = residue_nums[0], residue_nums[1]
      torsion_vals = [row['phi_variability'], row['psi_variability']]
      if pd.notna(row['omega_variability']):
        torsion_vals.append(row['omega_variability'])
      mean_torsion = np.nanmean(torsion_vals)
      if donor_res not in residue_torsion_flex:
        residue_torsion_flex[donor_res] = []
      if acceptor_res not in residue_torsion_flex:
        residue_torsion_flex[acceptor_res] = []
      residue_torsion_flex[donor_res].append(mean_torsion)
      residue_torsion_flex[acceptor_res].append(mean_torsion)
  return {res: np.nanmean(vals) if vals else np.nan for res, vals in residue_torsion_flex.items()}


@rescue_glycans
def make_correlation_matrix(glycan, stereo = None, my_path = None):
  """Computes a Pearson correlation matrix between residue positions across structures.
  Args:
      glycan (str or list): Glycan in IUPAC sequence or list of contact tables.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      pd.DataFrame: Correlation matrix showing relationships between residue positions.
  """
  if isinstance(glycan, str):
    dfs = get_contact_tables(glycan, stereo, my_path = my_path)
  elif isinstance(glycan, list):
    dfs = glycan
  if stereo is None and isinstance(glycan, str):
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  # Create an empty correlation matrix
  corr_sum = np.zeros((len(dfs[0]), len(dfs[0])))
  # Calculate the correlation matrix based on the distances
  for df in dfs:
    corr_sum += np.corrcoef(df.values, rowvar = False)
  return pd.DataFrame(corr_sum/len(dfs), columns = df.columns, index = df.columns)


@rescue_glycans
def inter_structure_frequency_table(glycan, stereo = None, threshold = 5, my_path = None):
  """Creates a table showing frequency of contacts between residues across structures.
  Args:
      glycan (str or list): Glycan in IUPAC sequence or list of contact tables.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      threshold (float): Maximum distance for determining a contact.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      pd.DataFrame: Table of contact frequencies across structures.
  """
  if stereo is None and isinstance(glycan, str):
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  if isinstance(glycan, str):
    dfs = get_contact_tables(glycan, stereo, my_path = my_path)
  elif isinstance(glycan, list):
    dfs = glycan
  # Apply thresholding and create a new list of transformed DataFrames
  binary_arrays = [df.values < threshold for df in dfs]
  # Sum up the transformed DataFrames to create the final DataFrame
  return pd.DataFrame(sum(binary_arrays), columns = dfs[0].columns, index = dfs[0].columns)


def extract_binary_interactions_from_PDB(coordinates_df):
  """Extracts binary interactions between C1/C2 atoms and oxygen atoms from coordinates.
  Args:
      coordinates_df (pd.DataFrame): Coordinate dataframe from extract_3D_coordinates.
  Returns:
      pd.DataFrame or list of pd.DataFrame: DataFrame with columns 'Atom', 'Column', and 'Value'
      showing interactions. Returns a list of DataFrames if multiple chains are present.
  """
  # Check if multiple chains exist
  unique_chains = coordinates_df['chain_id'].unique()
  if len(unique_chains) > 1:
    results = []
    for chain in unique_chains:
      chain_df = coordinates_df[coordinates_df['chain_id'] == chain]
      chain_result = process_interactions(chain_df)
      if not chain_result.empty:
        results.append(chain_result)
    return results
  else:
    return process_interactions(coordinates_df)


def process_interactions(coordinates_df):
  """Extracts binary interactions between C1/C2 atoms and oxygen atoms from coordinates.
  Args:
      coordinates_df (pd.DataFrame): Coordinate dataframe from extract_3D_coordinates.
  Returns:
      pd.DataFrame: DataFrame with columns 'Atom', 'Column', and 'Value' showing interactions.
  """
  # First check if we only have one monosaccharide
  unique_residues = coordinates_df['residue_number'].nunique()
  carbon_mask = (((~coordinates_df['monosaccharide'].str.contains(C2_PATTERN, na = False)) & (coordinates_df['atom_name'] == 'C1')) |
                 ((coordinates_df['monosaccharide'].str.contains(C2_PATTERN, na = False)) & (coordinates_df['atom_name'] == 'C2')))
  oxygen_mask = coordinates_df['atom_name'].isin({'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O8', 'O9', 'S1'})
  carbons = coordinates_df[carbon_mask]
  oxygens = coordinates_df[oxygen_mask]
  c_coords = carbons[['x', 'y', 'z']].values
  o_coords = oxygens[['x', 'y', 'z']].values
  c_residues = carbons['residue_number'].values
  o_residues = oxygens['residue_number'].values
  c_labels = [f"{r}_{m}_{a}" for r, m, a in zip(carbons['residue_number'], carbons['monosaccharide'], carbons['atom_name'])]
  o_labels = [f"{r}_{m}_{a}" for r, m, a in zip(oxygens['residue_number'], oxygens['monosaccharide'], oxygens['atom_name'])]
  interactions = []
  if unique_residues == 2 and "ROH" in coordinates_df.monosaccharide.tolist():
    roh_oxygens = oxygens[oxygens['monosaccharide'] == 'ROH']
    if not roh_oxygens.empty:
      roh_coord = roh_oxygens[['x', 'y', 'z']].values[0]
      # Find C1 of the monosaccharide and O1 of ROH
      for i, c_label in enumerate(c_labels):
        if carbons.iloc[i]['monosaccharide'] != 'ROH' and carbons.iloc[i]['atom_name'] in {'C1', 'C2'}:
          distance = np.abs(roh_coord - c_coords[i]).sum()
          interactions.append({
            'Atom': c_label,
            'Column': f"{roh_oxygens['residue_number'].iloc[0]}_ROH_O1",
            'Value': distance
            })
  else:
    for i, c_label in enumerate(c_labels):
      mask = (o_residues < c_residues[i])
      if np.any(mask):
        relevant_o_coords = o_coords[mask]
        distances = np.abs(relevant_o_coords - c_coords[i]).sum(axis = 1)
        if len(distances) > 0:
          min_dist_idx = np.argmin(distances)
          min_dist = distances[min_dist_idx]
          min_idx = np.where(mask)[0][min_dist_idx]
          interactions.append({
            'Atom': c_label,
            'Column': o_labels[min_idx],
            'Value': min_dist
          })
  df =  pd.DataFrame(interactions)
  if len(df) > 0:
    # Extract source and target monosaccharides
    df['source_mono'] = df['Atom'].str.split('_').str[:2].str.join('_')
    df['target_mono'] = df['Column'].str.split('_').str[:2].str.join('_')
    # Group by monosaccharide pairs and keep minimum distance
    df = df.loc[df.groupby(['source_mono', 'target_mono'])['Value'].idxmin()]
  return df[['Atom', 'Column', 'Value']].reset_index(drop = True) if len(df) > 0 else df


def create_mapping_dict_and_interactions(df, valid_fragments, n_glycan, furanose_end, d_end, is_protein_complex):
  """Creates mapping dictionaries for converting PDB residue names to IUPAC notation.
  Args:
      df (pd.DataFrame): Interaction dataframe from extract_binary_interactions_from_PDB.
      valid_fragments (set): Valid monosaccharide link fragments from glycowork.
      n_glycan (bool): If True, applies N-glycan-specific corrections.
      furanose_end (bool): If True, considers furanose forms for terminal residues.
      d_end (bool): If True, considers D-form for terminal residues.
      is_protein_complex (bool): If True, assumes glycan comes from protein-glycan PDB
  Returns:
      tuple: (mapping_dict, interaction_dict) for PDB to IUPAC conversion.
  """
  special_cases = {
            'Man(a1-4)', '-R', 'GlcNAc(a1-1)', 'GlcNAc(b1-1)', 'GalNAc(a1-1)', 'D-All(a1-1)', 'D-All(b1-1)'
            'GalNAc(b1-1)', 'Glc(a1-1)', 'Glc(b1-1)', 'Rha(a1-1)', 'Rha(b1-1)', "Glc6S(a1-1)",
            'Neu5Ac(a2-1)', 'Neu5Ac(b2-1)', 'Neu5Ac(a1-1)', 'Man(a1-1)', 'Man(b1-1)', 'Gal(a1-1)',
            'Gal(b1-1)', 'Fuc(a1-1)', 'Fuc(b1-1)', 'Xyl(a1-1)', 'Xyl(b1-1)', 'L-Gul(a1-1)',  'L-Gul(b1-1)',
            'GlcA(a1-1)', 'GlcA(b1-1)', 'GlcNS(a1-1)', 'GlcNS(b1-1)', 'GlcNAc6S(a1-1)', 'Alt(a1-1)', 'Alt(b1-1)',
            'GlcNAc6S(b1-1)', 'GlcNS6S(a1-1)', 'GlcNS6S(b1-1)', 'GlcNS3S6S(a1-1)', 'IdoA(a1-1)', 'IdoA(b1-1)',
            'GlcNS3S6S(b1-1)', '2-4-diacetimido-2-4-6-trideoxyhexose(a1-1)', 'D-Araf(a1-1)', 'D-Araf(b1-1)',
            'GlcA2S(a1-1)', 'GlcA2S(b1-1)', 'Ara(a1-1)', 'Ara(b1-1)', 'Araf(a1-1)', 'Araf(b1-1)', 'Fru(a2-1)', "Tag(a2-1)", "Tag(b2-1)",
            'Fru(b2-1)', 'Fruf(a2-1)', 'Fruf(b2-1)', 'ManNAc(a1-1)', 'ManNAc(b1-1)', "GalNAc6S(a1-1)", "GalA(a1-1)", "GalA(b1-1)",
            'Neu5Gc(a2-1)', 'Neu5Gc(b2-1)', 'Kdn(a2-1)', 'Kdn(b2-1)', 'Sor(a2-1)', 'Sor(b2-1)', 'Qui(a1-1)', 'Qui(b1-1)', 'L-Gul(a1-1)', 'L-Gul(b1-1)',
            'L-Tal(a1-1)', 'L-Tal(b1-1)', 'Lyx(a1-1)', 'Lyx(b1-1)', 'Ribf(a1-1)', 'Ribf(b1-1)', 'Psif(a1-1)', 'Psif(b1-1)'
        }

  def d_conversion(mono, trigger, i = 1):
    if mono.startswith(trigger):
      d_version = f"D-{mapped_to_check}"
      if d_version in valid_fragments or (i == 0 and d_end):
        return d_version
    return mono

  donors = {row['Atom'].rsplit('_', 1)[0] for _, row in df.iterrows()}
  acceptors = {row['Column'].rsplit('_', 1)[0] for _, row in df.iterrows()}
  reducing_end = (acceptors - donors).pop() if (acceptors - donors) else '1_ROH'
  reducing_res = reducing_end.split('_')[1]
  if reducing_res in map_dict and reducing_res != 'ROH':
    mapping_dict = {reducing_end: map_dict[reducing_res].split('(')[0]}
  else:
    mapping_dict = {reducing_end: '-R'}
  interaction_dict, interaction_dict2 = {}, {}
  wrong_mannose, individual_entities = [], []
  furanose_map = {'Fru': 'Fruf', 'Gal': 'Galf', 'Ara': 'Araf', 'D-Ara': 'D-Araf'}
  for i, row in df.iterrows():
    first_mono = row['Atom']
    second_mono = row['Column']
    mono = first_mono.rsplit('_', 1)[0]
    second_mono_base = second_mono.rsplit('_', 1)[0]
    first_val = ''.join(re.findall(r'\d+', first_mono.split('_')[-1]))
    last_val = ''.join(re.findall(r'\d+', second_mono.split('_')[-1]))
    individual_entities.extend([mono, second_mono_base])
    individual_entities = list(dict.fromkeys(individual_entities))
    # Handle special MAN case for n_glycan
    if mono.split('_')[1] + f'({first_val}-{last_val})' == "MAN(1-4)" and n_glycan:
      wrong_mannose.append(mono)
    for m in [mono, second_mono_base]:
      if m in wrong_mannose:
        m = f"{m.split('_')[0]}_BMA"
    mapped_to_check = f"{map_dict[mono.split('_')[1]]}{first_val}-{last_val})"
    mapped_to_check = d_conversion(mapped_to_check, 'Ara', i = i)
    mono_type = mapped_to_check.split('(')[0]
    if i == 0 and is_protein_complex:
     mapped_to_check2 = f"{map_dict[second_mono_base.split('_')[1]].split('(')[0]}"
     if mapped_to_check2 in valid_fragments:
       mapping_dict[second_mono_base] = f"{map_dict[second_mono_base.split('_')[1]]}1-1)"
     elif f"{mapped_to_check2}f" in valid_fragments:
       mapping_dict[second_mono_base] = f"{furanose_map[mapped_to_check2]}{map_dict[second_mono_base.split('_')[1]][len(mapped_to_check2):]}1-1)"
    if (mapped_to_check not in valid_fragments and (mapped_to_check not in special_cases or furanose_end) and mono_type in furanose_map):
      mapped_to_check = furanose_map[mono_type] + mapped_to_check[len(mono_type):]
      mapped_to_check = d_conversion(mapped_to_check, 'Araf')
    if (mapped_to_check in valid_fragments) or (mapped_to_check in special_cases):
      mapped_to_use =  'Man(b1-4)' if (mapped_to_check == 'Man(a1-4)' and n_glycan) else mapped_to_check
      mapping_dict[mono] = mapped_to_use
      mono_key = f"{mono.split('_')[0]}_({mapped_to_use.split('(')[1]}"
      if mono in interaction_dict:
        if second_mono_base not in interaction_dict[mono]:
          interaction_dict[mono].append(second_mono_base)
          interaction_dict2[mono] = [mono_key]
          interaction_dict2[mono_key] = [second_mono_base]
      else:
        interaction_dict[mono] = [second_mono_base]
        interaction_dict2[mono] = [mono_key]
        if mono_key in interaction_dict2:
          interaction_dict2[mono_key].append(second_mono_base)
        else:
          interaction_dict2[mono_key] = [second_mono_base]
  return mapping_dict, interaction_dict2


def extract_binary_glycontact_interactions(interaction_dict, mapping_dict):
  """Transforms PDB-based interactions into IUPAC binary interactions.
  Args:
      interaction_dict (dict): Dict of interactions from create_mapping_dict_and_interactions.
      mapping_dict (dict): Mapping dict from create_mapping_dict_and_interactions.
  Returns:
      list: List of binary interaction tuples in IUPAC format.
  """
  result = []
  for k, v in interaction_dict.items():
    new_k = k.split('_')[1].replace('(', '').replace(')', '') if '(' in k else mapping_dict.get(k, '').split('(')[0]
    new_v = v[0].split('_')[1].replace('(', '').replace(')', '') if '(' in v[0] else mapping_dict.get(v[0], '').split('(')[0]
    result.append((new_k, new_v))
  return result


def extract_binary_glycowork_interactions(graph_output):
  """Extracts binary interactions from glycowork graph output.
  Args:
      graph_output (tuple): Output from glycan_to_graph function.
  Returns:
      list: List of binary interaction pairs.
  """
  mask_dic, adj_matrix = graph_output
  n = len(mask_dic)
  return [(mask_dic[k], mask_dic[j]) for k in range(n) for j in range(k + 1, n) if adj_matrix[k, j] == 1]


def glycowork_vs_glycontact_interactions(glycowork_interactions, glycontact_interactions) :
  """Compares binary interactions from glycowork and glycontact for validation.
  Args:
      glycowork_interactions (list): Interactions from glycowork.
      glycontact_interactions (list): Interactions from glycontact.
  Returns:
      bool: True if interactions are consistent (excluding special cases).
  """
  ignore_pairs = {
        ('GlcNAc', 'a1-1'), ('a1-1', '-R'), ('a2-1', '-R'), ('b2-1', '-R'),
        ('GlcNAc', 'b1-1'), ('b1-1', '-R'), ('GalNAc', 'a1-1'), ('GalNAc', 'b1-1'),
        ('Glc', 'a1-1'), ('Glc', 'b1-1'), ('Rha', 'b1-1'), ('Rha', 'a1-1'),
        ('Neu5Ac', 'b2-1'), ('Neu5Ac', 'a2-1'), ('Neu5Ac', 'a1-1'), ('Man', 'b1-1'), ('Man', 'a1-1'),
        ('Gal', 'b1-1'), ('Gal', 'a1-1'), ('Fuc', 'b1-1'), ('Fuc', 'a1-1'), ('D-All', 'a1-1'), ('D-All', 'b1-1'),
        ('Xyl', 'b1-1'), ('Xyl', 'a1-1'), ('GlcA', 'a1-1'), ('GlcA', 'b1-1'), ("Glc6S", "a1-1"),
        ('GlcNS', 'a1-1'), ('GlcNS', 'b1-1'), ('GlcNAc6S', 'a1-1'), ('b1-4', ''), ('Alt', 'a1-1'), ('Alt', 'b1-1'),
        ('GlcNAc6S', 'b1-1'), ('GlcNS6S', 'a1-1'), ('GlcNS6S', 'b1-1'), ("GalNAc6S", "a1-1"), ("Tag", "a2-1"), ("Tag", "b2-1"),
        ('GlcNS3S6S', 'a1-1'), ('GlcNS3S6S', 'b1-1'), ('L-Gul', 'a1-1'), ('L-Gul', 'b1-1'), ('IdoA', 'a1-1'), ('IdoA', 'b1-1'),
        ('2-4-diacetimido-2-4-6-trideoxyhexose', 'a1-1'), ('GlcA2S', 'a1-1'), ('D-Araf', 'a1-1'), ('D-Araf', 'b1-1'),
        ('GlcA2S', 'b1-1'), ('Ara', 'a1-1'), ('Ara', 'b1-1'), ('Araf', 'a1-1'), ('Araf', 'b1-1'), ('Fru', 'a2-1'),
        ('Fru', 'b2-1'), ('ManNAc', 'a1-1'), ('ManNAc', 'b1-1'), ('Fruf', 'a2-1'), ('Fruf', 'b2-1'), ('GalA', 'a1-1'), ('GalA', 'b1-1'),
        ('Neu5Gc', 'a2-1'), ('Neu5Gc', 'b2-1'), ('Kdn', 'a2-1'), ('Kdn', 'b2-1'), ('Sor', 'a2-1'), ('Sor', 'b2-1'), ('Qui', 'a1-1'), ('Qui', 'b1-1'),
        ('Lyx', 'a1-1'), ('Lyx', 'b1-1'), ('Psif', 'a1-1'), ('Psif', 'b1-1'), ('Ribf', 'a1-1'), ('Ribf', 'b1-1'), ('L-Gul', 'a1-1'), ('L-Gul', 'b1-1'),
        ('L-Tal', 'a1-1'), ('L-Tal', 'b1-1')
    }
  differences = set(glycontact_interactions) ^ set(glycowork_interactions)
  filtered_differences = [pair for pair in differences if pair not in ignore_pairs]
  return (not filtered_differences and len(glycontact_interactions) >= len(glycowork_interactions))


def check_reconstructed_interactions(interaction_dict) :
  """Verifies if the reconstructed glycan is connected as a single component.
  Args:
      interaction_dict (dict): Dictionary of interactions.
  Returns:
      bool: True if glycan is correctly reconstructed as a single connected component.
  """
  G = nx.Graph()
  # Add nodes and edges from dictionary interactions
  G.add_edges_from((node, neighbor) for node, neighbors in interaction_dict.items() for neighbor in neighbors)
  return nx.is_connected(G)


def annotate_pdb_data(pdb_dataframe, mapping_dict) :
  """Annotates PDB data with IUPAC nomenclature using the mapping dictionary.
  Args:
      pdb_dataframe (pd.DataFrame): DataFrame with PDB coordinates.
      mapping_dict (dict): Mapping from PDB to IUPAC nomenclature.
  Returns:
      pd.DataFrame: Annotated dataframe with IUPAC column.
  """
  m_dict = copy.deepcopy(mapping_dict)
  pdb_dataframe = pdb_dataframe.copy()
  for m, v in m_dict.items():
    if "BMA" in m:
      mapping_dict[f"{m.split('_')[0]}_MAN"] = v #restore the corrected mannose into a wrong one for annotation
  pdb_dataframe['lookup_key'] = pdb_dataframe['residue_number'].astype(str) + '_' + pdb_dataframe['monosaccharide']
  # Map values using the dictionary, falling back to original monosaccharide
  pdb_dataframe['IUPAC'] = pdb_dataframe['lookup_key'].map(mapping_dict).fillna(pdb_dataframe['monosaccharide'])
  # Drop temporary column
  pdb_dataframe.drop('lookup_key', axis = 1, inplace = True)
  return pdb_dataframe


def correct_dataframe(df):
  """Corrects monosaccharide assignments in the dataframe based on atom counts.
  Args:
      df (pd.DataFrame): Annotated dataframe from annotate_pdb_data.
  Returns:
      pd.DataFrame: Corrected dataframe with fixed monosaccharide assignments.
  """
  use_chain = len(df['chain_id'].unique()) > 1
  group_cols = ['chain_id', 'residue_number'] if use_chain else ['residue_number']
  c_counts = df[df['element'] == 'C'].groupby(group_cols).size()
  high_carbon_residues = c_counts[c_counts >= 7].index
  # Function to check if a residue is in high carbon list
  def is_high_carbon(row):
    if use_chain:
      return (row['chain_id'], row['residue_number']) in high_carbon_residues
    return row['residue_number'] in high_carbon_residues
  # Apply replacements
  df.loc[(df['monosaccharide'] == 'GLC') & df.apply(is_high_carbon, axis = 1), 'monosaccharide'] = 'NGA'
  df.loc[(df['monosaccharide'] == 'BGC') & df.apply(is_high_carbon, axis = 1), 'monosaccharide'] = 'A2G'
  return df


def process_interactions_result(res, threshold, valid_fragments, n_glycan, furanose_end, d_end, is_protein_complex, glycan, df):
  """Process a single interaction result and return the annotation if valid."""
  if len(res) < 1:
    return pd.DataFrame(), {}
  if isinstance(threshold, float) or isinstance(threshold, int):
    res = res[res.Value < threshold].reset_index(drop = True)
  else:
    for thresh in sorted(threshold):
      res = res[res.Value < thresh].reset_index(drop = True)
      if len(res) > 0:
        break
  mapping_dict, interaction_dict = create_mapping_dict_and_interactions(res, valid_fragments,
                                                                      n_glycan, furanose_end, d_end, is_protein_complex)
  # Validate against glycowork
  glycowork_interactions = extract_binary_glycowork_interactions(glycan_to_graph(glycan))
  glycontact_interactions = extract_binary_glycontact_interactions(interaction_dict, mapping_dict)
  glycontact_interactions = [(x + 'f' if any(f'{x}f(' in s for s in valid_fragments) and not any(f'{x}(' in s for s in valid_fragments) else x,
                            y + 'f' if any(f'{y}f(' in s for s in valid_fragments) and not any(f'{y}(' in s for s in valid_fragments) else y)
                           for x, y in glycontact_interactions]
  if (glycowork_vs_glycontact_interactions(glycowork_interactions, glycontact_interactions) and
      check_reconstructed_interactions(interaction_dict)):
    annotated = annotate_pdb_data(df, mapping_dict)
    return annotated[~annotated['monosaccharide'].isin(NON_MONO)].reset_index(drop = True), interaction_dict
  return pd.DataFrame(), {}


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
  sequences = []
  visited = set()
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
  return list(set(sequences))


def get_annotation(glycan, pdb_file, threshold = 3.5):
  """Annotates a PDB file with IUPAC nomenclature for a given glycan.
  Args:
      glycan (str): IUPAC glycan sequence.
      pdb_file (str): Path to PDB file.
      threshold (float or list): Distance threshold for interactions.
  Returns:
      tuple: (annotated_dataframe, interaction_dict) or (empty_dataframe, {}) if validation fails.
  """
  if isinstance(pdb_file, tuple):
    return pdb_file
  CUSTOM_PDB = {
        "NAG6SO3": "GlcNAc6S", "NDG6SO3": "GlcNAc6S", "NDG3SO3": "GlcNAc3S6S", "IDR2SO32SO3": "IdoA2S",
        "NGA4SO3": "GalNAc4S", "IDR2SO3": "IdoA2S", "BDP3SO3": "GlcA3S", "TOA2SO3": "GalA2S",
        "BDP2SO3": "GlcA2S", "SIA9ACX": "Neu5Ac9Ac", "MAN3MEX": "Man3Me", "GLC6SO3": "Glc6S",
        "SIA9MEX": "Neu5Ac9Me", "NGC9MEX": "Neu5Gc9Me", "BDP4MEX": "GlcA4Me", "UYS6SO36SO3": "GlcNS6S",
        "GAL6SO3": "Gal6S", "NAG6PCX": "GlcNAc6PCho", "UYS6SO3": "GlcNS6S", "A2G6SO3": "GalNAc6S",
        "4YS6SO3": "GlcNS6S", "6YS6SO3": "GlcNS6S", "GCU2SO3": "GlcA2S", "GAL3SO3": "Gal3S", "GAL4SO3": "Gal4S",
        'VYS3SO3': 'GlcNS3S6S', 'VYS6SO3': 'GlcNS3S6S', 'FUC2MEX': 'Fuc2Me', 'FUC3MEX': 'Fuc3Me', 'FUC4MEX': 'Fuc4Me',
        "QYS3SO3": "GlcNS3S6S", "QYS6SO3": "GlcNS3S6S", "RAM2MEX": "Rha2Me", "RAM3MEX": "Rha3Me"
    }
  n_glycan = 'Man(b1-4)GlcNAc(b1-4)' in glycan or 'Man(b1-4)[Fuc(a1-3)]GlcNAc' in glycan
  furanose_end = glycan.endswith('f')
  d_end = glycan[glycan.rfind('-')-1] == "D"
  df = correct_dataframe(extract_3D_coordinates(pdb_file))
  unique_residues = set(df.monosaccharide.unique())
  if len(df) < 1:
    return pd.DataFrame(), {}
  is_protein_complex = df['record_name'].iloc[0] == 'HETATM'
  # Handle multiple instances of a single monosaccharide in protein complexes
  if is_protein_complex and ')' not in glycan:
    results = []
    for (res_num, chain) in df[['residue_number', 'chain_id']].drop_duplicates().values:
      instance_df = df[(df['residue_number'] == res_num) & (df['chain_id'] == chain)].copy()
      mono_type = instance_df['monosaccharide'].iloc[0]
      if glycan == map_dict[mono_type].split('(')[0]:
        # Create instance-specific mapping dictionary using the global map_dict
        instance_mapping = {f"{res_num}_{mono_type}": map_dict.get(mono_type, mono_type)}
        results.append((annotate_pdb_data(instance_df, instance_mapping), {}))
    return list(zip(*results)) if results else (pd.DataFrame(), {})
  if any(mm in unique_residues for mm in NON_MONO):
    # Process modified glycans
    to_modify_dict = {}
    dist_table = make_atom_contact_table(df)
    # Get residue mapping
    resdict = df.groupby('residue_number')['monosaccharide'].first().to_dict()
    # Process non-monosaccharide elements
    for key, val in resdict.items():
      if val in NON_MONO:
        element = f"{key}_{val}"
        contact_table = dist_table.filter(regex = element)
        # Filter contact table
        mask = ~contact_table.index.str.contains('|'.join(contact_table.columns))
        filtered_table = contact_table.loc[mask]
        filtered_table = filtered_table[~filtered_table.index.str.split('_').str[2].str.contains('H')]
        # Find closest partner
        if not filtered_table.empty:
          partners = filtered_table[filtered_table != 0].stack().idxmin()
          sugar_partner = partners[0]
          sugar_resnum, sugar, atom, _ = sugar_partner.split("_")
          link_pos = re.findall(r'\d+', atom)[0]
          modified_mono = sugar + link_pos + val
          if modified_mono in CUSTOM_PDB:
            to_modify_dict[int(sugar_resnum)] = modified_mono
            to_modify_dict[key] = [modified_mono, sugar_resnum]
    # Apply modifications to dataframe
    grouped_mods = {}
    for residue_num, modification in to_modify_dict.items():
      target_residue = modification[1] if isinstance(modification, list) else residue_num
      if target_residue not in grouped_mods:
        grouped_mods[target_residue] = []
      grouped_mods[target_residue].append(modification)
    for target_residue, mods in grouped_mods.items():
      if len(mods) > 1:
        # Multiple modifications for same residue - combine them
        base = mods[0][0] if isinstance(mods[0], list) else mods[0]  # Get base from first mod
        base = base[:3]  # Extract just the base part (e.g., 'FUC')
        mod_parts = [mod[0][3:] if isinstance(mod, list) else mod[3:] for mod in mods]  # Get just the modification part (e.g., '2MEX')
        combined_mod = base + ''.join(sorted(mod_parts))  # Combine with sorted modifications
        df.loc[df['residue_number'] == int(target_residue), 'monosaccharide'] = combined_mod
      else:
        # Single modification - use original logic
        mod = mods[0]
        if isinstance(mod, str):
          df.loc[df['residue_number'] == target_residue, 'monosaccharide'] = mod
        else:
          monosaccharide, new_residue = mod
          mask = df['residue_number'] == target_residue
          df.loc[mask, 'monosaccharide'] = monosaccharide
          df.loc[mask, 'residue_number'] = int(new_residue)
    df = df.sort_values('residue_number')
  # Extract and validate linkages
  disaccharides = [di for di in get_k_saccharides([glycan], just_motifs = True)[0] if '?' not in di] if '(' in glycan else []
  valid_fragments = {f"{x.split(')')[0]})" for x in disaccharides} | ({min_process_glycans([glycan])[0][-1]} if is_protein_complex else set())
  res = extract_binary_interactions_from_PDB(df)
  # Handle case where extract_binary_interactions_from_PDB returns a list of DataFrames (multiple chains)
  if isinstance(res, list):
    chain_ids = df.chain_id.unique()
    expected_residue_count = glycan.count('(') + 1
    # Try each chain's result and use the first one that successfully validates
    for i, chain_res in enumerate(res):
      if not chain_res.empty:
        max_residue = max(
          max([int(atom.split('_')[0]) for atom in chain_res['Atom']]),
          max([int(col.split('_')[0]) for col in chain_res['Column']])
        )
        if max_residue != expected_residue_count + 1:
          continue
      result = process_interactions_result(chain_res, threshold, valid_fragments,
                                         n_glycan, furanose_end, d_end, is_protein_complex, glycan, df[df.chain_id == chain_ids[i]])
      result = process_interactions_result(chain_res, threshold, valid_fragments,
                                         n_glycan, furanose_end, d_end, is_protein_complex, glycan, df[df.chain_id == chain_ids[i]])
      if len(result[0]) > 0:
        if len(result[1]) > 0:
          result[1]['__pdb_path__'] = pdb_file
        return result
    # If no chain validates successfully
    return pd.DataFrame(), {}
  else:
    # Original single-chain behavior
    result = process_interactions_result(res, threshold, valid_fragments,
                                     n_glycan, furanose_end, d_end, is_protein_complex, glycan, df)
    if len(result[1]) > 0:
      result[1]['__pdb_path__'] = pdb_file
    return result


@rescue_glycans
def annotation_pipeline(glycan, pdb_file = None, threshold = 3.5, stereo = None, my_path = None) :
  """Combines all annotation steps to convert PDB files to IUPAC annotations.
  Args:
      glycan (str): IUPAC glycan sequence.
      pdb_file (str or list, optional): Path(s) to PDB file(s).
      threshold (float): Distance threshold for interactions.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      my_path (Path, optional): Custom path to PDB folder
  Returns:
      tuple: (dataframes_list, interaction_dicts_list) for all processed PDBs.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  if pdb_file is None:
    pdb_file = fetch_pdbs(glycan, stereo = stereo, my_path = my_path)
  if not isinstance(pdb_file, list):
    pdb_file = [pdb_file]
  dfs, int_dicts = zip(*[get_annotation(glycan, pdb, threshold = threshold) for pdb in pdb_file])
  return dfs, int_dicts


def get_example_pdb(glycan, stereo = None, rng = None, my_path = None):
  """Gets a random example PDB file for a given glycan.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      rng (Random, optional): Random number generator instance.
      my_path (Path, optional): Custom path to pdb folder
  Returns:
      Path: Path to a randomly selected PDB file.
  """
  if rng is None:
    rng = Random(42)
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  matching_pdbs = fetch_pdbs(glycan, stereo = stereo, my_path = my_path)
  cluster_frequencies = get_all_clusters_frequency().get(glycan, [100.0])
  weights = cluster_frequencies if len(cluster_frequencies) == len(matching_pdbs) else None
  return rng.choices(matching_pdbs, weights = weights)[0]


def monosaccharide_preference_structure(df, monosaccharide, threshold, mode = 'default'):
  """Finds preferred partners for a given monosaccharide.
  Args:
      df (pd.DataFrame): Monosaccharide distance table.
      monosaccharide (str): Target monosaccharide type.
      threshold (float): Minimum distance to exclude covalent bonds.
      mode (str): 'default', 'monolink', or 'monosaccharide' for different reporting formats.
  Returns:
      dict: Dictionary of preferred partners for the target monosaccharide.
  """
  entities = df.columns.tolist()
  preferred_partners = {}
  if '(' not in monosaccharide:
    mono_mask = [e.split('_')[1].split('(')[0] == monosaccharide for e in entities]
  else:
    mono_mask = [e.split('_')[1] == monosaccharide for e in entities]
  target_entities = [e for e, m in zip(entities, mono_mask) if m]
  for entity in target_entities:
    distances = df[entity]
    valid_distances = distances[(distances != 0) & (distances >= threshold)]
    if not valid_distances.empty:
      closest_partner = valid_distances.idxmin()
      preferred_partners[entity] = closest_partner
  if mode == 'default':
    return preferred_partners
  elif mode == 'monolink':
    return {k: v.split('_')[1] for k, v in preferred_partners.items()}
  else:  # mode == 'monosaccharide'
    return {k: v.split('_')[1].split('(')[0] for k, v in preferred_partners.items()}


@rescue_glycans
def multi_glycan_monosaccharide_preference_structure(glycan, monosaccharide, stereo = None, threshold = 3.5, mode = 'default'):
  """Visualizes monosaccharide partner preferences across multiple structures.
  Args:
      glycan (str): IUPAC glycan sequence.
      monosaccharide (str): Target monosaccharide type.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      threshold (float): Minimum distance to exclude covalent bonds.
      mode (str): 'default', 'monolink', or 'monosaccharide' for different reporting formats.
  Returns:
      None: Displays a bar plot of partner frequencies.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  mono_tables = get_contact_tables(glycan, stereo = stereo)
  dict_list = [monosaccharide_preference_structure(dist, monosaccharide, threshold, mode) for dist in mono_tables]
  all_values = [v for d in dict_list for v in d.values()]
  if not all_values:
    return
  value_counts = Counter(all_values)
  plt.bar(value_counts.keys(), value_counts.values())
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.title(f'Frequency for {monosaccharide} above {threshold} across structures')
  plt.tight_layout()
  plt.show()


def get_all_clusters_frequency(fresh = False):
  """Extracts frequency data for all glycan clusters from GlycoShape.
  Args:
      fresh (bool): If True, fetches fresh data from GlycoShape.
  Returns:
      dict: Dictionary mapping IUPAC sequences to cluster frequency lists.
  """
  data = {}
  if fresh:
    response = requests.get("https://glycoshape.org/database/GLYCOSHAPE.json")
    if response.status_code == 200:
      data = response.json()
  else:
    data = glycoshape_mirror
  return {value["iupac"]: [100.0] if list(value["clusters"].values()) == ['None'] else list(value["clusters"].values()) for key, value in data.items()}


def glycan_cluster_pattern(threshold = 70, mute = False, fresh = False) :
  """Categorizes glycans based on their cluster distribution patterns.
  Args:
      threshold (float): Percentage threshold for major cluster classification.
      mute (bool): If True, suppresses print output.
      fresh (bool): If True, fetches fresh data from GlycoShape.
  Returns:
      tuple: (major_clusters_list, minor_clusters_list) sorted by cluster pattern.
  """
  all_frequencies = get_all_clusters_frequency(fresh = fresh)
  major_clusters, minor_clusters = [], []
  for glycan, freqs in all_frequencies.items():
    try:
      if float(freqs[0]) >= threshold:
        major_clusters.append(glycan)
      else:
        minor_clusters.append(glycan)
    except (IndexError, ValueError):
      continue
  if not mute:
    print(f"Number of glycans with one major cluster: {len(major_clusters)}")
    print(f"Number of glycans without a major cluster: {len(minor_clusters)}")
  return major_clusters, minor_clusters


def get_sasa_table(glycan, stereo = None, my_path = None, fresh = False):
  """Calculates solvent accessible surface area (SASA) for each monosaccharide.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      my_path (str, optional): Custom path to PDB folders.
      fresh (bool): If True, fetches fresh cluster frequencies.
  Returns:
      pd.DataFrame: Table with SASA values and statistics for each monosaccharide.
  """
  is_single_pdb = my_path is not None and isinstance(my_path, str) and "." in my_path
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  if my_path is None:
    pdb_files = fetch_pdbs(glycan, stereo = stereo)
  else:
    pdb_files = sorted(str(p) for p in Path(f"{my_path}{glycan}").glob(f"*{stereo}*")) if not is_single_pdb else [my_path]
  df = pd.DataFrame()
  for pdb_file in pdb_files:
    df, _ = get_annotation(glycan, pdb_file)
    if len(df) > 0:
      break
  if len(df) < 1:
    return pd.DataFrame(columns = ['Monosaccharide_id', 'Monosaccharide', 'SASA', 'Standard Deviation', 'Coefficient of Variation'])
  if not is_single_pdb:
    weights = np.array(get_all_clusters_frequency(fresh=fresh).get(glycan, [100.0])) / 100
    weights = np.tile(weights, 2) if len(weights) != len(pdb_files) else weights
    weights = [1.0] * len(pdb_files) if len(weights) != len(pdb_files) else weights
  else:
    weights = [1.0]
  residue_modifications = df.set_index('residue_number')['IUPAC'].to_dict()
  # Process each PDB file
  sasa_values = {}
  for p, pdb_file in enumerate(pdb_files):
    if not isinstance(pdb_file, tuple):
      structure = md.load(pdb_file)
    else:  # Create temporary file in a way that doesn't keep the file handle open
      fd, temp_path = tempfile.mkstemp(suffix = '.pdb')
      try:
        with os.fdopen(fd, 'w') as tmp:
          tmp.write(df_to_pdb_content(pdb_file[0]))
        structure = md.load(temp_path)  # Load the file after closing it
      finally:
        os.unlink(temp_path)  # Always clean up the temporary file
    if is_single_pdb:
      glycan_residues = set(df['residue_number'])
      glycan_chains = set(df['chain_id']) if 'chain_id' in df.columns else {None}
      coords = structure.xyz[0]
      glycan_coords = []
      glycan_atom_original_indices = []
      for atom in structure.topology.atoms:
        res = atom.residue
        chain_match = atom.residue.chain.chain_id in glycan_chains if glycan_chains != {None} else True
        if chain_match and (res.resSeq in glycan_residues or (res.name in NON_MONO and any(r.resSeq in glycan_residues for r in structure.topology.residues if r.chain == res.chain))):
          glycan_coords.append(coords[atom.index])
          glycan_atom_original_indices.append(atom.index)
      glycan_coords = np.array(glycan_coords)
      keep_atom_indices = []
      glycan_atom_indices = set()
      cutoff_distance = 1.0
      for atom in structure.topology.atoms:
        res = atom.residue
        if res.is_water or res.name in {'HOH', 'WAT', 'SOL', 'NA', 'CL', 'K', 'MG', 'CA', 'ZN'}:
          continue
        if atom.element.symbol == 'H':
          continue
        atom_coord = coords[atom.index]
        if atom.index in glycan_atom_original_indices:
          keep_atom_indices.append(atom.index)
          glycan_atom_indices.add(len(keep_atom_indices) - 1)
        else:
          distances = np.linalg.norm(glycan_coords - atom_coord, axis = 1)
          if np.min(distances) <= cutoff_distance:
            is_duplicate = False
            for existing_idx in keep_atom_indices:
              if np.linalg.norm(coords[existing_idx] - atom_coord) < 0.0001:
                is_duplicate = True
                break
            if not is_duplicate:
              keep_atom_indices.append(atom.index)
      structure = structure.atom_slice(keep_atom_indices)
    sasa = md.shrake_rupley(structure, mode = 'atom')
    # Group SASA by residue
    mono_sasa, modification_to_parent = {}, {}
    # First pass: identify modification groups and their parent residues
    parent_resSeq = None
    for res in structure.topology.residues:
      if res.name == 'PCX':
        # Special case: find the residue with PCho in residue_modifications
        for resSeq, resName in residue_modifications.items():
          if 'PCho' in resName:
            modification_to_parent[res.resSeq] = resSeq
            break
      elif res.name not in NON_MONO:
        parent_resSeq = res.resSeq
      else:
        # If this is a modification, assign it to the last seen non-modification residue
        if parent_resSeq is not None:
          modification_to_parent[res.resSeq] = parent_resSeq
    # Second pass: calculate SASA value
    for atom in structure.topology.atoms:
      if is_single_pdb and atom.index not in glycan_atom_indices:
        continue
      res = atom.residue
      res_seq = res.resSeq
      # If this is a modification residue, get its parent's resSeq
      if res.name in NON_MONO and res_seq in modification_to_parent:
        parent_resSeq = modification_to_parent[res_seq]
        if parent_resSeq not in mono_sasa:
          mono_sasa[parent_resSeq] = {
                'resName': residue_modifications.get(parent_resSeq, 'NAG'),  # Use IUPAC name if available
                'sasa': 0
                }
        mono_sasa[parent_resSeq]['sasa'] += sasa[0][atom.index]
        continue
      if res.name in NON_MONO:
        continue
      if res_seq not in mono_sasa:
        mono_sasa[res_seq] = {'resName': residue_modifications.get(res_seq, res.name), 'sasa': 0}
      mono_sasa[res_seq]['sasa'] += sasa[0][atom.index]  # Add SASA contribution
    pdb_file = pdb_file if isinstance(pdb_file, str) else str(p)
    sasa_values[pdb_file] = mono_sasa
  # Calculate statistics
  first_pdb = sasa_values[list(sasa_values.keys())[0]]
  stats = {resSeq: {
    'resName': first_pdb[resSeq]['resName'],
    'values': [sasa_values[pdb][resSeq]['sasa'] for pdb in sasa_values.keys()]
    } for resSeq in first_pdb}
  # Create DataFrame
  df_data = {
    'Monosaccharide_id': [], 'Monosaccharide': [],
    'SASA': [], 'Standard Deviation': [],
    'Coefficient of Variation': []
    }
  for resSeq, data in stats.items():
    values = np.array(data['values'])
    resName = data['resName']
    #if is_single_pdb and '(' not in resName:
     # continue
    df_data['Monosaccharide_id'].append(resSeq)
    df_data['Monosaccharide'].append(resName)
    if is_single_pdb:
      df_data['SASA'].append(values[0])
      df_data['Standard Deviation'].append(float('nan'))
      df_data['Coefficient of Variation'].append(float('nan'))
    else:
      mean = np.mean(values)
      df_data['SASA'].append(np.average(values, weights = weights))
      std = np.std(values)
      df_data['Standard Deviation'].append(std)
      df_data['Coefficient of Variation'].append(std / mean if mean != 0 else 0)
  df_data['SASA'] = [val * 100 for val in df_data['SASA']]  # Convert from nm² to Å²
  return pd.DataFrame(df_data)


def convert_glycan_to_class(glycan):
  """Converts monosaccharides in a glycan string to abstract classes.
  Args:
      glycan (str): IUPAC glycan sequence.
  Returns:
      str: Modified glycan string with abstracted monosaccharide classes.
  """
  MONO_CLASSES = {
    'Hex': ['Glc', 'Gal', 'Man', 'Ins', 'Galf', 'Alt', 'D-All', 'Hex'],
    'dHex': ['Fuc', 'Qui', 'Rha', 'dHex'],
    'HexA': ['GlcA', 'ManA', 'GalA', 'IdoA', 'HexA'],
    'HexN': ['GlcN', 'ManN', 'GalN', 'HexN'],
    'HexNAc': ['GlcNAc', 'GalNAc', 'ManNAc', 'HexNAc'],
    'Pen': ['Ara', 'Xyl', 'Rib', 'Lyx', 'Pen'],
    'Sia': ['Neu5Ac', 'Neu5Gc', 'Kdn', 'Sia']
    }
  MONO_MAP = {mono: class_name for class_name, monos in MONO_CLASSES.items() for mono in monos}
  CLASS_NAMES = {'Hex': 'X', 'dHex': 'dX', 'HexA': 'XA', 'HexN': 'XN', 'HexNAc': 'XNAc', 'Pen': 'Pen', 'Sia': 'Sia'}
  glycan = stemify_glycan(glycan)
  result = []
  for part in glycan.replace('[', ' [ ').replace(']', ' ] ').split(')'):
    mono = part.split('(')[0].strip()
    if mono in ['[', ']']:
      result.append(mono)
    else:
      mono_class = MONO_MAP.get(mono)
      result.append(CLASS_NAMES.get(mono_class, 'Unk') if mono_class else 'Unk')
  return ''.join(result)


def group_by_silhouette(glycan_list, mode = 'X'):
  """Groups glycans by their topological silhouette/branching pattern.
  Args:
      glycan_list (list): List of IUPAC glycan sequences.
      mode (str): 'X' for simple abstraction or 'class' for detailed classes.
  Returns:
      pd.DataFrame: DataFrame of glycans annotated with silhouette and group.
  """
  silhouettes, topo_groups = {}, {}
  for glycan in glycan_list:
    if mode == 'X':
      pattern = re.sub(r'[A-Za-z0-9]+(?:\([^\)]+\))?', 'X', glycan)
    else:
      pattern = convert_glycan_to_class(glycan)
    if pattern not in topo_groups:
      topo_groups[pattern] = len(topo_groups)
    silhouettes[glycan] = {
        'silhouette': pattern,
        'topological_group': topo_groups[pattern]
        }
  df = pd.DataFrame.from_dict(silhouettes, orient = 'index')
  df.index.name = 'glycan'
  df.reset_index(inplace = True)
  return df.sort_values('topological_group')


def compute_merge_SASA_flexibility(glycan, mode = 'weighted', stereo = None, my_path = None) :
  """Merges SASA and flexibility data for a glycan structure.
  Args:
      glycan (str): IUPAC glycan sequence.
      mode (str, optional): 'standard', 'amplify', or 'weighted' for flexibility calculation.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      pd.DataFrame: Combined table with SASA and flexibility (as RMSF) metrics.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  sasa = get_sasa_table(glycan, stereo = stereo, my_path = my_path)
  if my_path is not None and (isinstance(my_path, str) and "." in my_path) or (isinstance(my_path, Path)):
    df, interaction_dict = get_annotation(glycan, my_path)
    pdb_path = interaction_dict.get('__pdb_path__')
    linker_res_num = None
    linker_res_name = None
    if pdb_path is not None:
      glycan_residues = df[df['monosaccharide'].isin(set(map_dict.keys()))]
      if len(glycan_residues) > 0:
        min_glycan_res = glycan_residues['residue_number'].min()
        first_glycan = df[df['residue_number'] == min_glycan_res]
        c1_atoms = first_glycan[first_glycan['atom_name'] == 'C1']
        if len(c1_atoms) > 0:
          c1_coord = c1_atoms.iloc[0][['x', 'y', 'z']].values.astype(float)
          chain = first_glycan['chain_id'].iloc[0]
          with open(pdb_path, 'r') as f:
            all_lines = f.readlines()
            lines = [line for line in all_lines if line.startswith('ATOM')]
          for line in lines:
            if len(line) < 54:
              continue
            res_name = line[17:20].strip()
            if res_name in {'ASN', 'SER', 'THR', 'HYP'}:
              atom_name = line[12:16].strip()
              if atom_name in {'ND2', 'OG', 'OG1'}:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                distance = np.linalg.norm(c1_coord - np.array([x, y, z]))
                if distance < 1.6:
                  linker_res_num = int(line[22:26].strip())
                  linker_res_name = res_name
                  break
          if linker_res_num is not None:
            linker_atoms = []
            for line in lines:
              if len(line) >= 54 and int(line[22:26].strip()) == linker_res_num:
                linker_atoms.append({
                  'atom_number': int(line[6:11].strip()),
                  'atom_name': line[12:16].strip(),
                  'monosaccharide': linker_res_name,
                  'chain_id': line[21:22].strip(),
                  'residue_number': linker_res_num,
                  'x': float(line[30:38]),
                  'y': float(line[38:46]),
                  'z': float(line[46:54]),
                  'occupancy': float(line[54:60]) if len(line) > 60 else 1.0,
                  'temperature_factor': float(line[60:66]) if len(line) > 66 else 0.0,
                  'element': line[76:78].strip() if len(line) > 78 else '',
                  'IUPAC': linker_res_name
                })
            if linker_atoms:
              linker_df = pd.DataFrame(linker_atoms)
              df = pd.concat([df, linker_df], ignore_index = True)
              structure = md.load(pdb_path)
              linker_chain_id = linker_df['chain_id'].iloc[0]
              linker_atom_indices = [atom.index for atom in structure.topology.atoms if atom.residue.resSeq == linker_res_num and atom.residue.chain.chain_id == linker_chain_id]
              if linker_atom_indices:
                sasa_raw = md.shrake_rupley(structure, mode = 'atom')
                linker_sasa = sum(sasa_raw[0][idx] for idx in linker_atom_indices) * 100
                linker_sasa_row = pd.DataFrame({'Monosaccharide_id': [linker_res_num], 'Monosaccharide': [linker_res_name], 'SASA': [linker_sasa], 'Standard Deviation': [float('nan')], 'Coefficient of Variation': [float('nan')]})
                sasa = pd.concat([sasa, linker_sasa_row], ignore_index = True)
    flexibility = df.groupby('residue_number')['temperature_factor'].mean()
    flexibility_rmsf = np.sqrt(3 * flexibility / (8 * np.pi**2))
    monosaccharides = df.drop_duplicates('residue_number').set_index('residue_number')['IUPAC']
    flex_df = pd.DataFrame({'Monosaccharide_id': df.residue_number.unique(), 'Monosaccharide': monosaccharides, 'flexibility': flexibility_rmsf}).reset_index(drop = True)
    flex_df['torsion_flexibility'] = np.nan
  else:
    pdbs = fetch_pdbs(glycan, stereo = stereo, my_path = my_path)
    if not isinstance(pdbs[0], tuple):
      flex = (inter_structure_variability_table(glycan, stereo = stereo, mode = mode, my_path = my_path)).mean()
      conversion_factor = np.sqrt(np.pi/2)  # converts mean absolute deviation to standard deviation
      flex_rmsf = {monosac: value * conversion_factor for monosac, value in flex.items()}
      flex_df = pd.DataFrame(sorted(flex_rmsf.items(), key = lambda x: x[1]), columns = ['Monosaccharide_id_Monosaccharide', 'flexibility'])
      flex_df['Monosaccharide_id'] = flex_df['Monosaccharide_id_Monosaccharide'].str.split('_').str[0].astype(int)
      flex_df['Monosaccharide'] = flex_df['Monosaccharide_id_Monosaccharide'].str.split('_', n = 1).str[1]
      torsion_flex_dict = calculate_torsion_flexibility_per_residue(glycan, mode = mode, stereo = stereo, my_path = my_path)
      flex_df['torsion_flexibility'] = flex_df['Monosaccharide_id'].map(torsion_flex_dict)
    else:
      ex_df = pdbs[0][0]
      flexibility = pd.concat([df.groupby('residue_number')['temperature_factor'].mean() for df, inty in pdbs])
      flexibility = flexibility.groupby(flexibility.index).mean()
      flexibility_rmsf = np.sqrt(3 * flexibility / (8 * np.pi**2))
      monosaccharides = ex_df.drop_duplicates('residue_number').set_index('residue_number')['IUPAC']
      flex_df = pd.DataFrame({'Monosaccharide_id': ex_df.residue_number.unique(), 'Monosaccharide': monosaccharides, 'flexibility': flexibility_rmsf}).reset_index(drop = True)
      flex_df['torsion_flexibility'] = np.nan
  if sasa.empty:
    return flex_df
  merged = pd.merge(sasa, flex_df[['Monosaccharide_id', 'flexibility', 'torsion_flexibility']], on = 'Monosaccharide_id', how = 'outer')
  mask = merged['Monosaccharide'].isna()
  if mask.any():
    linker_info = flex_df[flex_df['Monosaccharide_id'].isin(merged.loc[mask, 'Monosaccharide_id'])][['Monosaccharide_id', 'Monosaccharide']]
    merged.loc[mask, 'Monosaccharide'] = merged.loc[mask, 'Monosaccharide_id'].map(linker_info.set_index('Monosaccharide_id')['Monosaccharide'])
  return merged


def compute_merge_SASA_flexibility_OH(glycan, mode = 'weighted', stereo = None, my_path = None):
  """Merges SASA, flexibility, and OH orientation data for a glycan structure."""
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in ['GlcNAc', 'Glc', 'Xyl']) else 'alpha'
  merged_df = compute_merge_SASA_flexibility(glycan, mode = mode, stereo = stereo, my_path = my_path)
  if merged_df.empty:
    return merged_df
  try:
    analysis = get_functional_group_analysis(glycan, stereo = stereo, my_path = my_path)
    if 'error' not in analysis:
      oh_groups = analysis['functional_groups']['oh_groups']
      residue_angles = analysis['residue_angles']
      residue_oh = {}
      for oh in oh_groups:
        res_id = oh['residue']
        if res_id not in residue_oh:
          residue_oh[res_id] = {'equatorial_oh': 0, 'axial_oh': 0, 'parallel_oh_pairs': 0, 'perpendicular_oh_pairs': 0}
        if oh.get('equatorial', False):
          residue_oh[res_id]['equatorial_oh'] += 1
        if oh.get('axial', False):
          residue_oh[res_id]['axial_oh'] += 1
      # Calculate pair counts per residue
      for res_id, angles in residue_angles.items():
        if res_id in residue_oh:
          residue_oh[res_id]['parallel_oh_pairs'] = sum(1 for angle in angles if angle < 30 or angle > 150)
          residue_oh[res_id]['perpendicular_oh_pairs'] = sum(1 for angle in angles if 60 < angle < 120)
      oh_data = []
      for _, row in merged_df.iterrows():
        res_id = row['Monosaccharide_id']
        if res_id in residue_oh:
          oh_data.append(residue_oh[res_id])
        else:
          oh_data.append({'equatorial_oh': 0, 'axial_oh': 0, 'parallel_oh_pairs': 0, 'perpendicular_oh_pairs': 0})
      for key in ['equatorial_oh', 'axial_oh', 'parallel_oh_pairs', 'perpendicular_oh_pairs']:
        merged_df[key] = [d[key] for d in oh_data]
  except:
    merged_df['equatorial_oh'] = 0
    merged_df['axial_oh'] = 0
    merged_df['parallel_oh_pairs'] = 0
    merged_df['perpendicular_oh_pairs'] = 0
  return merged_df


def map_data_to_graph(computed_df, interaction_dict, ring_conf_df = None, torsion_df = None):
  """Creates a NetworkX graph with node-level structural data including OH orientations."""
  edges = {(int(k.split('_')[0]), int(v.split('_')[0])) for k, values in interaction_dict.items() if isinstance(values, (list, tuple, set)) for v in values if k.split('_')[0] != v.split('_')[0]}
  G = nx.Graph()
  G.add_edges_from(edges)
  ring_conf_map = {}
  if ring_conf_df is not None:
    for _, row in ring_conf_df.iterrows():
      ring_conf_map[row['residue']] = {
        'Q': row['Q'],
        'theta': row['theta'],
        'conformation': row['conformation']
      }
  torsion_map = {}
  if torsion_df is not None:
    for _, row in torsion_df.iterrows():
      res_nums = [match.group() for match in re.finditer(r'(\d+)(?=_)', row['linkage'])]
      edge_key = tuple(sorted([int(res_nums[0]), int(res_nums[1])]))
      torsion_map[edge_key] = {
        'phi_angle': row['phi'],
        'psi_angle': row['psi'],
        'omega_angle': row['omega']
      }
  for _, row in computed_df.iterrows():
    node_id = row['Monosaccharide_id']
    attrs = {}
    attrs['Monosaccharide'] = row.get('Monosaccharide', node_id)
    for col in ['SASA', 'flexibility', 'torsion_flexibility', 'equatorial_oh', 'axial_oh', 'parallel_oh_pairs', 'perpendicular_oh_pairs']:
      if col in row:
        attrs[col] = row[col]
    if ring_conf_map and node_id in ring_conf_map:
      attrs.update(ring_conf_map[node_id])
    G.add_node(node_id, **attrs)
  for edge_key, torsion_data in torsion_map.items():
    if edge_key in G.edges():
      nx.set_edge_attributes(G, {edge_key: torsion_data})
  return G


def remove_and_concatenate_labels(graph):
  """Processes a graph by removing odd-indexed nodes and concatenating labels.
  Args:
      graph (nx.Graph): NetworkX graph object.
  Returns:
      nx.Graph: Modified graph with simplified structure.
  """
  graph = graph.copy()
  nodes_to_remove = []  # List to store nodes that need to be removed
  # Iterate through nodes in sorted order to ensure proper handling
  for node in sorted(graph.nodes):
    if node % 2 == 1:
      # When removing a node, look for who points TO it and where it points TO
      predecessors = list(graph.predecessors(node))
      successors = list(graph.successors(node))
      # Connect each node that pointed to this one to each node this one pointed to
      for pred in predecessors:
        for succ in successors:
          graph.add_edge(pred, succ)
      # Handle label concatenation
      predecessor = node - 1
      if predecessor in graph.nodes:
        pred_label = graph.nodes[predecessor].get("string_labels", "")
        current_label = graph.nodes[node].get("string_labels", "")
        graph.nodes[predecessor]["string_labels"] = f"{pred_label}({current_label})"
      nodes_to_remove.append(node)
  # Remove the odd-indexed nodes after processing
  graph.remove_nodes_from(nodes_to_remove)
  return graph


def trim_gcontact(G_contact):
  """Removes node 1 (-R terminal) from glycontact graph and connects its neighbors.
  Args:
      G_contact (nx.Graph): Glycontact graph.
  Returns:
      None: Modifies graph in-place.
  """
  # Remove node 1 which corresponds to -R, absent from G_work
  if 1 in G_contact and G_contact.nodes[1].get("Monosaccharide") == "-R":
    neighbors = list(G_contact.neighbors(1))  # Get the neighbors of node 1
    if len(neighbors) > 1:  # If node 1 has more than one neighbor
      for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
          G_contact.add_edge(neighbors[i], neighbors[j])  # Add edge between neighbors
    G_contact.remove_node(1)  # Remove node 1


def compare_graphs_with_attributes(G_contact, G_work):
  """Performs attribute-aware isomorphism check between two glycan graphs.
  Args:
      G_contact (nx.Graph): Glycontact graph.
      G_work (nx.Graph): Glycowork graph.
  Returns:
      dict: Mapping between node indices or empty dict if not isomorphic.
  """
  # Define a custom node matcher
  def node_match(node_attrs1, node_attrs2):
    # Ensure 'string_labels' in G is part of 'Monosaccharide' in G2
    return (
        'string_labels' in node_attrs1 and 'Monosaccharide' in node_attrs2
        and node_attrs1['string_labels'] in node_attrs2['Monosaccharide']
        )
  # Create an isomorphism matcher with the custom node matcher
  matcher = nx.isomorphism.GraphMatcher(G_work.to_undirected(), G_contact, node_match = node_match)
  mapping_dict = {} # format= gcontact_index: gwork_index
  if matcher.is_isomorphic():  # Check if the graphs are isomorphic
    # Extract the mapping of nodes
    mapping = matcher.mapping
    for node_g, node_g2 in mapping.items():
      mapping_dict[node_g2] = node_g
  else:
    print("The graphs are not isomorphic with the given attribute constraints.")
  return mapping_dict


def create_glycontact_annotated_graph(glycan: str, mapping_dict, g_contact, libr = None) -> nx.Graph:
  """Creates a glycowork graph annotated with glycontact structural data.
  Args:
      glycan (str): IUPAC glycan sequence.
      mapping_dict (dict): Node mapping from compare_graphs_with_attributes.
      g_contact (nx.Graph): Glycontact graph with structural attributes.
      libr (dict, optional): Custom library for glycan_to_nxGraph.
  Returns:
      nx.Graph: Annotated glycowork graph with combined information.
  """
  glycowork_graph = glycan_to_nxGraph(glycan, libr = libr).copy()
  original_labels = {node: data.get('labels', None) for node, data in glycowork_graph.nodes(data = True)}
  node_attributes = {node: g_contact.nodes[node] for node in g_contact.nodes}
  # Map attributes to the glycowork graph nodes
  flex_attribute_mapping = {
      mapping_dict[gcontact_node]: attributes
      for gcontact_node, attributes in node_attributes.items()
      if gcontact_node in mapping_dict
      }
  # Assign the mapped attributes to the glycowork graph
  nx.set_node_attributes(glycowork_graph, flex_attribute_mapping)
  for node, label in original_labels.items():
    if label is not None:
      glycowork_graph.nodes[node]['labels'] = label
  # Map torsion angles to linkage nodes
  edge_attributes = nx.get_edge_attributes(g_contact, 'phi_angle')
  for (u, v), phi in edge_attributes.items():
    # Find the linkage node between these monosaccharides in glycowork_graph
    u_mapped = mapping_dict[u]
    v_mapped = mapping_dict[v]
    # Find the node that represents the linkage between u_mapped and v_mapped
    linkage_node = min(u_mapped, v_mapped) + 1
    glycowork_graph.nodes[linkage_node].update({
        'phi_angle': g_contact[u][v]['phi_angle'],
        'psi_angle': g_contact[u][v]['psi_angle'],
        'omega_angle': g_contact[u][v]['omega_angle']
        })
  return glycowork_graph


def get_structure_graph(glycan, stereo = None, libr = None, example_path = None, sasa_flex_path = None, my_path = None, skip_sasa = False):
  """Creates a complete annotated structure graph for a glycan.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      libr (dict, optional): Custom library for glycan_to_nxGraph.
      example_path (str, optional): Path to a specific PDB, used for torsion angles and conformations.
      sasa_flex_path (str, optional): Path to a specific PDB, used for SASA/flexibility.
      my_path (Path, optional): Custom path to PDB folder
      skip_sasa (bool, optional): Whether to skip SASA/flexibility calculation
  Returns:
      nx.Graph: Fully annotated structure graph with all available properties.
  """
  glycan = canonicalize_iupac(glycan)
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  if not skip_sasa:
    sasa_flex_path = sasa_flex_path if sasa_flex_path else my_path
    merged = compute_merge_SASA_flexibility_OH(glycan, mode = 'weighted', stereo = stereo, my_path = sasa_flex_path)
  example = example_path if example_path is not None else get_example_pdb(glycan, stereo = stereo, my_path = my_path)
  res, datadict = get_annotation(glycan, example, threshold = 3.5)
  if skip_sasa:
    if len(res) == 0:
      return None
    iupac_map = res.drop_duplicates('residue_number').set_index('residue_number')['IUPAC']
    merged = pd.DataFrame({
      'Monosaccharide_id': iupac_map.index,
      'Monosaccharide': iupac_map.values,
      'SASA': 1.0,
      'flexibility': 0.0,
      'torsion_flexibility': 0.0,
    })
  ring_conf = get_ring_conformations(res)
  torsion_angles = get_glycosidic_torsions(res, datadict)
  G_contact = map_data_to_graph(merged, datadict, ring_conf_df = ring_conf, torsion_df = torsion_angles)
  G_work = glycan_to_nxGraph(glycan)
  G_work = remove_and_concatenate_labels(G_work)
  trim_gcontact(G_contact)
  m_dict = compare_graphs_with_attributes(G_contact, G_work)
  return create_glycontact_annotated_graph(glycan, mapping_dict = m_dict, g_contact = G_contact, libr = libr)


def check_graph_content(G) :
  """Prints node and edge information from a graph for inspection.
  Args:
      G (nx.Graph): NetworkX graph object.
  Returns:
      None: Prints information to console.
  """
  print("Graph Nodes and Their Attributes:")
  for node, attrs in G.nodes(data = True):
    print(f"Node {node}: {attrs}")
  print("\nGraph Edges:")
  for edge in G.edges():
    print(edge)


def extract_glycan_coords(pdb_filepath, residue_ids = None, main_chain_only = False):
  """Extracts coordinates of glycan residues from a PDB file.
  Args:
      pdb_filepath (str): Path to PDB file.
      residue_ids (list, optional): List of residue numbers to extract.
      main_chain_only (bool): If True, extracts only main chain atoms.
  Returns:
      tuple: (coordinates_array, atom_labels).
  """
  df = extract_3D_coordinates(pdb_filepath) if not isinstance(pdb_filepath, pd.DataFrame) else pdb_filepath
  if residue_ids:
    df = df[df['residue_number'].isin(residue_ids)]
  # Get common atoms present in most glycans
  if main_chain_only:
    common_atoms = {'C1', 'C2', 'C3', 'C4', 'C5', 'O5'}
    df = df[df['atom_name'].isin(common_atoms)]
  else:
    df = df[~df['atom_name'].str.startswith('H')]
  coords = df[['x', 'y', 'z']].to_numpy()
  residue_numbers = df['residue_number'].values
  monosaccharides = df['monosaccharide'].values
  atom_names = df['atom_name'].values
  atom_labels = [f"{r}_{m}_{a}" for r, m, a in zip(residue_numbers, monosaccharides, atom_names)]
  return coords, atom_labels


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


def calculate_torsion_angle(coords: List[List[float]]) -> float:
  """Calculate torsion angle from 4 xyz coordinates.
  Args:
    coords (list): List of 4 [x,y,z] coordinates
  Returns:
    float: Torsion angle in degrees
  """
  p = [np.array(p, dtype = float) for p in coords]
  v = [p[1] - p[0], p[2] - p[1], p[3] - p[2]]
  n1, n2 = np.cross(v[0], v[1]), np.cross(v[1], v[2])
  n1 /= np.linalg.norm(n1)
  n2 /= np.linalg.norm(n2)
  return np.degrees(np.arctan2(
      np.dot(np.cross(n1, n2), v[1] / np.linalg.norm(v[1])),
      np.dot(n1, n2)
      ))


def get_glycosidic_torsions(df_or_glycan, interaction_dict_or_pdb_path = None):
  """Calculate phi/psi/omega torsion angles for all glycosidic linkages in structure.
  Args:
    df_or_glycan: Either a pd.DataFrame with PDB atomic coordinates OR a glycan string (IUPAC)
    interaction_dict_or_pdb_path: Either a dict of glycosidic linkages OR a PDB file path (when first arg is glycan)
  Returns:
    pd.DataFrame: Phi/psi angles for each linkage
  """
  if isinstance(df_or_glycan, str):
    glycan = df_or_glycan
    pdb_path = interaction_dict_or_pdb_path
    if pdb_path is None:
      raise ValueError("When providing glycan as string, pdb_path must be provided")
    df, interaction_dict = get_annotation(glycan, pdb_path)
  else:
    df = df_or_glycan
    interaction_dict = interaction_dict_or_pdb_path
  if isinstance(interaction_dict, tuple):
      return pd.DataFrame()
  results = []
  pdb_path = interaction_dict.get('__pdb_path__')
  if pdb_path is not None:
    glycan_residues = df[df['monosaccharide'].isin(set(map_dict.keys()))]
    if len(glycan_residues) > 0:
      min_glycan_res = glycan_residues['residue_number'].min()
      first_glycan = df[df['residue_number'] == min_glycan_res]
      c1_atoms = first_glycan[first_glycan['atom_name'] == 'C1']
      if len(c1_atoms) > 0:
        c1_coord = c1_atoms.iloc[0][['x', 'y', 'z']].values.astype(float)
        chain = first_glycan['chain_id'].iloc[0]
        with open(pdb_path, 'r') as f:
          all_lines = f.readlines()
          lines = [line for line in all_lines if line.startswith('ATOM')]
        for line in lines:
          if len(line) < 54:
            continue
          line_chain = line[21:22].strip()
          res_name = line[17:20].strip()
          if res_name in {'ASN', 'SER', 'THR', 'HYP'}:
            atom_name = line[12:16].strip()
            if atom_name in {'ND2', 'OG', 'OG1'}:
              x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
              distance = np.linalg.norm(c1_coord - np.array([x, y, z]))
              if distance < 1.6:
                res_num = int(line[22:26].strip())
                linker_res_lines = [l for l in lines if len(l) >= 54 and int(l[22:26].strip()) == res_num]
                atoms = {}
                for lline in linker_res_lines:
                  latom = lline[12:16].strip()
                  if latom in {'CA', 'CB', 'CG', 'ND2', 'OG', 'OG1'}:
                    atoms[latom] = np.array([float(lline[30:38]), float(lline[38:46]), float(lline[46:54])])
                if len(atoms) < 3:
                  continue
                o5_coord = first_glycan[first_glycan['atom_name'] == 'O5'].iloc[0][['x', 'y', 'z']].values.astype(float)
                if res_name == 'ASN' and all(k in atoms for k in ['CA', 'CB', 'CG', 'ND2']):
                  coords_phi = [atoms['CA'], atoms['CB'], atoms['CG'], atoms['ND2']]
                  coords_psi = [atoms['CB'], atoms['CG'], atoms['ND2'], c1_coord]
                elif all(k in atoms for k in ['CA', 'CB']) and ('OG' in atoms or 'OG1' in atoms):
                  og = atoms.get('OG', atoms.get('OG1'))
                  coords_phi = [atoms['CA'], atoms['CB'], og, c1_coord]
                  coords_psi = [atoms['CB'], og, c1_coord, o5_coord]
                else:
                  continue
                results.append({'linkage': f"{res_name}{res_num}-{min_glycan_res}_{first_glycan['monosaccharide'].iloc[0]}", 'phi': round(calculate_torsion_angle(coords_phi), 2), 'psi': round(calculate_torsion_angle(coords_psi), 2),
                                'omega': np.nan, 'anomeric_form': 'linker', 'position': 0})
                break
  for donor_key, linkage_info in interaction_dict.items():
    if donor_key == '__pdb_path__':
      continue
    if not any('_(' in link for link in linkage_info):
      continue
    linkage_str = linkage_info[0]
    match = re.match(r'\d+_\(([\w])(\d+)-(\d+)\)', linkage_str)
    if not match:
      continue
    aform, pos = match.group(1), int(match.group(3))
    donor_res = int(donor_key.split('_')[0])
    acceptor_id = interaction_dict[linkage_str][0]
    acceptor_res = int(acceptor_id.split('_')[0])
    if df[df['residue_number'] == acceptor_res]['monosaccharide'].iloc[0] == 'ROH':
      continue
    donor = df[df['residue_number'] == donor_res]
    acceptor = df[df['residue_number'] == acceptor_res]
    # Special handling for sialic acid
    if any(mono in donor_key for mono in {'SIA', 'NGC', '0KN'}):
      o5_name = 'O6'  # In sialic acid, O5 is actually O6
      c1_name = 'C2'  # Use C2 instead of C1 for sialic acid
    elif any(mono in donor_key for mono in {'FRU', '1CU', '0CU', '4CD', '1CD'}):
      o5_name = 'O5'
      c1_name = 'C2'
    else:
      o5_name = 'O5'
      c1_name = 'C1'  # Normal C1 for other residues
    o_pos = f'O{pos}'
    coords_phi = [
        donor[donor['atom_name'] == o5_name].iloc[0][['x', 'y', 'z']].values.astype(float),
        donor[donor['atom_name'] == c1_name].iloc[0][['x', 'y', 'z']].values.astype(float),
        acceptor[acceptor['atom_name'] == o_pos].iloc[0][['x', 'y', 'z']].values.astype(float),
        acceptor[acceptor['atom_name'] == f'C{pos}'].iloc[0][['x', 'y', 'z']].values.astype(float)
    ]
    has_c6 = not acceptor[acceptor['atom_name'] == 'C6'].empty
    next_c = pos + 1 if (pos < 6 and has_c6) or (pos < 5 and not has_c6) else 1
    coords_psi = [coords_phi[1], coords_phi[2], coords_phi[3], acceptor[acceptor['atom_name'] == f'C{next_c}'].iloc[0][['x', 'y', 'z']].values.astype(float)]
    # Calculate omega angle for 1/2-6 linkages
    if pos == 6:
      try:
        coords_omega = [
            coords_phi[2],  # O6
            coords_phi[3],  # C6
            acceptor[acceptor['atom_name'] == 'C5'].iloc[0][['x', 'y', 'z']].values.astype(float),
            acceptor[acceptor['atom_name'] == 'O5'].iloc[0][['x', 'y', 'z']].values.astype(float)]
      except (IndexError, KeyError):
        coords_omega = []
    else:
      coords_omega = []
    results.append({
        'linkage': f"{donor_key}-{acceptor_id}",
        'phi': round(calculate_torsion_angle(coords_phi), 2),
        'psi': round(calculate_torsion_angle(coords_psi), 2),
        'omega': round(calculate_torsion_angle(coords_omega), 2) if coords_omega else np.nan,
        'anomeric_form': aform,
        'position': pos
        })
  return pd.DataFrame(results)


def calculate_ring_pucker(df: pd.DataFrame, residue_number: int) -> Dict:
  """Calculate ring puckering parameters for a monosaccharide using the Cremer-Pople method.
  Args:
    df (pd.DataFrame): DataFrame with PDB coordinates
    residue_number (int): Residue number to analyze
  Returns:
    dict: Dictionary with puckering parameters
  """
  residue = df[df['residue_number'] == residue_number]
  mono_type = residue['monosaccharide'].iloc[0]
  is_l_sugar = mono_type in {'FUC', 'RAM', 'ARA'}
  # Get ring atoms based on monosaccharide type
  iupac_type = residue['IUPAC'].iloc[0]
  is_sialic = any(x in iupac_type for x in {'Neu', 'Kdn'})
  is_furanose = any(x in iupac_type for x in {'Araf', 'Galf', 'Fruf'})
  if is_sialic:  # 9-atom sialic acid rings
    ring_atoms = ['C2', 'C3', 'C4', 'C5', 'C6', 'O6']
  elif is_furanose:  # 5-membered furanose rings
    ring_atoms = ['C1', 'C2', 'C3', 'C4', 'O4']
  else:  # Standard 6-membered pyranose rings
    ring_atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5']
  # Extract coordinates of ring atoms
  coords = []
  for atom in ring_atoms:
    atom_data = residue[residue['atom_name'] == atom]
    if atom_data.empty:
      raise ValueError(f"Missing ring atom {atom} in residue {residue_number}")
    coords.append(atom_data[['x', 'y', 'z']].values[0].astype(float))
  coords = np.array(coords)
  # Calculate geometrical center
  center = np.mean(coords, axis = 0)
  n = len(ring_atoms)
  # Define normal vector to mean plane
  z_vector = np.zeros(3)
  for j in range(n):
    k = (j + 1) % n
    z_vector += np.cross(coords[j] - center, coords[k] - center)
  z_vector /= np.linalg.norm(z_vector)
  # Project atoms onto mean plane
  y_vector = coords[0] - center
  y_vector -= np.dot(y_vector, z_vector) * z_vector
  y_vector /= np.linalg.norm(y_vector)
  x_vector = np.cross(y_vector, z_vector)
  # Calculate puckering coordinates
  zj = np.array([np.dot(coord - center, z_vector) for coord in coords])
  # Calculate puckering amplitudes
  qm = np.zeros(n//2)
  phi = np.zeros(n//2)
  for m in range(n//2):
    qm_sin, qm_cos = 0, 0
    for j in range(n):
      angle = 2 * np.pi * (m + 1) * j / n
      qm_sin += zj[j] * np.sin(angle)
      qm_cos += zj[j] * np.cos(angle)
    qm[m] = np.sqrt(qm_sin ** 2 + qm_cos ** 2) * (2 / n)
    phi[m] = np.degrees(np.arctan2(qm_sin, qm_cos)) % 360
  # Total puckering amplitude
  Q = np.sqrt(np.sum(qm ** 2))
  conformation = "Unknown"
  # Phase angle θ
  if is_furanose:  # For 5-membered rings, there are only two puckering parameters (q2 and φ2)
    q2 = qm[0]  # First (and only meaningful) puckering coordinate
    # For furanoses, use φ2 to determine conformation
    theta = phi[0]
    # Envelope and twist conformations for furanoses
    if q2 < 0.1:  # Almost planar
      conformation = "Planar"
    else:  # Determine envelope or twist based on the phase angle
      envelope_types = {0: "C3-endo", 72: "C4-endo", 144: "O4-endo", 216: "C1-endo", 288: "C2-endo"}
      twist_types = {36: "3T4", 108: "4TO", 180: "OT1", 252: "1T2", 324: "2T3"}
      if abs(theta % 72) < 18:  # Within 18° of a multiple of 72°
        # Envelope conformations (E)
        closest_angle = round(theta / 72) * 72
        conformation = envelope_types.get(closest_angle % 360, "")
      else:
        # Twist conformations (T)
        closest_angle = round((theta - 36) / 72) * 72 + 36
        conformation = twist_types.get(closest_angle % 360, "")
  else:
    if is_sialic:
      # For sialic acid rings (using second largest amplitude)
      theta = np.degrees(np.arccos(qm[2] / Q))
      # Adjust the phase angle calculation for the larger ring
      phi = [np.degrees(np.arctan2(
        np.sum([zj[j] * np.sin(2 * np.pi * (m + 1) * j / n) for j in range(n)]),
        np.sum([zj[j] * np.cos(2 * np.pi * (m + 1) * j / n) for j in range(n)])
        )) % 360 for m in range(n//2)]
      # Determine conformation
      if theta < 30:  # Sialic acids typically prefer a 2C5 chair conformation
        conformation = "2C5"
      elif theta > 150:
        conformation = "5C2"  # Less common inverted chair
      elif theta < 90:
        phi_main = phi[2]
        conformation = "B2,5" if (330 <= phi_main or phi_main < 30) else "B3,O6" if (150 <= phi_main < 210) else "S3,5"
      else:
        conformation = "S3,5"  # Most common skew form
    else:
      # For 6-membered rings
      q2, q3 = qm[1], qm[2]  # Second/Third puckering coordinate
      theta = np.degrees(np.arctan2(q2, q3))
      # Determine conformation
      if theta < 45:
        conformation = "4C1" if not is_l_sugar else "1C4"
      elif theta > 135:
        conformation = "1C4" if not is_l_sugar else "4C1"
      else:
        # Check for boat/skew-boat
        boat_types = {0: "B1,4", 60: "B2,5", 120: "B3,6", 180: "B1,4", 240: "B2,5", 300: "B3,6"}
        skew_types = {30: "1S3", 90: "2S6", 150: "3S1", 210: "4S2", 270: "5S3", 330: "6S4"}
        phi_main = phi[1]  # Main pseudorotational angle
        # Find closest reference angle
        if abs(phi_main % 60) < 30:
          # Boat conformation
          closest_angle = round(phi_main / 60) * 60
          conformation = boat_types.get(closest_angle % 360, "")
        else:
          # Skew-boat conformation
          closest_angle = round((phi_main - 30) / 60) * 60 + 30
          conformation = skew_types.get(closest_angle % 360, "")
  return {
    'residue': residue_number,
    'monosaccharide': mono_type,
    'Q': round(Q, 3),
    'theta': round(theta, 2),
    'phi': [round(p, 2) for p in phi],
    'conformation': conformation
    }


def get_ring_conformations(df: pd.DataFrame, exclude_types: List[str] = ['ROH', 'MEX', 'PCX', 'SO3', 'ACX']) -> pd.DataFrame:
  """Analyze ring conformations for all residues in structure.
  Args:
    df (pd.DataFrame): DataFrame with PDB coordinates
    exclude_types (list): List of residue types to exclude
  Returns:
    pd.DataFrame: DataFrame with ring parameters for each residue
  """
  if len(df) < 1:
    return pd.DataFrame(columns = ['residue', 'monosaccharide', 'Q', 'theta', 'phi', 'conformation'])
  results = []
  residues = df.groupby('residue_number')['monosaccharide'].first()
  for res_num, mono_type in residues.items():
    if mono_type in exclude_types:
      continue
    try:
      pucker = calculate_ring_pucker(df, res_num)
      results.append(pucker)
    except ValueError as e:
      print(f"Warning: {str(e)}")
      continue
  return pd.DataFrame(results)


def df_to_pdb_content(df):
  """Convert a DataFrame containing PDB-like data to PDB file content.
  Args:
    df: DataFrame with columns matching PDB HETATM/ATOM format
  Returns:
    String containing PDB-formatted content
  """
  pdb_lines = [
    "HEADER    GLYCAN STRUCTURE                        " + datetime.datetime.now().strftime("%d-%b-%y").upper(),
    "TITLE     GLYCAN GENERATED FROM DATAFRAME",
    "REMARK    GENERATED BY DF_TO_PDB_CONTENT FUNCTION"
  ]
  record_type = "ATOM"
  for _, row in df.iterrows():
    # Format each field according to PDB format
    line = f"{record_type:<6s}{row.atom_number:>5d}  {row.atom_name:<3s} {row.monosaccharide:<4s}X{row.residue_number:>4d}    "
    line += f"{row.x:>8.3f}{row.y:>8.3f}{row.z:>8.3f}{row.occupancy:>6.2f}{row.temperature_factor:>6.2f}      SYST {row.element:<2s}"
    pdb_lines.append(line)
    last_atom_number = row.atom_number
    last_residue_name = row.monosaccharide
    last_residue_number = row.residue_number
  # Add END record
  pdb_lines.append(f"TER    {last_atom_number + 1}      {last_residue_name} X   {last_residue_number}")
  pdb_lines.append("END")
  # Join lines with newlines
  pdb_content = "\n".join(pdb_lines)
  return pdb_content


def extract_functional_groups(df):
  """Extracts hydroxyl (-OH) and C-H group coordinates and orientations from PDB coordinates."""
  oh_groups, ch_groups = [], []
  residues = df['residue_number'].unique()
  for res_num in residues:
    residue_df = df[df['residue_number'] == res_num]
    mono_type = residue_df['IUPAC'].iloc[0] if 'IUPAC' in residue_df.columns else residue_df['monosaccharide'].iloc[0]
    if any(x in mono_type for x in ['ROH', 'MEX', 'PCX', 'SO3', 'ACX']):
      continue
    carbons = residue_df[residue_df['element'] == 'C']
    for _, carbon in carbons.iterrows():
      c_coord = carbon[['x', 'y', 'z']].values.astype(float)
      c_name = carbon['atom_name']
      o_name = f"O{c_name[1:]}" if len(c_name) > 1 and c_name[1:].isdigit() else f"O{c_name[-1]}"
      oxygen = residue_df[residue_df['atom_name'] == o_name]
      if not oxygen.empty:
        o_coord = oxygen.iloc[0][['x', 'y', 'z']].values.astype(float)
        oh_vector = o_coord - c_coord
        oh_length = np.linalg.norm(oh_vector)
        if oh_length > 0:
          oh_unit = oh_vector / oh_length
          oh_groups.append({
            'residue': res_num,
            'monosaccharide': mono_type,
            'carbon_atom': c_name,
            'oxygen_atom': o_name,
            'carbon_coord': c_coord,
            'oxygen_coord': o_coord,
            'oh_vector': oh_unit,
            'oh_length': oh_length
          })
  return {'oh_groups': oh_groups, 'ch_groups': ch_groups}


def calculate_ring_normals(df, functional_groups):
  """Calculates ring normal vectors to determine OH group orientations relative to ring plane."""
  for oh_group in functional_groups['oh_groups']:
    res_num = oh_group['residue']
    residue_df = df[df['residue_number'] == res_num]
    mono_type = oh_group['monosaccharide']
    is_sialic = any(x in mono_type for x in ['Neu', 'Kdn'])
    is_furanose = 'f' in mono_type.lower()
    if is_sialic:
      ring_atoms = ['C2', 'C3', 'C4', 'C5', 'C6', 'O6']
    elif is_furanose:
      ring_atoms = ['C1', 'C2', 'C3', 'C4', 'O4']
    else:
      ring_atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5']
    ring_coords = []
    for atom_name in ring_atoms:
      atom = residue_df[residue_df['atom_name'] == atom_name]
      if not atom.empty:
        ring_coords.append(atom.iloc[0][['x', 'y', 'z']].values.astype(float))
    if len(ring_coords) >= 3:
      ring_coords = np.array(ring_coords)
      v1 = ring_coords[1] - ring_coords[0]
      v2 = ring_coords[2] - ring_coords[0]
      normal = np.cross(v1, v2)
      normal = normal / np.linalg.norm(normal)
      oh_ring_angle = np.degrees(np.arccos(np.clip(np.dot(oh_group['oh_vector'], normal), -1, 1)))
      oh_group['oh_ring_angle'] = oh_ring_angle
      oh_group['equatorial'] = 60 < oh_ring_angle < 120
      oh_group['axial'] = oh_ring_angle < 30 or oh_ring_angle > 150
  return functional_groups


def get_functional_group_analysis(glycan, stereo = None, pdb_file = None, my_path = None):
  """Complete pipeline for analyzing functional group spatial arrangements."""
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in ['GlcNAc', 'Glc', 'Xyl']) else 'alpha'
  if pdb_file is None:
    pdb_file = get_example_pdb(glycan, stereo = stereo, my_path = my_path)
  df, interaction_dict = get_annotation(glycan, pdb_file, threshold = 3.5)
  if len(df) == 0:
    return {'error': 'No structure data available'}
  functional_groups = extract_functional_groups(df)
  functional_groups = calculate_ring_normals(df, functional_groups)
  # Calculate OH pair angles per residue
  residue_angles = {}
  for oh in functional_groups['oh_groups']:
    res_id = oh['residue']
    if res_id not in residue_angles:
      residue_angles[res_id] = []
  for i, oh1 in enumerate(functional_groups['oh_groups']):
    for j, oh2 in enumerate(functional_groups['oh_groups'][i + 1:], i + 1):
      if oh1['residue'] == oh2['residue']:  # Same residue
        cos_angle = np.dot(oh1['oh_vector'], oh2['oh_vector'])
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        residue_angles[oh1['residue']].append(angle)
  return {
    'functional_groups': functional_groups,
    'residue_angles': residue_angles,
    'glycan': glycan,
    'structure_file': pdb_file
  }


def calculate_hsic(X, Y, sigma = None):
  """Calculates Hilbert-Schmidt Independence Criterion between two variables. WILL BE REPLACED WITH glycowork VERSION IN 1.8.1
  Args:
    X (array): First variable (n_samples,)
    Y (array): Second variable (n_samples,)
    sigma (float): Kernel bandwidth, uses median heuristic if None
  Returns:
    tuple: (HSIC value, p-value from permutation test)
  """
  X = np.array(X).reshape(-1, 1)
  Y = np.array(Y).reshape(-1, 1)
  n = len(X)
  if sigma is None:
    sigma = np.median(np.sqrt(np.sum((X - X.T) ** 2, axis = 1)))
  K = rbf_kernel(X, gamma = 1/(2 * sigma ** 2))
  L = rbf_kernel(Y, gamma = 1/(2 * sigma ** 2))
  H = np.eye(n) - np.ones((n, n))/n
  HSIC = np.trace(K @ H @ L @ H) / (n - 1) ** 2
  # Approximation for p-value using gamma distribution
  eigenvals_K = np.linalg.eigvals(H @ K @ H).real
  eigenvals_L = np.linalg.eigvals(H @ L @ H).real
  eigenvals_K = eigenvals_K[eigenvals_K > 1e-12]
  eigenvals_L = eigenvals_L[eigenvals_L > 1e-12]
  theta = np.mean(eigenvals_K) * np.mean(eigenvals_L)
  df = 4 * np.mean(eigenvals_K) ** 2 / np.var(eigenvals_K) if np.var(eigenvals_K) > 0 else 1
  test_stat = HSIC * (n - 1) ** 2 / theta
  p_value = 1 - chi2.cdf(test_stat, df)
  return HSIC, p_value


def analyze_torsion_torsion_correlations(glycan, stereo = None, my_path = None):
  """Analyzes correlations between all pairs of glycosidic torsion angles.
  Args:
    glycan (str): IUPAC glycan sequence
    stereo (str): Stereochemistry specification
    my_path (str): Custom path to PDB folders
  Returns:
    dict: Results containing torsion-torsion correlation matrix
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in BETA) else 'alpha'
  dfs, int_dicts = annotation_pipeline(glycan, threshold = 3.5, stereo = stereo, my_path = my_path)
  if len(dfs) < 2:
    return {'error': 'Insufficient conformations for correlation analysis'}
  torsion_data = []
  for df, int_dict in zip(dfs, int_dicts):
    if len(df) < 1:
      continue
    torsion_df = get_glycosidic_torsions(df, int_dict)
    if len(torsion_df) > 0:
      torsion_dict = {}
      for _, row in torsion_df.iterrows():
        linkage = row['linkage']
        torsion_dict[f"{linkage}_phi"] = row['phi']
        torsion_dict[f"{linkage}_psi"] = row['psi']
        if row['omega'] is not None:
          torsion_dict[f"{linkage}_omega"] = row['omega']
      torsion_data.append(torsion_dict)
  if len(torsion_data) < 2:
    return {'error': 'Insufficient torsion data for analysis'}
  all_torsions = set()
  for td in torsion_data:
    all_torsions.update(td.keys())
  torsion_matrix = pd.DataFrame(index = range(len(torsion_data)), columns = sorted(all_torsions))
  for i, td in enumerate(torsion_data):
    for torsion in all_torsions:
      torsion_matrix.loc[i, torsion] = td.get(torsion, np.nan)
  torsion_matrix = torsion_matrix.dropna(axis = 1).astype(float)
  significant_correlations = []
  correlation_matrix = pd.DataFrame(index = torsion_matrix.columns, columns = torsion_matrix.columns)
  for col1, col2 in combinations(torsion_matrix.columns, 2):
    vals1 = torsion_matrix[col1].values
    vals2 = torsion_matrix[col2].values
    hsic_val, p_val = calculate_hsic(vals1, vals2)
    correlation_matrix.loc[col1, col2] = hsic_val
    correlation_matrix.loc[col2, col1] = hsic_val
    if p_val < 0.05:
      significant_correlations.append({
        'torsion1': col1,
        'torsion2': col2,
        'hsic': hsic_val,
        'p_value': p_val
      })
  # Fill diagonal
  for col in torsion_matrix.columns:
    correlation_matrix.loc[col, col] = 1.0
  return {
    'glycan': glycan,
    'correlation_matrix': correlation_matrix.astype(float),
    'significant_correlations': significant_correlations,
    'n_conformations': len(torsion_data)
  }


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
