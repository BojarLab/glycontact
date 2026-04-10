[![codecov](https://codecov.io/gh/BojarLab/glycontact/graph/badge.svg?token=4IELmnJzxt)](https://codecov.io/gh/BojarLab/glycontact)

# GlyContact: 3D Analysis of Glycan Structures

GlyContact is a Python package for retrieving, processing, and analyzing 3D glycan structures from GlycoShape/molecular dynamics, NMR, or X-ray crystallography. Indeed, while glycans are traditionally represented as linear text sequences, their branched structures and high flexibility create a complex 3D landscape that affects their biological function.

GlyContact provides a comprehensive toolkit for analyzing 3D glycan structures, enabling researchers to:

1. Visualize complex glycan structures with 3D-SNFG symbols
2. Quantify structural properties including solvent accessible surface area (SASA), motif flexibility and torsion angles
3. Analyze relationships between glycan composition/sequence/class and these structural properties
4. Compare different glycan structures that are either free, bound to lectins or covalently linked to glycoproteins 
5. Generate structural features for machine learning applications

These capabilities help bridge the gap between glycan sequence and function by revealing the critical spatial arrangements that determine molecular recognition.

## Installation

```bash
pip install git+https://github.com/lthomes/glycontact.git
```

An optional `[ml]` install is available for machine learning features:

```bash
pip install -e git+https://github.com/lthomes/glycontact.git#egg=glycontact[ml]
```

If you want to see how you can use `GlyContact` to work with glycans in lectin complexes or in glycoproteins, check out this Jupyter notebook with some pointers:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lthomes/glycontact/blob/main/examples.ipynb)

## Getting PDB files

**GlyContact** enables detailed exploration of glycan conformational properties by connecting glycan 3D structural data with their corresponding sequence information.

By default, users can input glycan sequences, which are automatically mapped to the appropriate conformer structures from **GlycoShape** through built-in functions.

To streamline structural analysis, **GlyContact** integrates directly with the GlycoShape API, allowing automatic retrieval of structures and eliminating the challenges associated with manual data downloads.

Alternatively, **GlyContact** can operate independently of GlycoShape if users prefer to supply their own structural data by providing file paths to local PDB files.




## Glycan Contact Maps

Contact maps reveal the spatial relationships between monosaccharides in a glycan structure. These maps help identify which parts of the glycan are in close proximity, providing insights into potential functional regions.


```python
from glycontact.process import get_contact_tables
# Get monosaccharide contact tables
glycan = "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"
contact_tables = get_contact_tables(glycan, level="monosaccharide")

from glycontact.visualize import draw_contact_map
# Visualize the first contact map
draw_contact_map(contact_tables[0], size=1.0)
```

    Glycan Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc not found locally. Downloading from GlycoShape...
    


    
![png](README_files/README_6_1.png)
    


## Surface Accessibility and Flexibility

The solvent-accessible surface area (SASA) and flexibility of monosaccharides are crucial determinants of glycan-protein interactions. GlyContact calculates these properties and allows visualization of their distribution across the glycan structure.


```python
from glycontact.visualize import plot_glycan_score
plot_glycan_score(glycan, attribute="SASA")
```




    
![png](README_files/README_8_0.png)
    



## Glycosidic Torsion Angles

Glycosidic torsion angles (phi/psi) determine the overall shape of glycans. GlyContact can analyze these angles across multiple structures to identify preferred conformations, similar to protein Ramachandran plots.


```python
from glycontact.visualize import ramachandran_plot
fig = ramachandran_plot("Gal(b1-4)GlcNAc")
```


    
![png](README_files/README_10_0.png)
    


## Contributing

Contributions to GlyContact are welcome! Please feel free to submit a Pull Request.

## Citation

If you use GlyContact in your research, please cite:

[Thomès et al., GlyContact analyzes glycan 3D structures at scale. Nat Commun, 16:11136, 2025](https://www.nature.com/articles/s41467-025-67590-y)


## License

This project is licensed under the MIT License—see the LICENSE file for details.


```python

```
