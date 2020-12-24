# Predicting-Molecular-Scalar-Couplings-using-SchNet
This final project for the CS-GY-9223 Deep Learning course at NYU Tandon implements SchNet, based on the paper by Schütt et al., for prediction of molecular scalar coupling constants


## 1. Introduction
The drug discovery process is one of the most challenging and expensive endeavors in biomedicine. While there are about <img src="https://render.githubusercontent.com/render/math?math=10^56"> atoms in the solar system, there are about <img src="https://render.githubusercontent.com/render/math?math=10^60"> chemical compounds with drug-like features that can be made. Since it is unfeasible for chemists to synthesize and evaluate every molecule, they’ve grown to rely on virtual screening to narrow down promising candidates. However, the challenge of searching this almost infinite space of potential molecules is the perfect substrate for deep learning techniques to improve the drug discovery process even further. While the growing number of large datasets for molecules has already enabled the creation of several useful models, the application of deep learning to drug discovery is still in its infancy. Some useful predictions that could expedite drug discovery include toxicity, ability to bind with a given protein, and quantum properties.

Researchers commonly use Nuclear Magnetic Resonance (NMR) to gain insight into a molecule’s structure and dynamics. NMR’s functionality largely depends on its ability to accurately predict scalar couplings which are the strength of magnetic interactions between pairs of atoms in a given molecule. It is possible to compute scalar couplings on an inputted 3D molecular structure using advanced quantum mechanical simulation methods such as Density Functional Theory (DFT) which approximate Schrödinger’s equation. However, these methods are limited by their high computational cost, and are therefore reserved for use on small systems, or other, less approximate, methods are adopted instead. My goal for this project was to develop a fast, reliable, and cheaper method to perform this task through the use of a graph convolutional neural network (GCN). In particular, I focused on implementing and optimizing SchNet: a novel GCN that has been shown to achieve state-of-the-art performance on quantum chemical property benchmarks. As a byproduct, I hoped to learn more about GCNs and how they could be used for chemical applications.

## 2. Literature Survey
### a. GCN motivation and basics

Many of the deep learning models designed to aid in drug discovery show improvement over traditional machine learning methods, but are limited due to two main reasons: first, they rely on hand-crafted features which prevents structural information to be learned directly from raw inputs, and, second, the existing architectures are not conducive for use on structured data such as molecules. Extraction of relevant features from images have already proven highly successful using convolutional neural networks (CNNs). Molecules can be represented as fully connected graphs in which atoms and bonds can be represented as nodes and edges, respectively. Graphs are irregularly shaped thereby making CNNs, which rely on convolution on regular grid-like structures, unsuitable for feature extraction [1]. 

Efforts have been made to generalize the convolution operation for graphs, resulting in the development of graph convolutional neural networks (GCNs). As Kipf and Welling describe in their seminal paper [2], the idea behind graph convolutional neural networks (GCNs), as shown in Fig. 1, is to perform convolutions on a graph by aggregating (through sum, average, etc) each node’s neighborhood feature vectors. This new aggregated vector is then passed through a neural network layer, and the output is the new vector representation of the node. Additional neural network layers repeat this same process, except the input is the updated vectors from the first layer. 

[Figure 1]

### b. Quantum mechanical property prediction

In 2017, Gilmer et al. [3] released a paper focusing on the specific use of neural networks for predicting quantum properties of molecules. They noted that the symmetries of atomic systems require graph neural networks that are invariant to graph isomorphism, and therefore reformulated existing models that fall into this category, including Kipf and Welling’s GCN, into a common framework called Message Passing Neural Networks (MPNNs). The “message passing” refers to the aggregation of neighborhood vector features described earlier. The MPNN that Gilmer et al. developed, called enn-s2s, managed to achieve state-of-the-art performance on an important molecular property benchmark using QM9: a dataset consisting of 130k molecules with 13 properties for each molecule as approximated by DFT. The neighborhood messages generated used both bond types and interatomic distances followed by a set2set model from Vinyals et al. [4]. 

Later on, Schutt et al. pointed out that enn-s2s was limited by the fact that atomic positions are discretized, and therefore the filter learned was also discrete which rendered it incapable of capturing the gradual positional changes of atoms [5]. In order to remedy this, Schutt et al. proposed a different method of graph convolution with continuous filters that mapped an atomic position to a corresponding filter value. This is advantageous in that it doesn't require atomic position data to lie on a grid, thereby resulting in smooth, rather than discrete energy predictions.

[Figure 2]

SchNet demonstrated superior performance over enn-s2s in predicting molecular energies and atomic forces on three different datasets. Fig. 3 provides an overview of the SchNet architecture. Molecules input into the model can be uniquely represented by a certain set of nuclear charges  and atomic positions  where  is the number of atoms. At each layer, the atoms in a given molecule are represented as a tuple of features:  with  where  and  are the number of layers, and feature maps, respectively. This representation is analogous to pixels in an image. In the embedding layer, the representation of each atom  is initialized at random using an embedding dependent on the atom type  which is optimized during training:  where  is the atom type embedding. 

Atom-wise layers, a recurring building block in this architecture, are dense layers that are applied to each representation  of atom : . These layers are responsible for the recombination of feature maps with shared weights across all atoms which allows the architecture to be scaled with respect to the size of the molecule. 

Interactions between atoms are modeled by three interaction blocks: as shown above, the sequence of atom-wise, interatomic continuous-filter convolution (cfconv), and two more atom-wise layers separated by a softplus non-linearity produces . The cfconv layer uses a radial basis function that acts as a continuous filter generator. Additionally, the residual connection between  and  allows for the incorporation of interactions between atoms and previously computed feature maps. 

## 3. Chainer Chemistry Implementation
### a. Dataset

I used the CHAMPS Scalar Coupling dataset which was provided for a Kaggle competition with a similar objective [6], and consists of the following: 
<ul type="disc">
  <li> ```train.csv``` — the training set which contains four columns: (1) the name of the molecule where the coupling constant originates, (2) and (3) the atom indices of the atom-pair which create the coupling, (4) the scalar coupling type, (5) the scalar coupling constant that we want to predict.</li>
  <li> ```scalar_coupling_contributions.csv — the scalar coupling constants in train.csv are a sum of four terms: Fermi contact, spin-dipolar, paramagnetic spin-orbit, and diamagnetic spin-orbit contributions which are contained in this file. It is organized into the following columns: (1) molecule name; (2) and (3) the atom indices of each atom-pair; (4) the type of coupling; and (5), (6), (7), and (8) are the four aforementioned terms.</li>
  <li> structures.csv — contains the x, y, and z cartesian coordinates for each atom in each molecule. It is organized into the following columns: (1) molecule name; (2) atom index; (3) atom name; and (4), (5), and (6) are the x, y, and z cartesian coordinates, respectively.</li>
</ul>


