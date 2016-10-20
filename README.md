## About
This is an implementation of the probabilistic spatial genetic model and associated ancestry localization algorithm, **GAP**
(Geographic Ancestry Positioning), as well as the related population stratification correction procedure for genome-wide association 
studies, **SCGAP**, that is described in the paper:

1. Novel probabilistic models of spatial genetic ancestry with applications to stratification correction in genome-wide association 
studies  
    Bhaskar A., Javanmard A., Courtade T.A., Tse D.N  
    Bioinformatics, under review


## Usage
The input genotype files will be assumed to be in [plink bed format](http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml). 
The associated .fam file will be needed for ancestry localization with GAP, while both the .bim and .fam files will be needed 
for the association testing in SCGAP.


### GAP
The script gap.py can be used to run either PCA or GAP on the genotype data. GAP uses a tunable parameter τ for embedding the genotype data
in two dimensions. This parameter can be tuned using either:
* cross-validation using a training dataset with known ancestral locations
* an alternate metric that doesn't need training data, and is described in the Supplement §1.3 of [1].

The output file with extension .gap (or .pca if using PCA) will contain one line per individual with fields,
* Individual ID (second column of .fam file)
* Ancestry coordinates (2 fields)

Some examples:  
Simulate date from the isotropic covariance decay model described in [1], with genotype data and simulated ancestry coordinates written to 
simulated_data.{bed,fam,bim,loc}.
```
> python simulate_localization.py -b simulated_data -f isotropic -n 2000 -p 50000 alpha0=1 alpha1=4 alpha2=1
```
Using a 20% subset of the simulated location data as training, we can infer ancestry coordinates with GAP written to file inferred_anc.gap by doing:
```
> shuf -n 400 simulated_data.loc > training_data.loc
> python gap.py -b simulated_data -l inferred_anc -t training_data.loc --cv_folds=5 gap
```
You can also change the grid size for tuning the parameter τ (default is 20) by passing `grid_size` as a keyword argument:
```
> python gap.py -b simulated_data -l inferred_anc -t training_data.loc --cv_folds=5 grid_size=10 gap
```
With the `nz_frac` keyword argument, you can also specify a value for τ indirectly by specifying the fraction of entries of the genetic distance matrix (in increasing 
order of their values) that should be used in the inference algorithm:
```
> python gap.py -b simulated_data -l inferred_anc -t training_data.loc --cv_folds=5 nz_frac=0.10 gap
```
  
You can also use the script to perform PCA on the data. Note that we still need some data for rescaling the coordinates inferred by PCA, 
since any orthogonal transformation of the principal components is an equally valid PCA decomposition of the data. For example, to output PCA inferred 
coordinates to inferred_anc.pca,
```
> python gap.py -b simulated_data -l inferred_anc -t training_data.loc pca
```


### SCGAP
SCGAP can take the ancestry coordinates inferred by GAP, PCA or some other procedure to generate ancestry-dependent covariates, 
and incorporate them in a retrospective assocation test for continuous or discrete phenotypes. The phenotype will be read from the 6th column of the .fam file (affection status field).

The output file with extension .scgap will contain one line per SNP containing these fields,
* SNP RSID (second field of .bim file)
* log of the likelihood ratio
* intercept term κ
* multiplicative genetic risk R
* p-value
For a description of the inferred parameters κ and R, see equation (6) of [1].

Some examples:  
To generate continuous phenotypes and genotypes that are both correlated with ancestry according to the simulation procedure described in [1],
```
> python simulate_association_test.py -b simulated_data -n 2000 -p 50000 -g 0.20 -a 0.10 -e 0.70 alpha0=1 alpha1=4 alpha2=1
```
To perform ancestry inference using GAP without using any training data to rescale coordinates, and then using these inferred coordinates with the association testing procedure SCGAP,
```
> python gap.py -b simulated_data -l inferred_anc gap
> python scgap.py -b simulated_data -o simulated_data -l inferred_anc.gap
```
