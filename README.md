
# ncEVE: Pathogenicity Prediction of Non-Coding Genomic Variants

## Overview

Back in 2022, I was in search of the next project in research. I learned about the work of Frazer et al., 2021 published in the paper ["Disease variant prediction with deep generative models of evolutionary data"](https://www.nature.com/articles/s41586-021-04043-8). The paper presented a way to score the pathogenicity of *coding* genomic variants that changes the sequences of amino-acid in a protein. I tried to quickly adapt this model to whole-genome sequence data, such that we can theoretically score the pathogenic effects of *non-coding* variants.


*EVE*: Fraser et al. trained a Bayesian Variational Autoencoder (VAE) to reconstruct the alignment of amino acid sequences between human and other species. **Hence, each model is trained on a single-protein's sequence alignment**. The original model focuses on coding sequences, leveraging evolutionary data to predict the impact of amino acid changes on protein function and associated disease risk.
![Figure 1: EVE framework, cited from [Fig. 1 of Frazer et al., 2021](https://www.nature.com/articles/s41586-021-04043-8/figures/1)](screenshots/EVE_fig1.png)

*ncEVE*: I worked with whole genome sequence alignment between human and 100 other species in the animal kingdom in this project (data from UCSC Genome Browser's 100-multiway). Hence, for each window of $N$-bp in the genome, we can in principle apply similar frameworks applied in Frazer et al., 2021 to the non-coding sequence data. Non-coding sequences, while not directly altering protein sequences, can have significant regulatory roles and contribute to disease risk through mechanisms affecting gene expression, splicing, and other regulatory functions.

## Approach

- **Model Adaptation**: The Bayesian VAE framework from Frazer et al. has been adapted to handle non-coding sequences.
- **Data**: The model is trained using multi-species sequence alignment data, allowing it to learn evolutionary conservation and divergence patterns in non-coding regions.
- **Pathogenicity Scoring**: The adapted model learns to score the pathogenicity of non-coding genetic variants, providing insights into potential disease-causing regulatory changes.

## Goals

- Extend the applicability of deep generative models to non-coding genomic sequences.
- Provide a tool for researchers to assess the pathogenicity of non-coding variants based on evolutionary data.
- Contribute to the understanding of non-coding regions in the genome and their role in disease.

## References

- Frazer, J., Notin, P., Dias, M., Gomez, A., Min, J. L., Brock, K., ... & Marks, D. S. (2021). Disease variant prediction with deep generative models of evolutionary data. *Nature Genetics*, 53(6), 759-768. doi:10.1038/s41588-021-00856-2

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We acknowledge Frazer et al. for their pioneering work and for providing a framework that facilitated this adaptation.
