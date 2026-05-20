# Processing and Analysis Pipeline for <br> Fluorescence Microscopy Imaging Data <br> > Single particle tracking (mRNA) and translocation through NPCs in *S. cerevisiae*

> Production-quality pipeline for automated nuclear envelope morphology analysis
> from dual-channel fluorescence microscopy with statistical rigor and FAIR
> compliance.

[![Python
3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

--------------------------------------------------------------------------------

## Overview

This pipeline processes and analyzes imaging data collected using fluorescence
microscopy into quantitative measurements of the nuclear pore complex (NPC) via
labeling of pairs of nucleoporins (Nups) with sub-pixel precision. Originally
developed for studying translocation of mRNA through NPC in yeast, the modular
architecture is designed to support diverse membrane analysis applications.

**Key Innovation:** Systematic transformation from research prototype to
production-ready, FAIR-compliant software demonstrating research software
engineering best practices.

--------------------------------------------------------------------------------

## Features

### Core Capabilities

-   ✅ **Automated NE Detection** - U-Net++ neural network segmentation with
    adaptive thresholding
-   ✅ **Sub-pixel Refinement** - spline fitting via intensity profile modeled
    using a Richards-Gaussian model fit through non-linear optimization
    (Levenberg-Marquardt algorithm)
-   ✅ **Multi-channel Registration** - Phase correlation with sub-pixel
    accuracy
-   ✅ **Statistical Rigor** - Likelihood ratio testing, AIC model selection,
    robust outlier filtering
-   ✅ **Quality Control** - Comprehensive validation with automated PDF reports
-   ✅ **Reproducible Analysis** - Complete parameter justification with
    citations

### Software Engineering

-   ✅ **Modular Architecture** - Object-oriented design with clear separation
    of concerns
-   ✅ **Checkpoint/Resume** - FOV-level checkpoints allow for distributed HPC
    computing
-   ✅ **FAIR Compliance** - Findable, Accessible, Interoperable, Reusable
-   ✅ **Production Quality** - Adaptive algorithms reduced optimization
    failures

--------------------------------------------------------------------------------

### Basic Usage

#### Configuration

``` json
{
    "pipe globals": {
        "strains": ["experiment_001"],
        "directories": {
            "imaging root": "/path/to/your/data",
            "camera root": "/path/to/calibration/images",
            "output root": "/path/to/output",
            "model root": "./trained_models"
        },
        "pixelsize": 128,
        "frame_duration": 0.02
    },
    "image processor": {
        "model_NE": "Modelweights_NE_segmentation.pt",
        "responsivity": {
            "ch1": {
                "bright": "bright_Ch1.tif",
                "dark": "dark_Ch1.tif"
            },
            "ch2": {
                "bright": "bright_Ch2.tif",
                "dark": "dark_Ch2.tif"
            }
        }
    }
}
```

#### Output / Results

Output structure:

```         
output_root/
├── responsivity/           # Camera calibration results
├── initial_fit/           # U-Net++ detection and initial splines
├── registration/          # Multi-channel alignment + PDF reports
├── refined_fit/           # Sub-pixel refined splines
└── distances/             # Dual-label distance measurements
```

--------------------------------------------------------------------------------

## Documentation

### In Development

-   Installation guide
-   Configuration Reference
-   Bibliography of statistical method justifications
-   User guide (data preparation, running pipeline, troubleshooting)
-   Methodology documentation (detailed algorithm descriptions)
-   Developer guide (architecture, contributing, testing)
-   API reference

--------------------------------------------------------------------------------

## System Requirements

Still undergoing testing on multiple systems. The [singularity
recipe](docs/recipe_yeast.txt) is provided.

### Tested On

-   **Local:** macOS (Apple Silicon M4 Max)
-   **HPC:** LSF cluster

--------------------------------------------------------------------------------

## Pipeline Workflow

### Processing Steps

1.  **Camera Calibration**
    -   Derives gain, offset, read noise per channel
    -   Enables accurate photon counting
2.  **Initial NE Detection**
    -   U-Net++ neural network segmentation (Zhou et al. 2018)
    -   Initial B-spline fitting to boundaries
    -   Segment merging for continuous envelopes
3.  **Multi-channel Registration**
    -   Phase correlation with sub-pixel precision (Guizar-Sicairos et al. 2008)
    -   Quality control filtering (MAD-based robust statistics)
    -   Drift correction and stability analysis
4.  **Spline Refinement**
    -   Richards-Gaussian profile fitting (Richards 1959; Zhang et al. 2007)
    -   Levenberg-Marquardt optimization with adaptive step sizing
    -   Likelihood ratio test for outlier detection (Smith et al. 2010)
    -   AIC-based model selection (Akaike 1974)
5.  **Bezier Bridging** (Optional)
    -   Interpolates gaps in segmented data
    -   Creates continuous, periodic splines
    -   Maintains biological curvature constraints (Zimmerberg & Kozlov 2006)
6.  **Distance Calculation**
    -   IoU-based label pairing between channels
    -   Perpendicular distance sampling (1000 points per nucleus)
    -   Statistical distance metrics (mean, std, distribution)

### Data Flow

```         
Raw Images → Camera Calibration → NE Detection → Registration
    ↓                                                    ↓
Quantified                                         Aligned Channels
Photons                                                  ↓
                                              Spline Refinement
                                                       ↓
                                              Distance Calculation
```

--------------------------------------------------------------------------------

## Scientific Background

### Research Application

This pipeline enables quantitative analysis of: - Nuclear pore complex (NPC)
spatial organization - Protein co-localization at nuclear membranes - Membrane
morphology in disease models - mRNA transport dynamics through NPCs

### Key Biological Insights Enabled

-   **Sub-pixel Localization:** protein positioning relative to membrane
-   **Statistical Rigor:** LRT-based quality control ensures data validity
-   **Reproducibility:** Complete parameter justification enables method
    validation
-   **Scalability:** Checkpoint/resume enables analysis of 100+ nuclei per
    experiment

## Project Evolution

### Research Prototype (2021-2023)

-   Proof-of-concept demonstrating algorithmic feasibility
-   Validated approach on biological data
-   Generated initial publication results

### Production Engineering (2024-PRESENT)

Systematic transformation underway for open-source release:

**Architecture:** - Monolithic scripts → Modular object-oriented design -
Hard-coded parameters → JSON configuration system - Minimal docs → Comprehensive
user/developer guides

**Statistical Rigor:** - Arbitrary thresholds → Citation-backed methods - No
justifications → peer-reviewed citations - Fixed parameters → Adaptive
optimization

**Reproducibility:** - "Works on my machine" → FAIR-compliant - No checkpointing
→ FOV-level resume capability - Manual validation → Automated quality control -
Scattered code → DRY principles

This evolution demonstrates research software engineering expertise:
transforming exploratory code into sustainable, community-ready tools.

--------------------------------------------------------------------------------

--------------------------------------------------------------------------------

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for
details.

### Key Points

-   ✅ Free for academic and commercial use
-   ✅ Modification and distribution permitted
-   ✅ Attribution required
-   ✅ No warranty (use at your own risk)

--------------------------------------------------------------------------------


## Contact

**Maintainer:** Jocelyn Petitto (formerly Tourtellotte)\
**Email:** jocelyn.petitto\@gmail.com\
**Institution:** RNA Therapeutics Institute, UMass Chan Medical School\

--------------------------------------------------------------------------------

**Last Updated:** January 2, 2026\
**Pipeline Version:** 2.0.0\
**Python Version:** 3.11+


